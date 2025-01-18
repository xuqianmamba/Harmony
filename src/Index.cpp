#include "Index.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <omp.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

#include "IVF.h"
#include "IVFScan.hpp"
#include "heap.hpp"

#define SUB_LIST_SIZE 8ul
#define IP_SUB_RATIO 1.5
#define RECALL_TEST_RATIO 0.1
// #define SUB_STATS

namespace tribase {



Index::Index(size_t d, size_t nlist, size_t nprobe, MetricType metric, bool verbose)
    : d(d),
      nlist(nlist),
      nprobe(nprobe),
      metric(metric),
      verbose(verbose)
{
    // nlist个聚类，IVF是一个聚类里面的所有点
    lists = std::make_unique<IVF[]>(nlist);
    // code是向量，nlist个向量，每个向量d维
    centroid_codes = std::make_unique<float[]>(nlist * d);
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    // centroid_ids初始化为0~nlist-1
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);
}
// void Index::initWorkers(size_t workerCount, float* querys, size_t nq, size_t blockCount, size_t nb) {

// }

void Index::preSearch(size_t nb, size_t workerCount, size_t blockCount, size_t warmUpSearchList, size_t warmUpSearchListSize, Param* param, std::string path) {

    this->workerCount = workerCount;
    this->blockCount = blockCount;
    this->warmUpSearchList = warmUpSearchList;
    this->warmUpSearchListSize = warmUpSearchListSize;
    this->param = param;

    if(param->mode == SearchMode::ORIGINAL) {
        return;
    }

    prepareDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file " + path);
    }
    // out.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
    // out.write(reinterpret_cast<const char*>(&nlist), sizeof(size_t));
    // // out.write(reinterpret_cast<const char*>(&nprobe), sizeof(size_t));
    // out.write(reinterpret_cast<const char*>(&metric), sizeof(MetricType));
    // out.write(reinterpret_cast<const char*>(&added_opt_level), sizeof(OptLevel));
    // out.write(reinterpret_cast<const char*>(&sub_k), sizeof(size_t));
    // out.write(reinterpret_cast<const char*>(&sub_nlist), sizeof(size_t));
    // out.write(reinterpret_cast<const char*>(&sub_nprobe), sizeof(size_t));


    // for (size_t i = 0; i < nlist; i++) {
    //     if (lists[i].get_list_size() > 0) {
    //         out.write(reinterpret_cast<const char*>(&i), sizeof(size_t));
    //         lists[i].save_IVF(out);
    //     }
    // }

    MyStopWatch watch(true);
    if(param->mode == SearchMode::DIVIDE_VECTOR) {

        for(size_t rank = 1; rank <= workerCount; rank++) {
            size_t beginIVF = (rank - 1) * (nlist / workerCount); 
            size_t ivfCount = (rank == workerCount) ? (nlist - beginIVF) : (nlist / workerCount);
            auto info = BaseWorker::InitInfo(d, workerCount, nlist, nprobe, nb, beginIVF, ivfCount);
            MPI_Send(&info, sizeof(BaseWorker::InitInfo), MPI_BYTE, rank, 0, MPI_COMM_WORLD);
            if(rank == 1)
                out.write(reinterpret_cast<const char*>(&info), sizeof(BaseWorker::InitInfo));


            auto listSizes = std::make_unique<size_t[]>(ivfCount);
            for (size_t listId = beginIVF; listId < beginIVF + ivfCount; listId++) {
                IVF& list = lists[listId];
                listSizes[listId - beginIVF] = list.get_list_size();
            }
            MPI_Send(listSizes.get(), ivfCount * sizeof(size_t), MPI_BYTE, rank, 0, MPI_COMM_WORLD);
            if(rank == 1)
                out.write(reinterpret_cast<const char*>(listSizes.get()), ivfCount * sizeof(size_t));

            for (size_t listId = beginIVF; listId < beginIVF + ivfCount; listId++) {
                IVF& list = lists[listId];
                MPI_Send(list.candidate_codes.get(), list.get_list_size() * d, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
                MPI_Send(list.candidate_id.get(), list.get_list_size() * sizeof(size_t), MPI_BYTE, rank, 0, MPI_COMM_WORLD);
                // printVector(list.candidate_codes.get(), d, YELLOW);
                if(rank == 1)
                    list.save_IVF(out);
            }
            distancesHeapBuffer = vector<std::unique_ptr<float[]>>(workerCount + 1);
            labelsHeapBuffer = vector<std::unique_ptr<idx_t[]>>(workerCount + 1);
            for(int rank = 1; rank <= workerCount; rank++) {
                distancesHeapBuffer[rank] = std::make_unique<float[]>(presumeK * presumeNq);
                labelsHeapBuffer[rank] = std::make_unique<idx_t[]>(presumeK * presumeNq);
            }
        }
        MPI_Bcast(centroid_codes.get(), nlist * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(centroid_ids.get(), nlist , MPI_INT64_T, 0, MPI_COMM_WORLD);

        // out.write(reinterpret_cast<const char*>(centroid_codes.get()), nlist * d * sizeof(float));
        // out.write(reinterpret_cast<const char*>(centroid_ids.get()), nlist * sizeof(idx_t)); // 0 ~ nlist-1
        
    } else if(param->mode == SearchMode::DIVIDE_DIM) {
        assert(d % workerCount == 0);
        // assert(nq % blockCount == 0);
        presumeTotalQueryCompareSize = presumeNq / nlist * nprobe * nb * 2;
        cout << "presume" << presumeTotalQueryCompareSize << endl;
        Worker::InitInfo info = Worker::InitInfo(d, d / workerCount, workerCount, nlist, blockCount, nprobe, nb, presumeTotalQueryCompareSize / blockCount);
        // 1. Info
        MPI_Bcast(&info, sizeof(Worker::InitInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
        out.write(reinterpret_cast<const char*>(&info), sizeof(Worker::InitInfo));
        // watch.print("Info");

        // 2. listSizes, listCodes
        auto listSizes = std::make_unique<size_t[]>(nlist);
        for (size_t listId = 0; listId < nlist; listId++) {
            IVF& list = lists[listId];
            listSizes[listId] = list.get_list_size();
        }
        MPI_Bcast(listSizes.get(), nlist * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        out.write(reinterpret_cast<const char*>(listSizes.get()), nlist * sizeof(size_t));

        for (size_t i = 0; i < nlist; i++) {
            MPI_Bcast(lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            out.write(reinterpret_cast<const char*>(lists[i].candidate_id.get()), listSizes[i] * sizeof(size_t));
        }

        auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
        for (size_t i = 0; i < info.nlist; i++) {
            listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.block_dim);
        }
        for(size_t rank = 1; rank <= workerCount; rank++) {
            for (size_t listId = 0; listId < nlist; listId++) {
                IVF& list = lists[listId];
                copy_n_partial_vector(list.candidate_codes.get(), listCodesBuffer[listId].get(), info.d, info.block_dim,
                              (rank - 1) * info.block_dim, list.get_list_size());
                if(rank == 1)
                    out.write(reinterpret_cast<const char*>(listCodesBuffer[listId].get()), list.get_list_size() * info.block_dim * sizeof(float));
                // MPI_Send(listCodesBuffer[listId].get(), list.get_list_size() * info.block_dim, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
            }
        }
        for (size_t listId = 0; listId < nlist; listId++) {
            IVF& list = lists[listId];
            MPI_Bcast(list.candidate_codes.get(), list.get_list_size() * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
            // printVector(list.candidate_codes.get(), d, YELLOW);
        }

        MPI_Bcast(centroid_codes.get(), nlist * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(centroid_ids.get(), nlist , MPI_INT64_T, 0, MPI_COMM_WORLD);

        out.write(reinterpret_cast<const char*>(centroid_codes.get()), nlist * d * sizeof(float));
        out.write(reinterpret_cast<const char*>(centroid_ids.get()), nlist * sizeof(idx_t)); // 0 ~ nlist-1
        // watch.print("listSizes, listCodes");

        // if(presumeTotalQueryCompareSize > INT_MAX) {
        //     presumeTotalQueryCompareSize /= (nlist / nprobe);
        // }
        // cout << presumeTotalQueryCompareSize << "presume" << endl;
        // if(presumeTotalQueryCompareSize > threashHold) {
        //     //所有block共用一个buffer
        //     cout << "reach threashHold" << threashHold << endl;
        //     distancesForNQuerys = std::make_unique<float[]>(presumeNq / blockCount * nb);
        // } else {
        //     distancesForNQuerys = std::make_unique<float[]>(presumeTotalQueryCompareSize);
        // }
        // try {
        //     distancesForNQuerys = std::make_unique<float[]>(presumeTotalQueryCompareSize);
        // } catch (const std::bad_alloc& e) {
        //     // Handle memory allocation failure
        //     cerr << YELLOW << "block malloc distancesForNQuerys" << RESET << endl;
        //     try {
        //         distancesForNQuerys = std::make_unique<float[]>(presumeTotalQueryCompareSize / blockCount);
        //         blockMalloc = true;
        //     } catch (const std::bad_alloc& e) {
        //         cerr << RED << "bad malloc distancesForNQuerys" << RESET << endl;
        //         exit(1);
        //     }
        // }
        // distancesForNQuerys = std::make_unique<float[]>(presumeK * presumeNq);
        // watch.print("malloc blockBuffer");
        
        // distanceHeapForBlock = std::make_unique<float[]>(presumeNq * presumeK);
        // idHeapForBlock = std::make_unique<idx_t[]>(presumeNq * presumeK);
        // init_result(METRIC_L2, presumeNq * presumeK, distanceHeapForBlock.get(), idHeapForBlock.get());

        // Search顺序
        workerSearchBlockOrder = vector<vector<idx_t>>(workerCount + 1);
        for (size_t i = 1; i <= workerCount; i++) {
            workerSearchBlockOrder[i] = vector<idx_t>(blockCount);
        }
        if(param->orderOptimize) {
            if(blockCount % workerCount == 0 && param->period) {
                cout << YELLOW << "use Period Block Order" << RESET << endl;
                size_t period = workerCount;
                size_t layer = blockCount / workerCount;
                for(size_t p = 1; p <= period; p++) {
                    for(size_t order = (p - 1)* layer; order < (p - 1)* layer + layer; order++) {
                        size_t blockStart = (order % layer) * workerCount;
                        for(size_t rank = p; rank < p + workerCount; rank++) {
                            size_t block = blockStart + rank - p;
                            workerSearchBlockOrder[(rank + workerCount - 1) % workerCount + 1][order] = block;
                        }
                    }
                }
                // for (size_t i = 1; i <= workerCount; i++) {
                //     for (size_t order = 0; order < blockCount; order++) {
                //         // size_t order = (block + (gap * (i - 1))) % blockCount;
                //         size_t block = %
                //         workerSearchBlockOrder[i][order] = block;
                //     }
                // }
            } else {
                size_t gap = (blockCount + workerCount - 1) / workerCount;
                for (size_t i = 1; i <= workerCount; i++) {
                    for (size_t block = 0; block < blockCount; block++) {
                        size_t order = (block + (gap * (i - 1))) % blockCount;
                        workerSearchBlockOrder[i][order] = block;
                    }
                }
            }
        } else {
            for (size_t i = 1; i <= workerCount; i++) {
                for (size_t order = i - 1; order < i - 1 + blockCount; order++) {
                    workerSearchBlockOrder[i][order % blockCount] = order - (i - 1);
                }
            } 
        }
        for (size_t i = 1; i <= workerCount; i++) {
            MPI_Send(workerSearchBlockOrder[i].data(), blockCount, MPI_INT64_T, i, 0, MPI_COMM_WORLD);
        }
        out.write(reinterpret_cast<const char*>(workerSearchBlockOrder[1].data()), blockCount* sizeof(idx_t));
        for (size_t i = 1; i <= workerCount; i++) {
            printVector(workerSearchBlockOrder[i], BLUE);
        }

        //对于每一个块，其搜索的顺序，即一系列rank
        blockSearchedOrder = vector<vector<idx_t>>(blockCount);
        for (size_t i = 0; i < blockCount; i++) {
            blockSearchedOrder[i] = vector<idx_t>();
        }
        for (size_t order = 0; order < blockCount; order++) {
            for(size_t rank = 1; rank <= workerCount; rank++) {
                size_t block = workerSearchBlockOrder[rank][order]; 
                blockSearchedOrder[block].push_back(rank);
            }
        }
        // for (size_t i = 0; i < blockCount; i++) {
        //     printVector(blockSearchedOrder[i], BLUE);
        // }

        //每一个worker，应该将某个block传递给下一个worker的rank
        auto sendNextWorker = vector<vector<idx_t>>(workerCount + 1);
        auto recvPrevWorker = vector<vector<idx_t>>(workerCount + 1);
        for (size_t i = 1; i <= workerCount; i++) {
            sendNextWorker[i] = vector<idx_t>(blockCount, 0);
            recvPrevWorker[i] = vector<idx_t>(blockCount, 0);
        }
        for (size_t block = 0; block < blockCount; block++) {
            for (size_t rankOrder = 0; rankOrder < workerCount - 1; rankOrder++) {
                size_t senderRank = blockSearchedOrder[block][rankOrder];
                size_t recvRank = blockSearchedOrder[block][rankOrder + 1];
                sendNextWorker[senderRank][block] = recvRank;  
                recvPrevWorker[recvRank][block] = senderRank;  
            }
        }
        // for(size_t rank = 1; rank <= workerCount; rank++) {
        //     printVector(sendNextWorker[rank], BLUE);
        // }
        // for(size_t rank = 1; rank <= workerCount; rank++) {
        //     printVector(recvPrevWorker[rank], BLUE);
        // }
        for(size_t rank = 1; rank <= workerCount; rank++) {
            MPI_Send(sendNextWorker[rank].data(), blockCount, MPI_INT64_T, rank, 0, MPI_COMM_WORLD);
            MPI_Send(recvPrevWorker[rank].data(), blockCount, MPI_INT64_T, rank, 0, MPI_COMM_WORLD);
        }
        out.write(reinterpret_cast<const char*>(sendNextWorker[1].data()), blockCount* sizeof(idx_t));
        out.write(reinterpret_cast<const char*>(recvPrevWorker[1].data()), blockCount* sizeof(idx_t));
        // watch.print("ordering");
        // watch.print("preSearch");

    } else if (param->mode == SearchMode::DIVIDE_GROUP) {

        groupSearchOrder = SearchOrder(param->teamCount, param->groupCount, true);
        blockSearchOrder = SearchOrder(param->teamSize, blockCount, true);
        // groupSearchOrder.print();
        // blockSearchOrder.print();

        beginIVFs = vector<size_t>(param->teamCount + 1);
        ivfCounts = vector<size_t>(param->teamCount + 1);

        presumeTotalQueryCompareSize = presumeNq / nlist * nprobe * nb * 2;

        for(size_t rank = 1; rank <= workerCount; rank++) {
            size_t teamId = (rank - 1) / param->teamSize + 1; //从1开始
            size_t rankInSideTeam = rank - (teamId - 1) * param->teamSize;
            size_t beginIVF = (teamId - 1) * (nlist / param->teamCount); 
            size_t ivfCount = (teamId == param->teamCount) ? (nlist - beginIVF) : (nlist / param->teamCount);

            beginIVFs[teamId] = beginIVF;
            ivfCounts[teamId] = ivfCount;

            GroupWorker::InitInfo info = GroupWorker::InitInfo(d, d / param->teamSize, workerCount, nlist, blockCount, nprobe,
                nb, presumeTotalQueryCompareSize / param->groupCount / blockCount, param->groupCount, param->teamCount, param->teamSize, beginIVF, ivfCount, teamId, rankInSideTeam);
            MPI_Send(&info, sizeof(info), MPI_BYTE, rank, 0, MPI_COMM_WORLD);
            if(rank == 1)
                out.write(reinterpret_cast<const char*>(&info), sizeof(GroupWorker::InitInfo));

            auto listSizes = std::make_unique<size_t[]>(nlist);
            for (size_t listId = 0; listId < nlist; listId++) {
                IVF& list = lists[listId];
                listSizes[listId] = list.get_list_size();
            }
            MPI_Send(listSizes.get(), nlist * sizeof(size_t), MPI_BYTE, rank, 0, MPI_COMM_WORLD);

            if(rank == 1)
                out.write(reinterpret_cast<const char*>(listSizes.get()), nlist * sizeof(size_t));

            for (size_t i = beginIVF; i < beginIVF + ivfCount; i++) {
                MPI_Send(lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, rank, 0, MPI_COMM_WORLD);
                if(rank == 1)
                    out.write(reinterpret_cast<const char*>(lists[i].candidate_id.get()), listSizes[i] * sizeof(size_t));
            }

            auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
            for (size_t i = beginIVF; i < beginIVF + ivfCount; i++) {
                listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.block_dim);
            }
            for (size_t listId = beginIVF; listId < beginIVF + ivfCount; listId++) {
                IVF& list = lists[listId];
                copy_n_partial_vector(list.candidate_codes.get(), listCodesBuffer[listId].get(), info.d, info.block_dim,
                            (rankInSideTeam - 1) * info.block_dim, list.get_list_size());
                // MPI_Send(listCodesBuffer[listId].get(), list.get_list_size() * info.block_dim, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
                if(rank == 1)
                    out.write(reinterpret_cast<const char*>(listCodesBuffer[listId].get()), list.get_list_size() * info.block_dim * sizeof(float));
            }
            for (size_t listId = beginIVF; listId < beginIVF + ivfCount; listId++) {
                IVF& list = lists[listId];
                MPI_Send(list.candidate_codes.get(), list.get_list_size() * d, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    preSearchTime = watch.watch.elapsedSeconds();
    cout << format("preSearch {}", watch.watch.elapsedSeconds()) << endl; 

    uniWatch = MyStopWatch(true, "masterUniWatch", CRAN);
    uniWatch.print("master cross barrier");
}

void Index::postSearch() {
    if(param->mode == SearchMode::DIVIDE_DIM) {
        if(param->cut) {
            cout << "Analyse skip rate" << endl;
            auto skipRates = vector<std::unique_ptr<double[]>>(workerCount + 1); //
            for(size_t rank = 1; rank <= workerCount; rank++) {
                skipRates[rank] = make_unique<double[]>(blockCount);
                MPI_Recv(skipRates[rank].get() , blockCount, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // printVector(skipRates[rank].get(), blockCount, YELLOW);
            }
            auto averageRate = vector<double>(workerCount + 1, 0);
            for(size_t block = 0; block < blockCount; block++) {
                for(size_t order = 0; order < blockSearchedOrder[block].size(); order++) {
                    size_t worker = blockSearchedOrder[block][order];
                    averageRate[order + 1] += skipRates[worker][block];
                }
            }
            for(size_t order = 1; order <= workerCount; order++) {
                averageRate[order] /= blockCount;
            }

            printVector(averageRate.data() + 1, workerCount, YELLOW);
        }
    }
}

Index& Index::operator=(Index&& other) noexcept {
    d = other.d;
    nlist = other.nlist;
    nprobe = other.nprobe;
    metric = other.metric;
    opt_level = other.opt_level;
    added_opt_level = other.added_opt_level;
    sub_k = other.sub_k;
    sub_nlist = other.sub_nlist;
    sub_nprobe = other.sub_nprobe;
    verbose = other.verbose;
    edge_device_enabled = other.edge_device_enabled;
    lists = std::move(other.lists);
    centroid_codes = std::move(other.centroid_codes);
    centroid_ids = std::move(other.centroid_ids);
    return *this;
}

void Index::train(size_t n, const float* codes, bool faiss, bool lite) {
    // 这里假设Clustering类已经定义好，并且有一个合适的构造函数和train方法
    auto tic1 = std::chrono::high_resolution_clock::now();
    // 默认faiss = false. lite = false
    if (!faiss) {
        ClusteringParameters cp;
        cp.metric = this->metric;
        // 20个iteration
        cp.niter = 20;  // 或其他合适的值
        if (lite) {
            cp.niter = 2;
        }
        cp.seed = 6666;                    // 或其他合适的值
        cp.max_points_per_centroid = 256;  // 或其他合适的值

        Clustering clustering(this->d, this->nlist, verbose, cp);
        clustering.train(n, codes);

        this->centroid_codes.reset(clustering.get_centroids());
    } else {
        // 使用faiss库的train
        faiss::IndexFlatL2 quantizer(d);  // the other index
        faiss::IndexIVFFlat index(&quantizer, d, nlist);
        index.train(n, codes);
        this->centroid_codes = std::make_unique<float[]>(nlist * d);
        std::copy_n(quantizer.get_xb(), nlist * d, this->centroid_codes.get());
    }
    if (metric == MetricType::METRIC_IP) {
        float* codes = this->centroid_codes.get();
#pragma omp parallel for
        for (size_t i = 0; i < nlist; i++) {
            float norm = calculatedInnerProduct(codes + i * d, codes + i * d, d);
            if (norm != 0) {
                for (size_t j = 0; j < d; j++) {
                    codes[i * d + j] /= sqrt(norm);
                }
            }
        }
    }
    auto tic2 = std::chrono::high_resolution_clock::now();
    if (verbose) {
        trainTime = std::chrono::duration<double>(tic2 - tic1).count();
        std::cout << std::format("train elapsed: {:.2f}s\n", std::chrono::duration<double>(tic2 - tic1).count());
    }
}

std::unique_ptr<IVFScanBase> Index::get_scanner(MetricType metric, OptLevel opt_level, size_t k,
                                                EdgeDevice edge_device_enabled) {
    if (metric == MetricType::METRIC_L2) {
        if (edge_device_enabled) {
            switch (opt_level) {
                case OptLevel::OPT_NONE:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_TRIANGLE:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRIANGLE, EdgeDevice::EDGEDEVIVE_ENABLED>(d,
                                                                                                                   k));
                case OptLevel::OPT_SUBNN_L2:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_L2, EdgeDevice::EDGEDEVIVE_ENABLED>(d,
                                                                                                                   k));
                case OptLevel::OPT_SUBNN_IP:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_IP, EdgeDevice::EDGEDEVIVE_ENABLED>(d,
                                                                                                                   k));
                case OptLevel::OPT_TRI_SUBNN_L2:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_L2, EdgeDevice::EDGEDEVIVE_ENABLED>(
                            d, k));
                case OptLevel::OPT_TRI_SUBNN_IP:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_IP, EdgeDevice::EDGEDEVIVE_ENABLED>(
                            d, k));
                case OptLevel::OPT_SUBNN_ONLY:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_ONLY, EdgeDevice::EDGEDEVIVE_ENABLED>(
                            d, k));
                case OptLevel::OPT_ALL:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_ALL, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                default:
                    throw std::runtime_error("Unsupported opt_level");
            }
        } else {
            switch (opt_level) {
                case OptLevel::OPT_NONE:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_TRIANGLE:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRIANGLE, EdgeDevice::EDGEDEVIVE_DISABLED>(d,
                                                                                                                    k));
                case OptLevel::OPT_SUBNN_L2:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_L2, EdgeDevice::EDGEDEVIVE_DISABLED>(d,
                                                                                                                    k));
                case OptLevel::OPT_SUBNN_IP:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_IP, EdgeDevice::EDGEDEVIVE_DISABLED>(d,
                                                                                                                    k));
                case OptLevel::OPT_TRI_SUBNN_L2:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_L2, EdgeDevice::EDGEDEVIVE_DISABLED>(
                            d, k));
                case OptLevel::OPT_TRI_SUBNN_IP:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_IP, EdgeDevice::EDGEDEVIVE_DISABLED>(
                            d, k));
                case OptLevel::OPT_SUBNN_ONLY:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_ONLY, EdgeDevice::EDGEDEVIVE_DISABLED>(
                            d, k));
                case OptLevel::OPT_ALL:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_ALL, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                default:
                    throw std::runtime_error("Unsupported opt_level");
            }
        }
    } else {
        if (edge_device_enabled) {
            switch (opt_level) {
                case OptLevel::OPT_NONE:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_TRIANGLE:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRIANGLE, EdgeDevice::EDGEDEVIVE_ENABLED>(d,
                                                                                                                   k));
                // case OptLevel::OPT_SUBNN_L2:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_L2,
                //     EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_SUBNN_IP:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_IP,
                //     EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_TRI_SUBNN_L2:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP,
                //     OptLevel::OPT_TRI_SUBNN_L2, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_TRI_SUBNN_IP:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP,
                //     OptLevel::OPT_TRI_SUBNN_IP, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_SUBNN_ONLY:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_ONLY,
                //     EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_ALL:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_ALL,
                //     EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                default:
                    throw std::runtime_error("Unsupported opt_level");
            }
        } else {
            switch (opt_level) {
                case OptLevel::OPT_NONE:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_TRIANGLE:
                    return std::unique_ptr<IVFScanBase>(
                        new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRIANGLE, EdgeDevice::EDGEDEVIVE_DISABLED>(d,
                                                                                                                    k));
                // case OptLevel::OPT_SUBNN_L2:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_L2,
                //     EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_SUBNN_IP:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_IP,
                //     EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_TRI_SUBNN_L2:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP,
                //     OptLevel::OPT_TRI_SUBNN_L2, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_TRI_SUBNN_IP:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP,
                //     OptLevel::OPT_TRI_SUBNN_IP, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_SUBNN_ONLY:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_ONLY,
                //     EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_ALL:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_ALL,
                //     EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                default:
                    throw std::runtime_error("Unsupported opt_level");
            }
        }
    }
};

void Index::single_thread_nearest_cluster_search(size_t n, const float* queries, float* distances, idx_t* labels) {
    if (n == 0) {
        return;
    }
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, 1);
    for (size_t i = 0; i < n; i++) {
        scaner_quantizer->set_query(queries + i * d);
        scaner_quantizer->lite_scan_codes(nlist, centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()), distances + i,
                                          labels + i);
        // no need to sort result, because only one result
    }
}

// void Index::add_lcx(size_t n, const float* codes) {

// }
void Index::add_simple(size_t n, const float* codes) {
    // const std::string BLUE = "\033[1;34m"; // Blue text
    // const std::string RESET = "\033[0m"; // Reset color
    std::cout << BLUE << "You are using Simple version of add" << RESET << std::endl;
    // for(size_t i = 0; i < nlist; i++) {
    //     std::cout << BLUE << centroid_ids[i] << RESET << std::endl;
    // }
    // 把n个向量加到聚类中心里面，codes是向量
    if (n == 0) {
        return;
    }
    // 向量到中心的距离，和向量对应聚类中心的id
    std::unique_ptr<float[]> candicate2centroid = std::make_unique<float[]>(n);
    std::unique_ptr<idx_t[]> listidcandicates = std::make_unique<idx_t[]>(n);

    init_result(metric, n, candicate2centroid.get(), listidcandicates.get());
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = n / nt;  // 一个线程分配多少向量
    size_t extra = n % nt;       // 多余的向量数
    // auto nearest_search_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        size_t start, end;
        if (i < extra) {
            start = i * (batch_size + 1);
            end = start + batch_size + 1;
        } else {
            start = i * batch_size + extra;
            end = start + batch_size;
        }
        if (start < end) {
            single_thread_nearest_cluster_search(end - start, codes + start * d, candicate2centroid.get() + start,
                                                 listidcandicates.get() + start);
        }
    }
    // auto nearest_search_end = std::chrono::high_resolution_clock::now();

    // if(verbose){
    //     std::cout << "nearest search elapsed: " << std::chrono::duration<double>(nearest_search_end -
    //     nearest_search_start).count() << "s" << std::endl;
    // }

    // auto sort_and_add_start = std::chrono::high_resolution_clock::now();
    // 每个聚类的向量数量
    std::unique_ptr<size_t[]> list_sizes = std::make_unique<size_t[]>(nlist);
    std::fill_n(list_sizes.get(), nlist, 0);

    // TODO: parallelize this part ?
    // 设置好list_sizes
    for (size_t i = 0; i < n; i++) {
        list_sizes[listidcandicates[i]]++;
    }

    // //一共加了多少个向量
    // size_t total_add = 0;
    // for (size_t i = 0; i < nlist; i++) {
    //     total_add += list_sizes[i];
    // }

#pragma omp parallel for
    for (size_t i = 0; i < nlist; i++) {
        lists[i].reset(list_sizes[i], d, sub_k, added_opt_level);
    }

    std::fill_n(list_sizes.get(), nlist, 0);

    // 加到聚类中的顺序，从距离聚类中心最近的向量开始加
    std::unique_ptr<size_t[]> add_order = std::make_unique<size_t[]>(n);
    std::iota(add_order.get(), add_order.get() + n, 0);
    // if (metric == MetricType::METRIC_L2) {
    std::sort(add_order.get(), add_order.get() + n,
              [&](size_t i, size_t j) { return candicate2centroid[i] < candicate2centroid[j]; });
    // } else {
    //     std::sort(add_order.get(), add_order.get() + n, [&](size_t i, size_t j) { return candicate2centroid[i] >
    //     candicate2centroid[j]; });
    // }

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        // 当前线程的id
        int tid = omp_get_thread_num();

        for (size_t oi = 0; oi < n; oi++) {
            size_t i = add_order[oi];                        // 当前要加的是第i个向量
            size_t list_id = listidcandicates[i];            // assert > 0 ， 第i个向量要加到聚类list_id中
            if (list_id % nt == tid) {                       // tid线程发现是自己负责的list_id
                size_t list_size = list_sizes[list_id];      // 当前聚类的向量数量
                lists[list_id].candidate_id[list_size] = i;  // 把i加到对应的聚类中
                // if ((opt_level & OptLevel::OPT_TRIANGLE) || (opt_level & OptLevel::OPT_SUBNN_IP)) {
                //     lists[list_id].candidate2centroid[list_size] = candicate2centroid[i];
                // }
                std::copy_n(codes + i * d, d, lists[list_id].candidate_codes.get() + list_size * d);  // 拷贝第i个向量
                list_sizes[list_id]++;
            }
        }
    }
    std::cout << BLUE << "end of add" << RESET << std::endl;
}

void Index::add(size_t n, const float* codes) {
    auto tic1 = std::chrono::high_resolution_clock::now();
    if (n == 0) {
        auto tic2 = std::chrono::high_resolution_clock::now();
        if (verbose) {
            std::cout << std::format("add elapsed: {:.2f}s\n", std::chrono::duration<double>(tic2 - tic1).count());
        }
        return;
    }

    added_opt_level = opt_level;
    std::unique_ptr<float[]> candicate2centroid = std::make_unique<float[]>(n);
    std::unique_ptr<idx_t[]> listidcandicates = std::make_unique<idx_t[]>(n);
    init_result(metric, n, candicate2centroid.get(), listidcandicates.get());

    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = n / nt;  // 一个线程分配多少向量
    size_t extra = n % nt;       // 多余的向量数
    auto nearest_search_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        size_t start, end;
        if (i < extra) {
            start = i * (batch_size + 1);
            end = start + batch_size + 1;
        } else {
            start = i * batch_size + extra;
            end = start + batch_size;
        }
        if (start < end) {
            single_thread_nearest_cluster_search(end - start, codes + start * d, candicate2centroid.get() + start,
                                                 listidcandicates.get() + start);
        }
    }
    auto nearest_search_end = std::chrono::high_resolution_clock::now();

    if (verbose) {
        std::cout << "nearest search elapsed: "
                  << std::chrono::duration<double>(nearest_search_end - nearest_search_start).count() << "s"
                  << std::endl;
    }

    auto sort_and_add_start = std::chrono::high_resolution_clock::now();
    // 每个聚类的向量数量
    std::unique_ptr<size_t[]> list_sizes = std::make_unique<size_t[]>(nlist);
    std::fill_n(list_sizes.get(), nlist, 0);

    // TODO: parallelize this part ?
    // 设置好list_sizes
    for (size_t i = 0; i < n; i++) {
        list_sizes[listidcandicates[i]]++;
    }

    // 一共加了多少个向量
    size_t total_add = 0;
    for (size_t i = 0; i < nlist; i++) {
        total_add += list_sizes[i];
    }

#pragma omp parallel for
    for (size_t i = 0; i < nlist; i++) {
        lists[i].reset(list_sizes[i], d, sub_k, added_opt_level);
    }

    std::fill_n(list_sizes.get(), nlist, 0);

    // 加到聚类中的顺序，从距离聚类中心最近的向量开始加
    std::unique_ptr<size_t[]> add_order = std::make_unique<size_t[]>(n);
    std::iota(add_order.get(), add_order.get() + n, 0);
    if (metric == MetricType::METRIC_L2) {
        std::sort(add_order.get(), add_order.get() + n,
                  [&](size_t i, size_t j) { return candicate2centroid[i] < candicate2centroid[j]; });
    } else {
        std::sort(add_order.get(), add_order.get() + n,
                  [&](size_t i, size_t j) { return candicate2centroid[i] > candicate2centroid[j]; });
    }

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        // 当前线程的id
        int tid = omp_get_thread_num();

        for (size_t oi = 0; oi < n; oi++) {
            size_t i = add_order[oi];                        // 当前要加的是第i个向量
            size_t list_id = listidcandicates[i];            // assert > 0 ， 第i个向量要加到聚类list_id中
            if (list_id % nt == tid) {                       // tid线程发现是自己负责的list_id
                size_t list_size = list_sizes[list_id];      // 当前聚类的向量数量
                lists[list_id].candidate_id[list_size] = i;  // 把i加到对应的聚类中
                if ((opt_level & OptLevel::OPT_TRIANGLE) || (opt_level & OptLevel::OPT_SUBNN_IP)) {
                    lists[list_id].candidate2centroid[list_size] = candicate2centroid[i];
                }
                std::copy_n(codes + i * d, d, lists[list_id].candidate_codes.get() + list_size * d);  // 拷贝第i个向量
                list_sizes[list_id]++;
            }
        }
    }
    auto sort_and_add_end = std::chrono::high_resolution_clock::now();
    auto sort_and_add_elapsed = std::chrono::duration<double>(sort_and_add_end - sort_and_add_start).count();
    if (verbose) {
        std::cout << "sort and add elapsed: " << sort_and_add_elapsed << "s" << std::endl;
    }

    if (metric == MetricType::METRIC_L2) {
        if ((opt_level & OptLevel::OPT_TRIANGLE) || (opt_level & OptLevel::OPT_SUBNN_IP)) {
#pragma omp parallel for
            for (size_t list_id = 0; list_id < nlist; list_id++) {
                size_t list_size = lists[list_id].list_size;
                for (size_t i = 0; i < list_size; i++) {
                    lists[list_id].sqrt_candidate2centroid[i] = std::sqrt(lists[list_id].candidate2centroid[i]);
                }
            }
        }

#pragma omp parallel for
        for (size_t list_id = 0; list_id < nlist; list_id++) {
            size_t list_size = lists[list_id].list_size;
            for (size_t i = 0; i < list_size; i++) {
                const float* code = codes + lists[list_id].candidate_id[i] * d;
                lists[list_id].candidate_norms[i] = calculatedInnerProduct(code, code, d);
            }
        }
    }

    if (metric == MetricType::METRIC_L2) {
        size_t total_processd = 0;

        size_t total_sub_count_ip = 0;
        size_t total_sub_recall_ip = 0;
        size_t total_sub_count_l2 = 0;
        size_t total_sub_recall_l2 = 0;
        size_t total_sub_count_ip_5 = 0;
        size_t total_sub_recall_ip_5 = 0;
        size_t total_sub_count_l2_5 = 0;
        size_t total_sub_recall_l2_5 = 0;
        Stopwatch logwatch;
        double train_elapsed = 0;
        double add_elapsed = 0;
        double search_elapsed = 0;
        double log_interval = 2;
        [[maybe_unused]] auto running_log = [&]() -> void {
            if (verbose) {
                if (logwatch.elapsedSeconds() > log_interval || total_processd == nlist) {
                    logwatch.reset();
                    std::cout << std::format("build: {:.2f}%", 100.0 * total_processd / nlist) << std::endl;
                    double total_elapsed = train_elapsed + add_elapsed + search_elapsed;
                    double train_percent = 100.0 * train_elapsed / total_elapsed;
                    double add_percent = 100.0 * add_elapsed / total_elapsed;
                    double search_percent = 100.0 * search_elapsed / total_elapsed;
                    std::cout << std::format("train: {:.2f}%    add: {:.2f}%    search: {:.2f}%    total: {:.2f}\n",
                                             train_percent, add_percent, search_percent, total_elapsed);
                    float sub_recall_ip = total_sub_count_ip ? 100.0 * total_sub_recall_ip / total_sub_count_ip : 0;
                    float sub_recall_l2 = total_sub_count_l2 ? 100.0 * total_sub_recall_l2 / total_sub_count_l2 : 0;
                    float sub_recall_ip_5 =
                        total_sub_count_ip_5 ? 100.0 * total_sub_recall_ip_5 / total_sub_count_ip_5 : 0;
                    float sub_recall_l2_5 =
                        total_sub_count_l2_5 ? 100.0 * total_sub_recall_l2_5 / total_sub_count_l2_5 : 0;
                    std::cout << std::format(
                        "Recall    SUBNN_IP top5: {:.2f}%    topk: {:.2f}%    SUBNN_L2 top5: {:.2f}%    topk: {:.2f}%  "
                        "  {}/{}\n",
                        sub_recall_ip_5, sub_recall_ip, sub_recall_l2_5, sub_recall_l2, sub_nprobe, sub_nlist);
                }
            }
        };

        [[maybe_unused]] auto end_log = [&]() -> void {
            if (verbose) {
                double total_elapsed = train_elapsed + add_elapsed + search_elapsed;
                std::cout << std::format("build: 100.0%\n");
                std::cout << std::format("train: {:.2f}   add: {:.2f}    search: {:.2f}    total: {:.2f}\n",
                                         train_elapsed, add_elapsed, search_elapsed, total_elapsed);
            }
        };
#pragma omp parallel for
        for (size_t listid = 0; listid < nlist; listid++) {
            IVF& list = lists[listid];
            const float* xb = list.get_candidate_codes();
            size_t nb = list.get_list_size();

            size_t this_sub_nlist_L2 = std::min(sub_nlist, (nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE);
            size_t this_sub_nprobe_L2 =
                std::min(std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_L2 * sub_nprobe / sub_nlist)),
                         this_sub_nlist_L2);

            size_t this_sub_nlist_IP = std::min(std::max(1ul, static_cast<size_t>(sub_nlist)),
                                                static_cast<size_t>((nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE));
            size_t this_sub_nprobe_IP = std::min(
                std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_IP * sub_nprobe / sub_nlist * IP_SUB_RATIO)),
                this_sub_nlist_IP);

            const float* centroid_code = centroid_codes.get() + listid * d;




#pragma omp critical
            {
                total_processd++;
                running_log();
            }
        }
        end_log();
    } else {
        // do nothing
    }

    auto tic2 = std::chrono::high_resolution_clock::now();
    if (verbose) {
        addTime = std::chrono::duration<double>(tic2 - tic1).count();
        std::cout << std::format("add elapsed: {:.2f}s\n", std::chrono::duration<double>(tic2 - tic1).count());
    }
}
void Index::findNearNprobeOfCentroidIds(size_t n, const float* queries) {
    MyStopWatch watch(true, "FindN watch", RED);

    if(param->hardInBalance) {
        cout << YELLOW << "Hard InBalance" << RESET << endl;
        listidqueries = std::make_unique<idx_t[]>(n * nprobe);  // 最近的nprobe个聚类中心的id
        size_t hardInBalanceTeamSize = workerCount / param->hardInBalanceTeam;
        // auto nlistPerTeam = distribute_jobs(nprobe, param->hardInBalanceTeam, 0.8);
        auto nlistPerTeam = vector<std::vector<int>>(param->hardInBalanceTeam);
        for(int i = 0; i < nlistPerTeam.size(); i++) {
            size_t beginJob = i * (nprobe / param->hardInBalanceTeam); 
            size_t jobCount = (i == (nlistPerTeam.size() - 1)) ? (nprobe - beginJob) : (nprobe / param->hardInBalanceTeam); 
            nlistPerTeam[i] = distribute_jobs(jobCount, hardInBalanceTeamSize, param->hardInBalanceRatio);
        }
        for (int i = 0; i < nlistPerTeam.size(); ++i) {
            printVector(nlistPerTeam[i], BLUE, format("{}", i));
        }
        std::random_device rd;           // Obtain a random seed from the system
        std::mt19937 gen(42);          // Initialize the random number generator (Mersenne Twister)
    
        // Define a uniform integer distribution within the range [min, max]
        vector<std::uniform_int_distribution<>> dis = vector<std::uniform_int_distribution<>>(workerCount);
        for(int i = 0; i < dis.size(); i++) {
            size_t beginIVF = i  * (nlist / workerCount); 
            size_t ivfCount = (i == (workerCount - 1)) ? (nlist - beginIVF) : (nlist / workerCount);
            dis[i] = std::uniform_int_distribution<>(beginIVF, beginIVF + ivfCount - 1); //[beginIVF, beginIVF + ivfCount - 1]
        }

        // Generate and return the random number
        std::cout << "Job distribution:" << std::endl;
        size_t pos = 0;
        for(size_t q = 0; q < n; q++) {
            for(int i = 0; i < workerCount; i++) {
                int allocatedList = nlistPerTeam[i / hardInBalanceTeamSize][i % hardInBalanceTeamSize];
                if(q == 0) {
                    std::cout << "worker " << i+1 << " : " << allocatedList << " list" << std::endl;
                }
                while(allocatedList--) {
                    listidqueries[pos] = dis[i](gen);
                    pos++;
                }
            }
            // cout << pos << endl;
            std::mt19937 g(rd());
            // std::shuffle(listidqueries.get(), listidqueries.get() + nprobe, g);
            printVector(listidqueries.get(), nprobe, BLUE);
        }
        // size_t offset = 0;
        // for(size_t q = 0; q < n; q++) {
        //     for(size_t i = 0; i < nprobe; i++) {
        //         listidqueries[q * nprobe  + i] = i + offset;
        //     }
        // }
        // return listidqueries;
        watch.print("");
        return;
    }

    std::unique_ptr<float[]> centroid2queries =
        std::make_unique<float[]>(n * nprobe);  // n个查询向量到nprobe个聚类中心的距离
    listidqueries = std::make_unique<idx_t[]>(n * nprobe);  // 最近的nprobe个聚类中心的id
    // watch.print(format("malloc"));
    init_result(metric, n * nprobe, centroid2queries.get(),
                listidqueries.get());  // 优先队列，存储离n个查询向量最近的nprobe个聚类中心
    // watch.print(format("init result"));

    // 下面四个向量都和i绑定，也就是和每一个查询绑定
    // float* centroids2query = centroid2queries.get();  // 单个查询对应的距离
    // idx_t* listids = listidqueries.get();             // 单个查询对应的聚类中心id

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        // 每一个i对应一个查询
        float* centroids2query = centroid2queries.get() + i * nprobe;  // 单个查询对应的距离
        idx_t* listids = listidqueries.get() + i * nprobe;             // 单个查询对应的聚类中心id
        std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, nprobe);  // 搜索最近的聚类中心
        scaner_quantizer->set_query(queries + i * d);
        // 获取最近的nprobe个聚类中心
        scaner_quantizer->lite_scan_codes(nlist, centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()),
                                          centroids2query,  // ret
                                          listids);         // ret , 分别对应堆
        // 要查的聚类id存储在listids的前nprobe个
        sort_result(metric, nprobe, centroids2query, listids);
        // centroids2query += nprobe;
        // listids += nprobe;
    }
    // watch.print(format("main loop"));
    // return listidqueries;
}
void Index::warmUpSearch(size_t n, const float* queries, size_t k, float* distances, idx_t* labels,
                         idx_t* listidqueries) {
    // std::cout << BLUE << "simple version of search" << RESET;
    // n是查询向量的数量，queries是查询向量的起始位置，distance是结果存放的起始位置, k指前k个
    // std::unique_ptr<IVFScanBase> scaner = get_scanner(metric, opt_level, k);  // 在聚类中心内部搜

    // 下面四个向量都和i绑定，也就是和每一个查询绑定
    // float* disi = distances;         // 结果，查询向量最近的k个向量的距离
    // idx_t* idxi = labels;            // 结果，查询向量最近的k个向量的id
    // idx_t* listids = listidqueries;  // 单个查询对应的聚类中心id

    cout << GREEN << "[warmupSearchList " << warmUpSearchList << ", warmupSearchListSize " << warmUpSearchListSize << "]" << RESET << endl;
    if (warmUpSearchList * warmUpSearchListSize < k) {
        cout << YELLOW << "WARNING: not enough warmup" << RESET << endl;
    }
    // assert(warmUpSearchList % nlist == 0);
    // 设置为k

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        // 每一个i对应一个查询
        std::unique_ptr<IVFScanBase> scaner = get_scanner(metric, opt_level, k);  // 在聚类中心内部搜
        scaner->set_query(queries + i * d);
        float* disi = distances + i * k;         // 结果，查询向量最近的k个向量的距离
        idx_t* idxi = labels + i * k;            // 结果，查询向量最近的k个向量的id
        idx_t* listids = listidqueries + i * nprobe;  // 单个查询对应的聚类中心id
        // 获取最近的nprobe个聚类中心
        // 要查的聚类id存储在listids的前nprobe个

        //由于listidqueries里面的聚类id是从近到远排列的，因此可以从0开始遍历
        // size_t totalCompare = 0;
        for (size_t j = 0; j < warmUpSearchList; j++) {
            // 在第j个聚类中搜索所有点
            // list代表聚类
            IVF& list = lists[listids[j]];

            // 查询点到中心的距离
            // 聚类中的点的数量
            size_t list_size = list.get_list_size();

            if (list_size < warmUpSearchListSize) {
                // cout << YELLOW << std::format("WARNING: warmupSearchListSize set to list_size({})", list_size) << RESET << endl;
                // totalCompare += list_size;
                scaner->lite_scan_codes(list_size, list.get_candidate_codes(), list.get_candidate_id(), disi, idxi);
            } else {
                // totalCompare += warmUpSearchListSize;
                scaner->lite_scan_codes(warmUpSearchListSize, list.get_candidate_codes(), list.get_candidate_id(), disi, idxi);
            }
            

        }
        // cout << std::format("warm{} {}", i, totalCompare) << endl;
        // sort_result(metric, k, disi, idxi);
        // disi += k;
        // idxi += k;
        // listids += nprobe;
    }
}


void Index::search_group_master(size_t n, const float* queries, size_t k, float* distances, idx_t* labels) {
    this->groupSize = n / param->groupCount;
    this->blockSize = groupSize / blockCount;
    // cout << groupSize << " " << blockSize << endl;

    printIndex();

    MyStopWatch watch(true, "search_group_master", CRAN);

    // n个查询向量对应的nprobe个聚类中心id
    findNearNprobeOfCentroidIds(n, queries);


    // 3. nq, querys
    MPI_Bcast(&n, sizeof(n), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(const_cast<float*>(queries), n * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // watch.print("nq,querys");

    // 4.listidqueries
    MPI_Bcast(listidqueries.get(), n * nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);

    // printVector(beginIVFs, GREEN);
    // printVector(ivfCounts, RED);

    // for(int i = 0; i < param->groupCount; i++) {
    //     for(int j = 0; j < blockCount; j++) {
    //         cout << format("group {} block {} tag {}", i, j, GroupWorker::getTag(i, j, blockCount)) << endl;
    //     }
    // }
    //算出每个worker的每个查询向量的数据量是多少
    vector<std::unique_ptr<idx_t[]>> queryCompareSize = vector<std::unique_ptr<idx_t[]>>(param->teamCount + 1);
    for(int teamId = 1; teamId <= param->teamCount; teamId++) {
        queryCompareSize[teamId] = std::make_unique<idx_t[]>(n);
    }
    
    for(size_t q = 0; q < n; q++) {
        idx_t* listIds = listidqueries.get() + q * nprobe;
        for(size_t i = 0; i < nprobe; i++) {
            idx_t listId = listIds[i];
            for(size_t teamId = 1; teamId <= param->teamCount; teamId++) {
                if(listId >= beginIVFs[teamId] && listId < beginIVFs[teamId] + ivfCounts[teamId]) {
                    queryCompareSize[teamId][q] += lists[listId].get_list_size();
                    // cout << format("rank {} q{} listid {}", rank, q, listId) << endl;
                }
            }
        }
    }
    // for(size_t teamId = 1; teamId <= param->teamCount; teamId++) {
    //     cout << queryCompareSize[teamId][0] << endl;
    // }
    vector<std::unique_ptr<idx_t[]>> queryCompareSizePreSum = vector<std::unique_ptr<idx_t[]>>(param->teamCount + 1);
    for(int teamId = 1; teamId <= param->teamCount; teamId++) {
        queryCompareSizePreSum[teamId] = std::make_unique<idx_t[]>(n + 1);
    }
    for(size_t teamId = 1; teamId <= param->teamCount; teamId++) {
        for (size_t q = 1; q < n + 1; q++) {
            queryCompareSizePreSum[teamId][q] = queryCompareSizePreSum[teamId][q - 1] + queryCompareSize[teamId][q - 1];
        }
    }
    for(size_t teamId = 1; teamId <= param->teamCount; teamId++) {
        for(size_t rank = (teamId - 1) * param->teamSize + 1; rank <= teamId * param->teamSize; rank++) {
            // MPI_Send(queryCompareSizePreSum[rank].get(), n,MPI_INT64_T, rank, 0, MPI_COMM_WORLD);
            MPI_Send(queryCompareSize[teamId].get(), n,MPI_INT64_T, rank, 0, MPI_COMM_WORLD);
            MPI_Send(queryCompareSizePreSum[teamId].get(), n + 1,MPI_INT64_T, rank, 0, MPI_COMM_WORLD);
        }
    }
    // watch.print("queryCompareSize");


    std::unique_ptr<float[]> heapTops = std::make_unique<float[]>(n);
    warmUpSearch(n, queries, k, distances, labels, listidqueries.get());
    // watch.print("warmupSearch");
    for(size_t i = 0; i < n; i++) {
        heapTops[i] = distances[i * k];
    }
    // 最大堆广播
    MPI_Bcast(heapTops.get(), n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    init_result(METRIC_L2, n * k, distances, labels);


    watch.print("Query准备工作完成,开始receive");
    
    bool first = false;
    // MyStopWatch loadWatch; //负载均衡
#pragma omp parallel for
    for(size_t teamId = 1; teamId <= param->teamCount; teamId++) {
        MyStopWatch recvWatch(true, "Master Recv Watch", CRAN);
        for(size_t groupOrder = 0; groupOrder < param->groupCount; groupOrder++) {
            size_t groupId = groupSearchOrder.workerSearchOrder[teamId][groupOrder];
            // cout << "teamid" << teamId << "groupId" << groupId << endl;
#pragma omp parallel for
            for(size_t blockId = 0; blockId < blockCount; blockId++) {
                size_t senderRank = blockSearchOrder.searchedWorkerOrder[blockId][blockSearchOrder.searchedWorkerOrder[blockId].size() - 1] + (teamId - 1) * param->teamSize;
                size_t q = groupId * groupSize + blockId * blockSize;
                int tag = GroupWorker::getTag(groupId, blockId, blockCount);
                // cout << format("blockId {} sender {} group {} team {} q {} tag {}", blockId, senderRank, groupId, teamId, q, tag) << endl;
                MPI_Recv(distances + q * k , blockSize * k, MPI_FLOAT, senderRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(labels    + q * k , blockSize * k, MPI_INT64_T, senderRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            recvWatch.print(format("接收到 team {} group {} 的所有block", teamId, groupId));

            for(size_t q = groupId * groupSize; q < (groupId + 1) * groupSize; q++) {
                heapTops[q] = distances[q * k];
            }

            size_t receiverTeam = groupSearchOrder.sendNextWorker[teamId][groupId];
            if(receiverTeam != 0) {
                size_t q = groupId * groupSize;
                for(size_t rank = (receiverTeam - 1) * param->teamSize + 1; rank <= receiverTeam * param->teamSize; rank++) {
                    // cout << format("receiver {} group {} team {} q {} tag {} {}", rank, groupId, teamId, q, GroupWorker::getDistanceHeapTag(groupId), GroupWorker::getIdHeapTag(groupId)) << endl;
                    MPI_Send(heapTops.get() + q, groupSize, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
                    //TODO 优化只需要发一个块的heap就可以了
                    MPI_Send(distances + q * k, groupSize * k, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
                    MPI_Send(labels + q * k, groupSize * k, MPI_INT64_T, rank, 0, MPI_COMM_WORLD);
                }
                recvWatch.print(format("发送 group {} 的Heap到 team {}", groupId, receiverTeam));
            }
        }
    }
    watch.print("Recv完成"); 
    for(int q = 0; q < n; q++) {
        sort_result(METRIC_L2, k, distances + q * k, labels + q * k);
    }
    watch.print("sort result"); 
    // loadWatch.print("Load Balance Time(first to Last Block)");
    // watch.print("完成搜索"); 
    // cout << CRAN << "finish search" << RESET << endl;
}
void Index::single_thread_search_block(size_t n, const float* queries, size_t k, float* distances, idx_t* labels) {
    this->blockSize = n / blockCount;
    printIndex();
    MyStopWatch watch(true);
    // std::cout << GREEN << "block search" << RESET << std::endl;
    // n个查询向量对应的nprobe个聚类中心id
    findNearNprobeOfCentroidIds(n, queries);
    // watch.print("findNearNprobeOfCentroidIds");

    
    // 算出每个查询向量一共要和多少个向量比较
    std::unique_ptr<idx_t[]> queryCompareSize = std::make_unique<idx_t[]>(n);
#pragma omp parallel for
    for (size_t q = 0; q < n; q++) {
        for (size_t i = 0; i < nprobe; i++) {
            queryCompareSize[q] += lists[listidqueries[q * nprobe + i]].get_list_size();
        }
    }

    // 为了计算第q个向量的distancesForQueryies的偏移量,偏移量是queryCompareSizePreSum[q], 注意大小是n + 1
    std::unique_ptr<idx_t[]> queryCompareSizePreSum = std::make_unique<idx_t[]>(n + 1);
    for (size_t q = 1; q < n + 1; q++) {
        queryCompareSizePreSum[q] = queryCompareSizePreSum[q - 1] + queryCompareSize[q - 1];
    }
    // 所有查询向量加起来一共要比多少个向量
    idx_t totalQueryCompareSize = queryCompareSizePreSum[n - 1] + queryCompareSize[n - 1];

    std::unique_ptr<idx_t[]> queryCompareSizeForBlocks = std::make_unique<idx_t[]>(blockCount);
    for (size_t i = 0; i < blockCount; i++) {
        size_t queryStart = i * blockSize;
        idx_t compareSizeForBlock =
            queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        queryCompareSizeForBlocks[i] = compareSizeForBlock;
    }
    // watch.print("queryCompareSize");

    // 3. nq, querys
    MPI_Bcast(&n, sizeof(n), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(const_cast<float*>(queries), n * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // watch.print("nq,querys");

    // 4.listidqueries
    MPI_Bcast(listidqueries.get(), n * nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
    // 5.queryCompareSize
    MPI_Bcast(queryCompareSize.get(), n, MPI_INT64_T, 0, MPI_COMM_WORLD);
    // 6.queryCompareSizePreSum
    MPI_Bcast(queryCompareSizePreSum.get(), (n + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);

    // 最大堆
    if(param->fullWarmUp) {
        if(param->heapTops) {
            MPI_Bcast(param->heapTops, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        } else {
            cerr << "heapTop not exist" << endl;
            exit(1);
        }
    } else {
        std::unique_ptr<float[]> heapTops = std::make_unique<float[]>(n);
        warmUpSearch(n, queries, k, distances, labels, listidqueries.get());
        // watch.print("warmupSearch");
        for(size_t i = 0; i < n; i++) {
            heapTops[i] = distances[i * k];
        }
        // 最大堆广播
        MPI_Bcast(heapTops.get(), n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // watch.print("init heapTops");
    // watch.print("BroadCast queryCompareSize");

    // watch.print("BroadCast heaptops");


    // cout << totalQueryCompareSize << endl;
    // if (presumeTotalQueryCompareSize < totalQueryCompareSize) {
    //     distancesForNQuerys = std::make_unique<float[]>(totalQueryCompareSize);
    // }
    // watch.print("might malloc distancesForNquerys");


    //重置由于warmup导致的heap已经有了一些值
    init_result(METRIC_L2, n * k, distances, labels);

    // watch.print("init result");

    watch.print("search block before loop");

    bool first = false;
    MyStopWatch loadWatch; //负载均衡
#pragma omp parallel for
    for(size_t blockId = 0; blockId < blockCount; blockId++) {
        size_t senderRank = blockSearchedOrder[blockId][blockSearchedOrder[blockId].size() - 1];
        size_t q = blockId * blockSize;
        size_t queryOffset = queryCompareSizePreSum[q];
        // cout << format("master waiting for block({}) from node({})", blockId, senderRank) << endl;
        // uniWatch.print(format("master waiting block {}", blockId), false);
        // MyStopWatch wt(true, "recv watch", RED);
        MPI_Recv(distances + q * k , blockSize * k, MPI_FLOAT, senderRank, blockId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(labels + q * k , blockSize * k, MPI_INT64_T, senderRank, blockId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printVector(labels + q, blockSize *);
        // if(queryCompareSizeForBlocks[blockId] > INT_MAX) {
        //     idx_t sizeToSend = queryCompareSizeForBlocks[blockId];
        //     while(sizeToSend > INT_MAX) {
        //         MPI_Recv(distancesForNQuerys.get() + queryOffset + queryCompareSizeForBlocks[blockId] - sizeToSend, 
        //                 INT_MAX, MPI_FLOAT, senderRank, blockId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //         sizeToSend -= INT_MAX;
        //     }
        //     MPI_Recv(distancesForNQuerys.get() + queryOffset + queryCompareSizeForBlocks[blockId] - sizeToSend, 
        //                 sizeToSend, MPI_FLOAT, senderRank, blockId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // } else {
        //     MPI_Recv(distancesForNQuerys.get() + queryOffset, queryCompareSizeForBlocks[blockId],
        //              MPI_FLOAT, senderRank, blockId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // }
        // wt.print(format("receive block {}", blockId));
        // auto clock1 = std::chrono::high_resolution_clock::now();
        // std::cout << format("receive block {} on {:.2f}s", blockId, std::chrono::duration<double>(clock1 - clock).count()) << std::endl;
        uniWatch.print(format("master receive block {}", blockId), false);
        // cout << format("master received block({}) from node({})", blockId, senderRank) << endl;
        if(first == false) {
            first = true;
            loadWatch.reset();
        }
    }
    loadWatch.print("Load Balance Time(first to Last Block)");
    watch.print("searchblock");
    

// #pragma omp parallel for
//     for (size_t q = 0; q < n; q++) {
//         float* disHeap = distances + q * k;
//         idx_t* idHeap = labels + q * k;
//         size_t offsetOutOfList = 0;
//         for (size_t i = 0; i < nprobe; i++) {
//             IVF& list = lists[listidqueries[i + q * nprobe]];
//             for (size_t j = 0; j < list.get_list_size(); j++) {
//                 float dis = distancesForNQuerys[queryCompareSizePreSum[q] + offsetOutOfList];
//                 if (dis < disHeap[0]) {
//                     // 比堆顶
//                     idx_t id = list.candidate_id[j];
//                     heap_replace_top<MetricType::METRIC_L2>(k, disHeap, idHeap, dis, id);
//                 }
//                 offsetOutOfList++;
//             }
//         }
//         sort_result(metric, k, disHeap, idHeap);
//         // printVector(disHeap, k, BLUE);
//         // printVector(idHeap, k, GREEN);
//         // idHeap += k;
//         // disHeap += k;
//     }
//     //     // auto clock7 = std::chrono::high_resolution_clock::now();
//     //     // std::cout << "heap:" << std::chrono::duration<double>(clock7 - clock6).count() << "s" << std::endl;
//     watch.print("heap");
//     // MPI_Recv(&resultInfo, sizeof(Node::SearchResultInfo), MPI_BYTE, MPI_ANY_SOURCE, Node::SearchResultTag::INFO,
//     // MPI_COMM_WORLD, &status);

    cout << CRAN << "finish search" << RESET << endl;
}
void Index::search_divide_ivf(size_t n, const float* queries, size_t k, float* distances, idx_t* labels) {
    printIndex();
    MyStopWatch watch(true);
    // n个查询向量对应的nprobe个聚类中心id
    findNearNprobeOfCentroidIds(n, queries);
    watch.print("findNearNprobeOfCentroidIds");

    // 3. nq, k, querys
    MPI_Bcast(&n, sizeof(n), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(const_cast<float*>(queries), n * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    watch.print("nq,querys");

    // 4.listidqueries
    MPI_Bcast(listidqueries.get(), n * nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
    if (presumeK * presumeNq < n * k) {
        // distancesForNQuerys = std::make_unique<float[]>(totalQueryCompareSize);
        cerr << "presumeK and presumeNq too small" << endl;
        exit(1);
    }
    init_result(METRIC_L2, n * k, distances, labels);

    bool first = false;
    MyStopWatch loadWatch; //负载均衡
#pragma omp parallel for
    for(size_t rank = 1; rank <= workerCount; rank++) {
        // uniWatch.print(format("master waiting block {}", blockId), false);
        MPI_Recv(distancesHeapBuffer[rank].get() , n * k, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(labelsHeapBuffer[rank].get() , n * k, MPI_INT64_T, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        uniWatch.print(format("master receive rank {}", rank), false);
        if(first == false) {
            first = true;
            loadWatch.reset();
        }
    }
    loadWatch.print("Load Balance Time(first to Last Block)");
    watch.print("searchblock");
    

#pragma omp parallel for
    for (size_t q = 0; q < n; q++) {
        float* disHeap = distances + q * k;
        idx_t* idHeap = labels + q * k;
        // size_t queryOffset = k * q;
        for(size_t rank = 1; rank <= workerCount; rank++) {
            for(size_t i = k * q; i < k * q + k; i++) {
                float dis = distancesHeapBuffer[rank][i];
                if (dis < disHeap[0]) {
                    // 比堆顶
                    idx_t id = labelsHeapBuffer[rank][i];
                    heap_replace_top<MetricType::METRIC_L2>(k, disHeap, idHeap, dis, id);
                }
            }
        }
        sort_result(metric, k, disHeap, idHeap);
    }
    watch.print("heap");

    cout << CRAN << "finish search" << RESET << endl;
}
void Index::printIndex() {
    cout << YELLOW << "Index:[";
    cout << "d:" << d;
    cout << ",nlist:" << nlist;
    cout << ",nprobe:" << nprobe;
    cout << ",node:" << workerCount;
    cout << "]";
    cout << endl;
}
void Index::single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels,
float ratio, Stats* stats) {
    // std::cout << BLUE << "simple version of search" << RESET;
    //n是查询向量的数量，queries是查询向量的起始位置，distance是结果存放的起始位置, k指前k个

    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, nprobe); //搜索最近的聚类中心
    std::unique_ptr<IVFScanBase> scaner = get_scanner(metric, opt_level, k); //在聚类中心内部搜

    std::unique_ptr<float[]> centroid2queries = std::make_unique<float[]>(n * nprobe); //n个查询向量到nprobe个聚类中心的距离 
    listidqueries = std::make_unique<idx_t[]>(n * nprobe); //最近的nprobe个聚类中心的id 
    init_result(metric, n * nprobe, centroid2queries.get(), listidqueries.get()); //优先队列，存储离n个查询向量最近的nprobe个聚类中心

    //下面四个向量都和i绑定，也就是和每一个查询绑定
    float* disi = distances; //结果，查询向量最近的k个向量的距离
    idx_t* idxi = labels; //结果，查询向量最近的k个向量的id
    float* centroids2query = centroid2queries.get(); //单个查询对应的距离
    idx_t* listids = listidqueries.get();//单个查询对应的聚类中心id

    for (size_t i = 0; i < n; i++) {
        //每一个i对应一个查询
        scaner_quantizer->set_query(queries + i * d);
        scaner->set_query(queries + i * d);
        //获取最近的nprobe个聚类中心
        scaner_quantizer->lite_scan_codes(nlist,
                                          centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()),
                                          centroids2query, //ret
                                          listids); //ret , 分别对应堆
        //要查的聚类id存储在listids的前nprobe个
        sort_result(metric, nprobe, centroids2query, listids);

        if (metric == MetricType::METRIC_L2) {
            for (size_t j = 0; j < nprobe; j++) {
                //在第j个聚类中搜索所有点
                //list代表聚类
                IVF& list = lists[listids[j]];

                //查询点到中心的距离
                float centroid2query = centroids2query[j];
                //聚类中的点的数量
                size_t list_size = list.get_list_size();

                size_t scan_begin = 0;
                size_t scan_end = list_size;

                // for (size_t k = 0; k < list_size; k++) {
                //     const float* candicate = list.get_candidate_codes() + k * d;
                //     float dis = 0;
                //     dis = calculatedEuclideanDistance(query, candicate, d);
                //     if (dis < disi[0]) {
                //         //比堆顶
                //         heap_replace_top<metric>(k, disi, idxi, dis, ids[k]);
                //     }
                // }
                // cout << i <<  " " << list_size;
                // printVector(list.get_candidate_codes(), list_size * d, BLUE);
                // printVector(list.get_candidate_id(), list_size, MAG);
                // printVector(scaner->query, d, GREEN);
                scaner->lite_scan_codes(list_size,
                                          list.get_candidate_codes(),
                                          list.get_candidate_id(),
                                          disi, //ret
                                          idxi); //ret , 分别对应堆
                // scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(),
                //                    list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
                //                    list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
                //                    list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(),
                //                    list.get_sub_farest_IP_dis(), list.get_sub_nearest_L2_id(),
                //                    list.get_sub_nearest_L2_dis(), nullptr, disi, idxi, stats,
                //                    centroid_codes.get() + listids[j] * d);
            }
        }
        sort_result(metric, k, disi, idxi);
        disi += k;
        idxi += k;
        centroids2query += nprobe;
        listids += nprobe;
    }
}
// void Index::single_thread_search_worker(size_t n, const float* queries, float* distances, float ratio, Stats* stats, Param* param, float* originalQuery, float* heapTop, idx_t* listidqueries) {
//     // std::cout << BLUE << "simple version of search" << RESET;
//     //n是查询向量的数量，queries是查询向量的起始位置，distance是结果存放的起始位置, k指前k个

//     // std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, nprobe); //搜索最近的聚类中心
//     // // std::unique_ptr<IVFScanBase> scaner = get_scanner(metric, opt_level, k); //在聚类中心内部搜

//     // std::unique_ptr<float[]> centroid2queries = std::make_unique<float[]>(n * nprobe); //n个查询向量到nprobe个聚类中心的距离 
//     // std::unique_ptr<idx_t[]> listidqueries = std::make_unique<idx_t[]>(n * nprobe); //最近的nprobe个聚类中心的id 
//     // init_result(metric, n * nprobe, centroid2queries.get(), listidqueries.get()); //优先队列，存储离n个查询向量最近的nprobe个聚类中心

//     //下面四个向量都和i绑定，也就是和每一个查询绑定
//     // float* disi = distances; //结果，查询向量最近的k个向量的距离
//     // idx_t* idxi = labels; //结果，查询向量最近的k个向量的id
//     // float* centroids2query = centroid2queries.get(); //单个查询对应的距离
//     // idx_t* listids = listidqueries.get();//单个查询对应的聚类中心id
//     idx_t* listids = listidqueries;//单个查询对应的聚类中心id

//     idx_t disPos = 0;
//     size_t skip = 0;
//     for (size_t i = 0; i < n; i++) {
//         //每一个i对应一个查询
//         // scaner_quantizer->set_query(originalQuery + i * d);
//         // // scaner->set_query(queries + i * param->block_dim);
//         // //获取最近的nprobe个聚类中心
//         // scaner_quantizer->lite_scan_codes(nlist,
//         //                                   centroid_codes.get(),
//         //                                   reinterpret_cast<const size_t*>(centroid_ids.get()),
//         //                                   centroids2query, //ret
//         //                                   listids); //ret , 分别对应堆
//         // //要查的聚类id存储在listids的前nprobe个
//         // sort_result(metric, nprobe, centroids2query, listids);
//         // // if(i == 0) {

//         // printVector();
//         // }

//         if (metric == MetricType::METRIC_L2) {
//             for (size_t j = 0; j < nprobe; j++) {
//                 //在第j个聚类中搜索所有点
//                 //list代表聚类
//                 IVF& list = lists[listids[j]];

//                 //查询点到中心的距离
//                 // float centroid2query = centroids2query[j];
//                 //聚类中的点的数量
//                 size_t list_size = list.get_list_size();


//                 for (size_t k = 0; k < list_size; k++) {
//                     if(param->cut) {
//                         if(distances[disPos] == INFINITY) {
//                             skip++;
//                         } else {
//                             const float* candicate = list.get_candidate_codes() + k * param->block_dim;
//                             float dis = calculatedEuclideanDistance(queries + i * param->block_dim, candicate, param->block_dim);
//                             distances[disPos] += dis;
//                             if (distances[disPos] > heapTop[i]) {
//                                 distances[disPos] = INFINITY;
//                             }
//                         }
//                     } else {
//                         const float* candicate = list.get_candidate_codes() + k * param->block_dim;
//                         float dis = calculatedEuclideanDistance(queries + i * param->block_dim, candicate, param->block_dim);
//                         distances[disPos] += dis;
//                     }
//                     disPos++;
//                 }
//                 // cout << i <<  " " << list_size;
//                 // printVector(list.get_candidate_codes(), list_size * d, BLUE);
//                 // printVector(list.get_candidate_id(), list_size, MAG);
//                 // printVector(scaner->query, d, GREEN);
//                 // scaner->lite_scan_codes(list_size,
//                 //                           list.get_candidate_codes(),
//                 //                           list.get_candidate_id(),
//                 //                           disi, //ret
//                 //                           idxi); //ret , 分别对应堆
//                 // scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(),
//                 //                    list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
//                 //                    list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
//                 //                    list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(),
//                 //                    list.get_sub_farest_IP_dis(), list.get_sub_nearest_L2_id(),
//                 //                    list.get_sub_nearest_L2_dis(), nullptr, disi, idxi, stats,
//                 //                    centroid_codes.get() + listids[j] * d);
//             }
//         }
//         // sort_result(metric, k, disi, idxi);
//         // disi += k;
//         // idxi += k;
//         // centroids2query += nprobe;
//         listids += nprobe;
//     }
// }
void Index::single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio,
                                 Stats* stats) {
    // n是查询向量的数量，queries是查询向量的起始位置，distance是结果存放的起始位置, k指前k个
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, nprobe);  // 搜索最近的聚类中心
    std::unique_ptr<IVFScanBase> scaner = get_scanner(metric, opt_level, k);                // 在聚类中心内部搜

    std::unique_ptr<float[]> centroid2queries =
        std::make_unique<float[]>(n * nprobe);  // n个查询向量到nprobe个聚类中心的距离
    auto listidqueries = std::make_unique<idx_t[]>(n * nprobe);  // 最近的nprobe个聚类中心的id
    init_result(metric, n * nprobe, centroid2queries.get(),
                listidqueries.get());  // 优先队列，存储离n个查询向量最近的nprobe个聚类中心


    // 下面四个向量都和i绑定，也就是和每一个查询绑定
    float* simi = distances;
    idx_t* idxi = labels;
    float* centroids2query = centroid2queries.get();  // 单个查询对应的距离
    idx_t* listids = listidqueries.get();             // 单个查询对应的IVF聚类中心id

    for (size_t i = 0; i < n; i++) {
        // 每一个i对应一个查询
        scaner_quantizer->set_query(queries + i * d);
        scaner->set_query(queries + i * d);
        // 把和nlist个聚类中心计算距离的结果放进centroids2query这个堆里面
        scaner_quantizer->lite_scan_codes(nlist, centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()),
                                          centroids2query,  // ret
                                          listids);         // ret , 分别对应两个堆
        // 取前nprobe个聚类中心
        sort_result(metric, nprobe, centroids2query, listids);

        if (metric == MetricType::METRIC_L2) {
            for (size_t j = 0; j < nprobe; j++) {
                // 在第j个聚类中搜索所有点
                // list代表聚类
                IVF& list = lists[listids[j]];

                // 查询点到中心的距离
                float centroid2query = centroids2query[j];
                // 聚类中的点的数量
                size_t list_size = list.get_list_size();

                std::unique_ptr<bool[]> if_skip = std::make_unique<bool[]>(list_size);

                size_t skip_count = 0;
                size_t skip_count_large = 0;
                size_t scan_begin = 0;
                size_t scan_end = list_size;

                // if (opt_level & OptLevel::OPT_TRIANGLE) {
                //     const float* sqrt_candidate2centroid = list.get_sqrt_candidate2centroid();
                //     const float* candidate2centroid = list.get_candidate2centroid();
                //     float sqrt_simi = ratio * sqrt(simi[0]);  // TODO:
                //     float sqrt_centroid2query = sqrt(centroid2query);
                //     for (size_t ii = 0; ii < list_size; ii++) {
                //         float tmp = sqrt_simi + sqrt_candidate2centroid[ii];
                //         if (tmp < sqrt_centroid2query) {
                //             skip_count++;
                //         } else {
                //             break;
                //         }
                //     }

                //     for (int64_t ii = list_size - 1; ii >= 0; ii--) {
                //         float tmp_large = sqrt_simi + sqrt_centroid2query;
                //         tmp_large *= tmp_large;
                //         if (tmp_large < candidate2centroid[ii]) {
                //             skip_count_large++;
                //         } else {
                //             break;
                //         }
                //     }
                //     scan_begin = skip_count;
                //     scan_end -= skip_count_large;
                // }

                IF_STATS {
                    stats->skip_triangle_count += skip_count;
                    stats->skip_triangle_large_count += skip_count_large;
                    stats->total_count += list_size;
                }

                scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(),
                                   list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
                                   list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
                                   list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(),
                                   list.get_sub_farest_IP_dis(), list.get_sub_nearest_L2_id(),
                                   list.get_sub_nearest_L2_dis(), if_skip.get(), simi, idxi, stats,
                                   centroid_codes.get() + listids[j] * d);
            }
        } else {
            for (size_t j = 0; j < nprobe; j++) {
                IVF& list = lists[listids[j]];
                const float* candidate2centroid = list.get_candidate2centroid();
                float centroid2query = centroids2query[j];
                float s_centroid2query = sqrt(1 - centroid2query * centroid2query);
                float s_simi = sqrt(1 - simi[0] * simi[0]);
                size_t list_size = list.get_list_size();
                size_t scan_begin = 0;
                size_t scan_end = list_size;
                if (opt_level & OptLevel::OPT_TRIANGLE) {
                    float min_cut_degree_cos;
                    float max_cut_degree_cos;
                    if (simi[0] < centroid2query) {  // 0 ~ c + s
                        max_cut_degree_cos = 1;
                        min_cut_degree_cos = simi[0] * centroid2query - s_simi * s_centroid2query;
                        while (scan_begin < scan_end && candidate2centroid[scan_end - 1] < min_cut_degree_cos) {
                            scan_end--;
                        }
                    } else {  // c - s ~ c + s
                        max_cut_degree_cos = simi[0] * centroid2query + s_simi * s_centroid2query;
                        min_cut_degree_cos = simi[0] * centroid2query - s_simi * s_centroid2query;
                        while (scan_begin < scan_end && candidate2centroid[scan_begin] > max_cut_degree_cos) {
                            scan_begin++;
                        }
                        while (scan_begin < scan_end && candidate2centroid[scan_end - 1] < min_cut_degree_cos) {
                            scan_end--;
                        }
                    }
                    IF_STATS {
                        stats->skip_triangle_count += scan_begin + list_size - scan_end;
                        stats->total_count += list_size;
                    }
                }
                scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(),
                                   simi, idxi);
            }
        }
        sort_result(metric, k, simi, idxi);

        simi += k;
        idxi += k;
        centroids2query += nprobe;
        listids += nprobe;
    }
}

int Index::single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio,
                                 Stats* stats, size_t startIVF, size_t ivfCount) {
    // n是查询向量的数量，queries是查询向量的起始位置，distance是结果存放的起始位置, k指前k个
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, nprobe);  // 搜索最近的聚类中心
    std::unique_ptr<IVFScanBase> scaner = get_scanner(metric, opt_level, k);                // 在聚类中心内部搜

    std::unique_ptr<float[]> centroid2queries =
        std::make_unique<float[]>(n * nprobe);  // n个查询向量到nprobe个聚类中心的距离
    std::unique_ptr<idx_t[]> listidqueries = std::make_unique<idx_t[]>(n * nprobe);  // 最近的nprobe个聚类中心的id
    init_result(metric, n * nprobe, centroid2queries.get(),
                listidqueries.get());  // 优先队列，存储离n个查询向量最近的nprobe个聚类中心

    // 下面四个向量都和i绑定，也就是和每一个查询绑定
    float* simi = distances;
    idx_t* idxi = labels;
    float* centroids2query = centroid2queries.get();  // 单个查询对应的距离
    idx_t* listids = listidqueries.get();             // 单个查询对应的IVF聚类中心id

    int calculatedCount = 0;

    for (size_t i = 0; i < n; i++) {
        // 每一个i对应一个查询
        scaner_quantizer->set_query(queries + i * d);
        scaner->set_query(queries + i * d);
        // 把和nlist个聚类中心计算距离的结果放进centroids2query这个堆里面
        scaner_quantizer->lite_scan_codes(nlist, centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()),
                                          centroids2query,  // ret
                                          listids);         // ret , 分别对应两个堆
        // 取前nprobe个聚类中心
        sort_result(metric, nprobe, centroids2query, listids);

        if (metric == MetricType::METRIC_L2) {
            for (size_t j = 0; j < nprobe; j++) {
                // 在第j个聚类中搜索所有点
                // list代表聚类
                idx_t ivfId = listids[j];
                if(!(ivfId >= startIVF && ivfId < startIVF + ivfCount)) {
                    continue;
                }
                calculatedCount++;
                IVF& list = lists[listids[j] - startIVF];

                // 查询点到中心的距离
                float centroid2query = centroids2query[j];
                // 聚类中的点的数量
                size_t list_size = list.get_list_size();

                std::unique_ptr<bool[]> if_skip = std::make_unique<bool[]>(list_size);

                size_t skip_count = 0;
                size_t skip_count_large = 0;
                size_t scan_begin = 0;
                size_t scan_end = list_size;

                for (size_t v = 0; v < list_size; v++) {
                    const float* candicate = list.get_candidate_codes() + v * d;
                    float dis = 0;
                    dis = calculatedEuclideanDistance(queries + i * d, candicate, d);
                    if (dis < simi[0]) {
                        //比堆顶
                        heap_replace_top<METRIC_L2>(k, simi, idxi, dis, list.get_candidate_id()[v]);
                    }
                }
                // if (opt_level & OptLevel::OPT_TRIANGLE) {
                //     const float* sqrt_candidate2centroid = list.get_sqrt_candidate2centroid();
                //     const float* candidate2centroid = list.get_candidate2centroid();
                //     float sqrt_simi = ratio * sqrt(simi[0]);  // TODO:
                //     float sqrt_centroid2query = sqrt(centroid2query);
                //     for (size_t ii = 0; ii < list_size; ii++) {
                //         float tmp = sqrt_simi + sqrt_candidate2centroid[ii];
                //         if (tmp < sqrt_centroid2query) {
                //             skip_count++;
                //         } else {
                //             break;
                //         }
                //     }

                //     for (int64_t ii = list_size - 1; ii >= 0; ii--) {
                //         float tmp_large = sqrt_simi + sqrt_centroid2query;
                //         tmp_large *= tmp_large;
                //         if (tmp_large < candidate2centroid[ii]) {
                //             skip_count_large++;
                //         } else {
                //             break;
                //         }
                //     }
                //     scan_begin = skip_count;
                //     scan_end -= skip_count_large;
                // }

                IF_STATS {
                    stats->skip_triangle_count += skip_count;
                    stats->skip_triangle_large_count += skip_count_large;
                    stats->total_count += list_size;
                }

                // scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(),
                //                    list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
                //                    list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
                //                    list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(),
                //                    list.get_sub_farest_IP_dis(), list.get_sub_nearest_L2_id(),
                //                    list.get_sub_nearest_L2_dis(), if_skip.get(), simi, idxi, stats,
                //                    centroid_codes.get() + listids[j] * d);
            }
        } 
        sort_result(metric, k, simi, idxi);

        simi += k;
        idxi += k;
        centroids2query += nprobe;
        listids += nprobe;
    }
    return calculatedCount;
}

Stats Index::search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio) {

    if (n == 0) {
        return Stats();
    }
    if ((opt_level & added_opt_level) != opt_level) {
        std::cerr << "opt_level: " << opt_level << " added_opt_level: " << added_opt_level << std::endl;
        throw std::runtime_error("opt_level is not subset of added_opt_level");
    }
    if (nprobe > nlist) {
        nprobe = nlist;
    }
    // distance是最终结果，是nq个k维的float向量，每一个float对应着和一个相近向量的距离, labels是对应相近向量的id
    // 在distance内部的每一行，对应着一个查询，将其看作一个容量为k的优先队列
    init_result(metric, n * k, distances, labels);

    if(param) {
        if (param->mode == SearchMode::DIVIDE_DIM) {
            std::cout << BLUE << "block version of search" << RESET << std::endl;
            single_thread_search_block(n, queries, k, distances, labels);
            return Stats();
        } else if (param->mode == SearchMode::DIVIDE_VECTOR) {
            std::cout << BLUE << "Divide IVF version of search" << RESET << std::endl;
            search_divide_ivf(n, queries, k, distances, labels);
            return Stats();
        } else if (param->mode == SearchMode::DIVIDE_GROUP) {
            std::cout << BLUE << "Group version of search" << RESET << std::endl;
            search_group_master(n, queries, k, distances, labels);
            return Stats();
        }
    }

    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = n / nt;
    size_t extra = n % nt;
    std::vector<Stats> stats(nt);

    if(param && param->mode == SearchMode::ORIGINAL) {

        cout << YELLOW << "original version of search" << endl;
    }
    
    // 把n个查询交给多线程
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        // cout << " orint = " << omp_get_num_threads() << endl;
        size_t start, end;
        if (i < extra) {
            start = i * (batch_size + 1);
            end = start + batch_size + 1;
        } else {
            start = i * batch_size + extra;
            end = start + batch_size;
        }
        if (start < end) {
            // end - start是查询向量的数量，queries + start * d是查询向量的起始位置，distance + start * k
            // 是结果存放的起始位置
            single_thread_search(end - start, queries + start * d, k, distances + start * k, labels + start * k, ratio,
                                 &stats[i]);
            // single_thread_search_simple(end - start, queries + start * d, k, distances + start * k, labels + start * k, ratio,
            //                      &stats[i]);
        }
    }

    [[maybe_unused]] Stats total_stats = mergeStats(stats);
    return total_stats;
}

void Index::save_index(std::string path) const {
    prepareDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file " + path);
    }
    out.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&nlist), sizeof(size_t));
    // out.write(reinterpret_cast<const char*>(&nprobe), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(MetricType));
    out.write(reinterpret_cast<const char*>(&added_opt_level), sizeof(OptLevel));
    out.write(reinterpret_cast<const char*>(&sub_k), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&sub_nlist), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&sub_nprobe), sizeof(size_t));

    out.write(reinterpret_cast<const char*>(centroid_codes.get()), nlist * d * sizeof(float));
    // out.write(reinterpret_cast<const char*>(centroid_ids.get()), nlist * sizeof(idx_t)); // 0 ~ nlist-1

    for (size_t i = 0; i < nlist; i++) {
        if (lists[i].get_list_size() > 0) {
            out.write(reinterpret_cast<const char*>(&i), sizeof(size_t));
            lists[i].save_IVF(out);
        }
    }
}

void Index::load_index(std::string path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file " + path);
    }
    in.read(reinterpret_cast<char*>(&d), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&nlist), sizeof(size_t));
    // in.read(reinterpret_cast<char*>(&nprobe), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&metric), sizeof(MetricType));
    in.read(reinterpret_cast<char*>(&added_opt_level), sizeof(OptLevel));
    opt_level = OptLevel::OPT_NONE;
    in.read(reinterpret_cast<char*>(&sub_k), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&sub_nlist), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&sub_nprobe), sizeof(size_t));

    centroid_codes = std::make_unique<float[]>(nlist * d);
    in.read(reinterpret_cast<char*>(centroid_codes.get()), nlist * d * sizeof(float));
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);

    lists = std::make_unique<IVF[]>(nlist);
    Stopwatch watch;
    while (true) {
        size_t listid;
        in.read(reinterpret_cast<char*>(&listid), sizeof(size_t));
        if (in.eof()) {
            break;
        }
        lists[listid].load_IVF(in);
        if (watch.elapsedSeconds() > 2) {
            watch.reset();
            std::cout << std::format("loaded list {}/{}, size {}", listid, nlist, lists[listid].get_list_size())
                      << std::endl;
        }
    }
}

void Index::load_SPANN(std::string path) {
    // std::filesystem::path p(path);
    // std::ifstream in(p / "selected.bin", std::ios::binary);
    // if (!in.is_open()) {
    //     throw std::runtime_error("Cannot open file " + (p / "selected.bin").string());
    // }
    // nlist = in.tellg() / sizeof(int32_t);
    // centroid_ids = std::make_unique<idx_t[]>(nlist);
    // in.seekg(0);
    // if (!in.read(reinterpret_cast<char*>(centroid_ids.get()), nlist * sizeof(int32_t))) {
    //     throw std::runtime_error("Cannot read file " + (p / "selected.bin").string());
    // }
    // in.close();

    // std::ifstream in2(p / "selection.bin", std::ios::binary);
    // size_t listno = 0;
    // while (true) {
    //     int32_t node, tonode;
    //     in2.read(reinterpret_cast<char*>(&node), sizeof(int32_t));
    //     in2.read(reinterpret_cast<char*>(&tonode), sizeof(int32_t));
    //     if (tonode == listno) {
    //         // lists[listno].load_SPANN(p, node);
    //     }
    //     if (in2.eof()) {
    //         break;
    //     }
    // }
}

}  // namespace tribase