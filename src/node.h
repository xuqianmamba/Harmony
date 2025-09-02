#ifndef NODE_H
#define NODE_H
#include <mpi.h>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

#include "IVF.h"
#include "cassert"
#include "utils.h"
#include "heap.hpp"
#include "IVFScan.hpp"
#include "order.h"

namespace tribase {

class Index;

using namespace std;

const int presumeNq = 10000;

float calculatedDistance(const float* vec1, const float* vec2, size_t size, MetricType metric);
void try_heap_replace_top(size_t k, float* bh_val, idx_t* bh_ids, float val, idx_t id, MetricType metric);

class Worker {
private:
public:
    struct InitInfo {
        size_t d, block_dim, workerCount;
        size_t nlist;
        size_t blockCount, nprobe, nb;
        idx_t presumeBlockDistancesSize;
        InitInfo(size_t d, size_t block_dim, size_t workerCount, size_t nlist, size_t blockCount, size_t nprobe,
                 size_t nb, idx_t presumeBlockDistancesSize)
            : d(d),
              block_dim(block_dim),
              workerCount(workerCount),
              nlist(nlist),
              blockCount(blockCount),
              nprobe(nprobe),
              nb(nb),
              presumeBlockDistancesSize(presumeBlockDistancesSize)
              {}
        InitInfo() : d(0), block_dim(0), workerCount(0), nlist(0), blockCount(0), nprobe(0), nb(0), presumeBlockDistancesSize(0)  {}
        void print() {
            cout << GREEN << "InitInfo:[";
            cout << "d:" << d;
            cout << ",nlist:" << nlist;
            cout << ",nprobe:" << nprobe;
            cout << ",worker:" << workerCount;
            cout << ",block_dim:" << block_dim;
            cout << ",nb:" << nb;
            cout << ",presume:" << presumeBlockDistancesSize / 1000000000 << "GB";
            cout << "]";
            cout << endl;
        }
    };
    // 代表一次搜索请求中需要包含的信息

    size_t rank = 0;
    size_t blockSize = 0;
    size_t nq = 0, k = 0;
    // std::unique_ptr<IVF[]> ivfs;      // nlist个聚类,聚类内部的向量维度是block_dim
    std::unique_ptr<size_t[]> listSizes;         // nlist个聚类的向量数
    // vector<std::unique_ptr<float[]>> listCodes;  // nlist个聚类的向量数
    std::unique_ptr<float[]> querys;             // nq个查询向量, 维度是block_dim
    // std::unique_ptr<SearchBlock[]> blocks;       // 按照搜索顺序排列，第1个先搜索
    std::unique_ptr<idx_t[]> blockSearchOrder;   // search block的顺序，第i个元素是第i个要进行search的blockId
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe 查询向量相近的聚类id
    std::unique_ptr<idx_t[]> queryCompareSize;  // nq * nprobe 查询向量相近的聚类id
    // 为了计算第q个向量的distancesForQueryies的偏移量,偏移量是queryCompareSizePreSum[q]
    std::unique_ptr<idx_t[]> queryCompareSizePreSum;

    vector<std::unique_ptr<float[]>> distanceHeap;
    vector<std::unique_ptr<idx_t[]>> idHeap;

    // std::unique_ptr<float[]> heapTops;
    vector<std::unique_ptr<float[]>> distancesForBlocks;
    size_t blockDistancesSize;
    double waitTime = 0, searchTime = 0;
    // size_t presumeNq = 370, presumeK = 100;
    size_t presumeK = 100;

    // 每一块的计算结果的临时存储
    class DistanceBufferPool {
        vector<std::unique_ptr<float[]>> distancesForBlocks; //实际存储缓冲区
        vector<int> blockIds; //distancesForBlocks每一个的blockid是什么
        vector<bool> used; // distancesForBlocks是否正在被使用
        size_t spareBlock = 0; //可用的缓冲区数
        bool useDynamicAlloc = false; //如果为真则distancesForBlocks的下标不代表blockid， 否则代表
        // size_t size = 0;
        int MAXCHUNKSIZE;

        struct Action {
            bool use;
            size_t blockId;
            idx_t compareSize;
            size_t sender;
            Action(size_t blockId) : blockId(blockId) {
                use = true;
            }
            Action(size_t blockId, idx_t compareSize, size_t sender) 
            : blockId(blockId), compareSize(compareSize), sender(sender) {
                use = false;
            }
        };
        queue<Action> waitList; //bool代表是不是use

        int generateChunkTag(size_t blockId, int chunkIndex) {
            // 使用一个足够大的系数，确保 chunkIndex 在一个 block 内不会冲突
            // 并且不超出 MPI_TAG_UB（一般最小是 32767）
            const int chunkMultiplier = 10000;
            int tag = static_cast<int>(blockId) * chunkMultiplier + chunkIndex;
            if (chunkMultiplier < chunkIndex) {
                cerr << "Error: chunkMultiplier < chunkIndex" << chunkMultiplier<< " < " << chunkIndex << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // 可选：你可以检查 tag 是否超过 MPI_TAG_UB（最大支持的 tag）
            int tag_ub;
            int found;
            MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &found);
            if (found && tag > tag_ub) {
                cerr << "Error: MPI tag exceeds MPI_TAG_UB: " << tag << " > " << tag_ub << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            return tag;
        }
        void IRecvSplit(float* buffer, size_t blockId, idx_t compareSize, size_t sender, vector<MPI_Request>& reqs) {
            if(reqs.size() != 0) {
                cerr << "what the fuck" << endl;
            }
            if (compareSize > MAXCHUNKSIZE) {
                cout << YELLOW << format("WARNING: block {} compare size {} too big, split into {}", blockId, compareSize, (compareSize + MAXCHUNKSIZE - 1) / MAXCHUNKSIZE) << RESET << endl;

                idx_t sizeToReceive = compareSize;
                int chunkIndex = 0;
                int offset = 0;

                while (sizeToReceive > MAXCHUNKSIZE) {
                    MPI_Request req;
                    int tag = generateChunkTag(blockId, chunkIndex);

                    MPI_Irecv(buffer + offset, MAXCHUNKSIZE, MPI_FLOAT, sender, tag, MPI_COMM_WORLD, &req);
                    // cout << format("IRecvSplit worker {} tag {}", worker->rank, tag) << endl; 
                    reqs.push_back(req);

                    sizeToReceive -= MAXCHUNKSIZE;
                    offset += MAXCHUNKSIZE;
                    chunkIndex++;
                }

                // 最后一块
                MPI_Request req;
                int tag = generateChunkTag(blockId, chunkIndex);
                MPI_Irecv(buffer + offset, sizeToReceive, MPI_FLOAT, sender, tag, MPI_COMM_WORLD, &req);
                // cout << format("IRecvSplit worker {} last tag {}", worker->rank, tag) << endl; 
                reqs.push_back(req);

            } else {
                MPI_Request req;
                int tag = generateChunkTag(blockId, 0);
                MPI_Irecv(buffer, compareSize, MPI_FLOAT, sender, tag, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
        }
        


        // void IRecvSplit(float* buffer, size_t blockId, idx_t compareSize, size_t sender, vector<MPI_Request>& reqs) {
        //     if(compareSize > INT_MAX) {
        //         cout << YELLOW << format("WARNING: block {} compare size too big , split into {}", blockId, compareSize / INT_MAX + 1) << RESET << endl;
        //         idx_t sizeToSend = compareSize;
        //         while(sizeToSend > INT_MAX) {
        //             MPI_Request req;
        //             //TODO 这样不行，Irecv顺序不一样
        //             MPI_Irecv(buffer + compareSize - sizeToSend, INT_MAX, MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &req);
        //             reqs.push_back(req);
        //             sizeToSend -= INT_MAX;
        //         }
        //         MPI_Irecv(buffer + compareSize - sizeToSend, sizeToSend, MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &reqs[0]);
        //     } else {
        //         MPI_Irecv(buffer, compareSize, MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &reqs[0]);
        //     }
        // }
        Worker* worker;
    public:

        DistanceBufferPool(const InitInfo& info, Worker* worker) : worker(worker) {
            // distancesForBlocks = vector<std::unique_ptr<float[]>>(info.blockCount);
            MAXCHUNKSIZE = INT_MAX;
            distancesForBlocks = vector<std::unique_ptr<float[]>>();
            waitList = queue<Action>();
            try {
                for (size_t i = 0; i < info.blockCount; i++) {
                    distancesForBlocks.push_back(std::make_unique<float[]>(info.presumeBlockDistancesSize));
                }
                // for (size_t i = 0; i < info.blockCount; i++) {
                //     distancesForBlocks[size] = std::make_unique<float[]>(info.presumeBlockDistancesSize);
                //     size++;
                //     // cout << format("buffer {}", size) << endl;
                // }
                cout << YELLOW << format("use Normal Alloc") << RESET << endl;
            } catch (const std::bad_alloc& e) { 
                useDynamicAlloc = true;
                // blockIds = vector<int>(size, -1);
                // used = vector<bool>(size, false);
                // spareBlock = size;
                blockIds = vector<int>(distancesForBlocks.size(), -1);
                used = vector<bool>(distancesForBlocks.size(), false);
                spareBlock = distancesForBlocks.size();
                cout << YELLOW << format("use Dynamic Alloc") << RESET << endl;
            }
            // for (size_t i = 0; i < info.blockCount / 2; i++) {
            //     distancesForBlocks[size] = std::make_unique<float[]>(info.presumeBlockDistancesSize);
            //     size++;
            //     // cout << format("buffer {}", size) << endl;
            // }
            // useDynamicAlloc = true;
            cout << format("buffer size {}", distancesForBlocks.size()) << endl;
        }
        float* getBuffer(size_t blockId) { //获取blockId的缓冲区
            if(!useDynamicAlloc) {
                return distancesForBlocks[blockId].get();
            }
            for (int i = 0; i < distancesForBlocks.size(); i++) {
                if(used[i] && (blockIds[i] == blockId)) {
                    return distancesForBlocks[i].get();
                }
            }
            cerr << RED << format("getBuffer {} not found", blockId) << RESET << endl;
            return nullptr;
        }

        bool IRecv(size_t blockId, idx_t compareSize, size_t sender, vector<MPI_Request>& reqs) {
            if(!useDynamicAlloc) {
                IRecvSplit(distancesForBlocks[blockId].get(), blockId, compareSize, sender, reqs);
                return true;
            } else {
                if(spareBlock <= 0) {
                    waitList.push(Action(blockId, compareSize, sender));
                    return false;
                } else {
                    for (int i = 0; i < distancesForBlocks.size(); i++) {
                        if(used[i] == false) {
                            used[i] = true;
                            blockIds[i] = blockId;
                            // MPI_Irecv(distancesForBlocks[i].get(), compareSize, MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &reqs[0]);
                            IRecvSplit(distancesForBlocks[i].get(), blockId, compareSize, sender, reqs);
                            cout << format("Irecv {}", blockId) << endl;
                            spareBlock--;
                            return true;
                        }
                    }
                }
                return false;
                // cerr << ""
            }
        }
        void ISendSplit(const float* buffer, size_t blockId, idx_t compareSize, size_t receiver, std::vector<MPI_Request>& reqs) {
            cout << YELLOW << format("WARNING: block {} ISendSplit, split into {}", blockId, (compareSize + MAXCHUNKSIZE - 1) / MAXCHUNKSIZE) << RESET << endl;
            idx_t sizeLeft = compareSize;
            idx_t offset = 0;
            int chunkIndex = 0;

            while (sizeLeft > 0) {
                int chunkSize = static_cast<int>(std::min<idx_t>(sizeLeft, MAXCHUNKSIZE));
                int tag = generateChunkTag(blockId, chunkIndex);
                MPI_Request req;

                MPI_Isend(buffer + offset, chunkSize, MPI_FLOAT, receiver, tag, MPI_COMM_WORLD, &req);
                // cout << format("ISendSplit worker {} tag {}", worker->rank, tag) << endl; 
                reqs.push_back(req);

                offset += chunkSize;
                sizeLeft -= chunkSize;
                chunkIndex++;
            }
        }
        bool use(size_t blockId) { //要求分配缓冲区给blockId的block，返回分配是否成功
            if(!useDynamicAlloc) {
                return true;
            } else {
                if(spareBlock <= 0) {
                    waitList.push(Action(blockId));
                    return false;
                } 
                for (int i = 0; i < distancesForBlocks.size(); i++) {
                    if(used[i] == false) {
                        used[i] = true;
                        blockIds[i] = blockId;
                        cout << format("use {}", blockId) << endl;
                        spareBlock--;
                        return true;
                    }
                }
                return false;
            }
        }
        bool releaseBuffer(size_t blockId) { //释放缓冲区
            if(!useDynamicAlloc) {
                return true;
            } else {
                for (int i = 0; i < distancesForBlocks.size(); i++) {
                    if(used[i] && (blockIds[i] == blockId)) {
                        used[i] = false;
                        blockIds[i] = -1;
                        cout << format("release {}", blockId) << endl;
                        spareBlock++;
                        if(!waitList.empty()) {
                            auto action = waitList.front();
                            if(action.use) {
                                use(action.blockId);
                            } else {
                                IRecv(action.blockId, action.compareSize, action.sender, worker->disRequests[action.blockId]);
                            }
                            waitList.pop();
                        }
                        return true;
                    }
                }
                return false;
            }
        }
    };
    std::unique_ptr<DistanceBufferPool> distanceBufferPool;

    // idx_t presumeBlockDistancesSize;
    InitInfo info;

    void addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer);

    // MPI_Comm worker_comm;

    //vector的每一个元素对应一个block
    // vector<MPI_Request> infoRequests;
    vector<vector<MPI_Request>> disRequests;
    vector<vector<MPI_Request>> sendRequests;
    // vector<MPI_Request> sendRequests;
    vector<MPI_Request> sendDistanceRequests;
    vector<MPI_Request> sendIdRequests;
    
    // vector<MPI_Status> statuses;

    std::unique_ptr<idx_t[]> sendNextWorker; //接受到blockId为i的块的时候，应该发送给rank为sendNextWorker[i]的机器, sendNextWorker[i]=0说明可以发给master
    std::unique_ptr<idx_t[]> recvPrevWorker; //blockId为i的块应该从rank为recvPrevWorker[i]的机器接收，如果recvPrevWorker[i] = 0, 说明不需要接收, 直接可以算

    MyStopWatch uniWatch;

    std::unique_ptr<Index> index;

    bool blockSend = false;
    bool cut = false;
    std::unique_ptr<float[]> heapTops;
    vector<double> skipRates;


    void init(int rank, bool blockSend);
    bool shouldSendHeap(size_t blockId) {
        return  sendNextWorker[blockId] == 0;
    }
   
    void addQuerys(const float* querys, size_t nq) {
        // this->querys = std::make_unique<float[]>(nq * info.block_dim);
        copy_n_partial_vector(querys, this->querys.get(), info.d, info.block_dim, info.block_dim * (rank - 1), nq);
    }

    void search(bool cut);

    idx_t getTotalQueryCompareSize(size_t blockId) {
        size_t queryStart = blockId * blockSize;
        idx_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        // if(totalQueryCompareSize > INT_MAX) {
        //     // cerr << RED << "increase block size" << RESET << endl;
        //     cerr << YELLOW << "warning : totalQueryCompareSize > INT_MAX" << RESET << endl;
        //     // cout << totalQueryCompareSize << endl;
        //     // throw std::invalid_argument("increase block size");
        // }
        return totalQueryCompareSize;
    }

    void postSearch() {
        if(cut) {
            MPI_Send(skipRates.data(), info.blockCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); 
        }
    }

    idx_t totalSkip = 0;
    idx_t totalCompare = 0;

    void searchBlock(size_t blockId, bool cut);
    
};

class BaseWorker {
private:
public:
    struct InitInfo {
        size_t d, workerCount;
        size_t nlist;
        size_t nprobe, nb;
        size_t startIVFId, ivfCount;
        InitInfo(size_t d, size_t workerCount, size_t nlist, size_t nprobe,
                 size_t nb, size_t beginIVF, size_t ivfCount)
            : d(d),
              workerCount(workerCount),
              nlist(nlist),
              nprobe(nprobe),
              nb(nb),
              startIVFId(beginIVF),
              ivfCount(ivfCount)
              {}
        InitInfo() : d(0), workerCount(0), nlist(0), nprobe(0), nb(0), startIVFId(0),
              ivfCount(0) {}
        void print() {
            cout << GREEN << "InitInfo:[";
            cout << "d:" << d;
            cout << ",nlist:" << nlist;
            cout << ",nprobe:" << nprobe;
            cout << ",worker:" << workerCount;
            cout << ",nb:" << nb;
            cout << ",startIVFId:" << startIVFId;
            cout << ",ivfCount:" << ivfCount;
            cout << "]";
            cout << endl;
        }
    };
    // 代表一次搜索请求中需要包含的信息

    size_t rank = 0;
    MetricType metric;
    size_t nq = 0;
    size_t k = 0;
    std::unique_ptr<size_t[]> listSizes;         // nlist个聚类的向量数
    vector<std::unique_ptr<float[]>> listCodes;  // nlist个聚类的向量数
    vector<std::unique_ptr<size_t[]>> listIds;  // nlist个聚类的向量数
    std::unique_ptr<float[]> querys;             // nq个查询向量, 维度是block_dim
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe 查询向量相近的聚类id
    // std::unique_ptr<size_t[]> queryCompareSize;  
    // std::unique_ptr<size_t[]> queryCompareSizePreSum;
    std::unique_ptr<float[]> distances;
    std::unique_ptr<idx_t[]> labels;
    std::unique_ptr<float[]> heapTops;

    InitInfo info;

    MyStopWatch uniWatch;

    std::unique_ptr<Index> index;
    // Index* index;

    double waitTime = 0, searchTime = 0;

    bool cut = false;

    // void init(int rank, tribase::Index* index) {
    void init(int rank, MetricType metric);
    
    
    void single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries, float* heapTops);
    void single_thread_search_fast(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries);
    // void single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels) {

    //     std::unique_ptr<IVFScanBase> scaner = std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(info.d, k)); 

    //     //下面四个向量都和i绑定，也就是和每一个查询绑定
    //     float* disi = distances; //结果，查询向量最近的k个向量的距离
    //     idx_t* idxi = labels; //结果，查询向量最近的k个向量的id
    //     idx_t* listids = listidqueries.get();             // 单个查询对应的IVF聚类中心id
    //     MyStopWatch w;
    //     for (size_t i = 0; i < n; i++) {
    //         //每一个i对应一个查询
    //         scaner->set_query(queries + i * info.d);

    //         for (size_t j = 0; j < info.nprobe; j++) {
    //             //在第j个聚类中搜索所有点

    //             idx_t ivfId = listids[j];
    //             if(!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
    //                 continue;
    //             }
    //             idx_t index = ivfId - info.startIVFId;
    //             size_t listSize = listSizes[index];
    //             float* codes = listCodes[index].get();
    //             size_t* ids = listIds[index].get();


    //             // MyStopWatch wa;
    //             scaner->lite_scan_codes(listSize, codes, ids, disi, idxi);
    //             // if(i == 3) {
    //                 // cout << RED << listSize << RESET << endl;
    //                 // wa.print(format("q {} ivf {}", i, j), false);
    //             // }
    //         }
    //         disi += k;
    //         idxi += k;
    //         // w.print(format("q {}", i));
    //     }


    //     // std::unique_ptr<IVFScanBase> scaner = std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(info.d, k)); 

    //     // //下面四个向量都和i绑定，也就是和每一个查询绑定
    //     // float* disi = distances; //结果，查询向量最近的k个向量的距离
    //     // idx_t* idxi = labels; //结果，查询向量最近的k个向量的id
    //     // idx_t* listids = listidqueries.get();//单个查询对应的聚类中心id

    //     // for (size_t i = 0; i < n; i++) {
    //     //     //每一个i对应一个查询
    //     //     scaner->set_query(queries + i * d);
    //     //     //获取最近的nprobe个聚类中心
    //     //     for (size_t j = 0; j < nprobe; j++) {
    //     //         //在第j个聚类中搜索所有点
    //     //         //list代表聚类
    //     //         // IVF& list = lists[listids[j]];
    //     //         idx_t ivfId = listids[j];
    //     //         idx_t index = ivfId - info.startIVFId;
    //     //         size_t listSize = listSizes[index];
    //     //         float* codes = listCodes[index].get();
    //     //         size_t* ids = listIds[index].get();

    //     //         //查询点到中心的距离
    //     //         float centroid2query = centroids2query[j];
    //     //         //聚类中的点的数量
    //     //         size_t list_size = list.get_list_size();

    //     //         size_t scan_begin = 0;
    //     //         size_t scan_end = list_size;

    //     //         scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(),
    //     //                         list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
    //     //                         list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
    //     //                         list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(),
    //     //                         list.get_sub_farest_IP_dis(), list.get_sub_nearest_L2_id(),
    //     //                         list.get_sub_nearest_L2_dis(), nullptr, disi, idxi, stats,
    //     //                         centroid_codes.get() + listids[j] * d);
    //     //     }
    //     //     sort_result(METRIC_L2, k, disi, idxi);
    //     //     disi += k;
    //     //     idxi += k;
    //     //     listids += nprobe;
    //     // }
    // }

    void search(bool cut);
//     void search() {
//         uniWatch.print(format("node {} start search", rank), false);

//         size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), nq);
//         size_t batch_size = nq / nt;
//         size_t extra = nq % nt;
//         cout << nt << endl;
// #pragma omp parallel for num_threads(nt)
//         for (size_t i = 0; i < nt; i++) {
//             size_t start, end;
//             if (i < extra) {
//                 start = i * (batch_size + 1);
//                 end = start + batch_size + 1;
//             } else {
//                 start = i * batch_size + extra;
//                 end = start + batch_size;
//             }
//             if (start < end) {
//                 // end - start是查询向量的数量，queries + start * d是查询向量的起始位置，distance + start * k
//                 // 是结果存放的起始位置

//                 // cout << format("thread {} {}", omp_get_thread_num(), start) << endl;
//                 MyStopWatch w;
//                 single_thread_search_simple(end - start, querys.get() + start * info.d, k, distances.get() + start * k, labels.get() + start * k);
//                 // w.print(format("single thread {}", i));
//             }
//         }
// // #pragma omp parallel for schedule(dynamic)
// //         for(size_t q = 0; q < nq; q++) {
// //             // size_t nt = omp_get_num_threads();
// //             // cout << nt << endl;
// //             // auto scaner = scaners[omp_get_thread_num()].get();
// //             // scaner->set_query(querys.get() + q * info.d);
// //             // float* simi = distances.get() + q * k;
// //             // idx_t* idxi = labels.get() + q * k;
// //             // float* query = querys.get() + q * info.d;
// //             for (size_t i = 0; i < info.nprobe; i++) {
// //                 idx_t ivfId = listidqueries[q * info.nprobe + i];
// //                 // if(!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
// //                 //     continue;
// //                 // }
// //                 idx_t index = ivfId - info.startIVFId;
// //                 size_t listSize = listSizes[index];
// //                 // float* codes = listCodes[index].get();
// //                 // size_t* ids = listIds[index].get();
// //                 // scaner->lite_scan_codes(listSize, codes, ids, distances.get() + q * k, labels.get() + q * k);
// //                 for (size_t j = 0; j < listSize; j++) {
// //                     //和每一个待比较向量进行比较，每一个i和一个待比较向量绑定
// //                     // const float* candicate = codes + j * info.d;
// //                     // float dis = calculatedEuclideanDistance(query, candicate, info.d);
// //                     // if (dis < simi[0]) {
// //                     //     //比堆顶
// //                     //     heap_replace_top<METRIC_L2>(k, simi, idxi, dis, ids[j]);
// //                     //     // simi[0] = dis;
// //                     // }
// //                         //比堆顶
// //                     heap_replace_top<METRIC_L2>(k, distances.get(), labels.get(), 0, 0);
// //                         // simi[0] = dis;
// //                 }
// //             }
// //         }
//         uniWatch.print(format("node {} search", rank), false);
//         MPI_Send(distances.get(), k * nq, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
//         MPI_Send(labels.get(), k * nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD); 
//         uniWatch.print(format("node {} send", rank), false);
//     }
    
};


class GroupWorker {
private:
public:
    struct InitInfo {
        const size_t d, block_dim, workerCount;
        const size_t nlist;
        const size_t blockCount, nprobe, nb;
        const idx_t presumeBlockDistancesSize;
        const size_t groupCount, teamCount, teamSize;
        const size_t startIVFId, ivfCount;
        const size_t teamId, rankInsideTeam;

        // Constructor with default values for all fields
        InitInfo(size_t d = 0, size_t block_dim = 0, size_t workerCount = 0, size_t nlist = 0, size_t blockCount = 0,
                size_t nprobe = 0, size_t nb = 0, idx_t presumeBlockDistancesSize = 0, size_t groupCount = 0,
                size_t teamCount = 0, size_t teamSize = 0, size_t startIVFId = 0, size_t ivfCount = 0, 
                size_t teamId = 0, size_t rankInsideTeam = 0)
            : d(d), block_dim(block_dim), workerCount(workerCount), nlist(nlist), blockCount(blockCount), 
            nprobe(nprobe), nb(nb), presumeBlockDistancesSize(presumeBlockDistancesSize), groupCount(groupCount), 
            teamCount(teamCount), teamSize(teamSize), startIVFId(startIVFId), ivfCount(ivfCount), 
            teamId(teamId), rankInsideTeam(rankInsideTeam) {}

        // Default constructor with all values set to zero
        // InitInfo()
        //     : d(0), block_dim(0), workerCount(0), nlist(0), blockCount(0), nprobe(0), nb(0),
        //     presumeBlockDistancesSize(0), groupCount(0), teamCount(0), teamSize(0), 
        //     startIVFId(0), ivfCount(0), teamId(0), rankInsideTeam(0) {}

        // Print method
        void print() const {
            cout << GREEN << "InitInfo: [";
            cout << "d:" << d;
            cout << ", nlist:" << nlist;
            cout << ", nprobe:" << nprobe;
            cout << ", worker:" << workerCount;
            cout << ", block_dim:" << block_dim;
            cout << ", nb:" << nb;
            cout << ", presume:" << presumeBlockDistancesSize / 1000000000 << "GB"; // Presumed size in GB
            cout << ", groupCount:" << groupCount;
            cout << ", teamCount:" << teamCount;
            cout << ", teamSize:" << teamSize;
            cout << ", startIVFId:" << startIVFId;
            cout << ", ivfCount:" << ivfCount;
            cout << ", teamId:" << teamId;
            cout << ", rankInsideTeam:" << rankInsideTeam;
            cout << "]";
            cout << "\033[0m" << endl;  // Reset color after printing
        }
    };
    
    struct Group {
    };
    static int getTag(size_t groupId, size_t blockId, size_t blockCount) {
        return groupId * blockCount + blockId;
    }
    static int getDistanceHeapTag(size_t groupId) {
        return groupId * 2;
    }
    static int getIdHeapTag(size_t groupId) {
        return groupId * 2 + 1;
    }
    // 代表一次搜索请求中需要包含的信息

    size_t rank = 0;
    size_t blockSize = 0;
    size_t groupSize = 0;
    size_t nq = 0, k = 0;

    std::unique_ptr<size_t[]> listSizes;         // nlist个聚类的向量数
    std::unique_ptr<float[]> querys;             // nq个查询向量, 维度是block_dim
    // std::unique_ptr<idx_t[]> blockSearchOrder;   // search block的顺序，第i个元素是第i个要进行search的blockId
    std::unique_ptr<idx_t[]> listidqueries;      // nq * nprobe 查询向量相近的聚类id
    std::unique_ptr<idx_t[]> queryCompareSize;  // nq * nprobe 查询向量相近的聚类id
    // 为了计算第q个向量的distancesForQueryies的偏移量,偏移量是queryCompareSizePreSum[q]
    std::unique_ptr<idx_t[]> queryCompareSizePreSum;

    vector<std::unique_ptr<float[]>> distanceHeap; //vector的每一个指针代表一个group的所有block的heap
    vector<std::unique_ptr<idx_t[]>> idHeap;

    vector<vector<std::unique_ptr<float[]>>> distancesForBlocks;
    size_t blockDistancesSize;
    double waitTime = 0, searchTime = 0;
    size_t presumeK = 100;
    // size_t presumeNq = 370, presumeK = 100;
    

    // idx_t presumeBlockDistancesSize;
    InitInfo info;

    void addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer);

    // MPI_Comm worker_comm;

    //vector的每一个元素对应一个block
    vector<vector<MPI_Request>> disRequests;
    vector<MPI_Request> sendRequests;
    vector<MPI_Request> sendDistanceRequests;
    vector<MPI_Request> sendIdRequests;
    
    // std::unique_ptr<idx_t[]> sendNextWorker; //接受到blockId为i的块的时候，应该发送给rank为sendNextWorker[i]的机器, sendNextWorker[i]=0说明可以发给master
    // std::unique_ptr<idx_t[]> recvPrevWorker; //blockId为i的块应该从rank为recvPrevWorker[i]的机器接收，如果recvPrevWorker[i] = 0, 说明不需要接收, 直接可以算
    SearchOrder groupSearchOrder, blockSearchOrder;

    MyStopWatch uniWatch;

    std::unique_ptr<Index> index;

    bool blockSend = false;
    bool cut = false;
    bool minorCut = false;

    std::unique_ptr<float[]> heapTops;

    vector<double> skipRates;


    void init(int rank, bool blockSend);
    bool shouldSendHeap(size_t blockId) {
        return blockSearchOrder.sendNextWorker[info.rankInsideTeam][blockId] == 0;
    }
    void receiveQuery();
   
    void addQuerys(const float* querys, size_t nq) {
        // this->querys = std::make_unique<float[]>(nq * info.block_dim);
        copy_n_partial_vector(querys, this->querys.get(), info.d, info.block_dim, info.block_dim * (info.rankInsideTeam - 1), nq);
    }

    void search(bool cut, bool minorCut);
    

    idx_t getBlockQueryCompareSize(size_t groupId, size_t blockId) {
        size_t queryStart = getQueryOffset(groupId, blockId);
        idx_t totalQueryCompareSize = queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart];
        // if((double)queryCompareSizePreSum[queryStart + blockSize] - queryCompareSizePreSum[queryStart] > INT_MAX) {
        //     cerr << RED << "increase block size" << RESET << endl;
        //     throw std::invalid_argument("increase block size");
        // }
        return totalQueryCompareSize;
    }
    idx_t getQueryOffset(size_t groupId, size_t blockId) {
        return groupId * groupSize + blockId * blockSize;
    }

    void postSearch() {
        if(cut) {
            MPI_Send(skipRates.data(), info.blockCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); 
        }
    }

    idx_t totalSkip = 0;
    idx_t totalCompare = 0;

    
    // void print() {
    //     cout << groupSize << " " << blockSize << endl;
    // }
private:
    void searchGroup(idx_t groupId);
    void searchBlock(size_t blockId, size_t groupId);
    void reset() {
        // for (size_t i = 0; i < info.blockCount; i++) {
        //     init_result(METRIC_L2, blockSize * k, distanceHeap[i].get(), idHeap[i].get());
        // }
        //distancefornblocks
        disRequests = vector<vector<MPI_Request>>(info.blockCount);
        for(int i = 0; i < disRequests.size(); i++) {
            disRequests[i] = vector<MPI_Request>(1);
        }
        // waitTime = searchTime = 0;
    }
    size_t getSender(idx_t groupId, idx_t blockId);
    size_t getReceiver(idx_t groupId, idx_t blockId);
    void single_thread_searchBlock(size_t n, size_t blockId, size_t groupId, float* queries, float* distanceBuffer, float* simi, idx_t* idxi, idx_t* listidqueries);
};



}  // namespace tribase
#endif