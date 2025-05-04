#include "node.h"
#include "Index.h"
namespace tribase {

// void Worker::addIVFs(vector<std::unique_ptr<float[]>>& listCodesBuffer) {
//     assert(rank != 0 && info.block_dim != 0);

//     listCodes = vector<std::unique_ptr<float[]>>(info.nlist);
//     // #pragma omp parallel for
//     for (size_t i = 0; i < info.nlist; i++) {
//         this->listCodes[i] = std::make_unique<float[]>(listSizes[i] * info.d);
//         copy_n_partial_vector(listCodesBuffer[i].get(), this->listCodes[i].get(), info.d, info.block_dim,
//                               (rank - 1) * info.block_dim, listSizes[i]);
//     }

// }

void Worker::init(int rank, bool blockSend) {
    MyStopWatch watch(true);

    this->rank = rank;
    this->blockSend = blockSend;

    // InitInfo
    MPI_Bcast(&info, sizeof(InitInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    info.print();

    index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);
    watch.print("index");

    // IVF的大小，IVF的向量表示
    listSizes = std::make_unique<size_t[]>(info.nlist);
    MPI_Bcast(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    watch.print("listSizes");

#pragma omp parallel for
    for (size_t i = 0; i < info.nlist; i++) {
        index->lists[i].reset(listSizes[i], info.block_dim, 0);
    }
    watch.print("lists");
        
    for (size_t i = 0; i < info.nlist; i++) {
        MPI_Bcast(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    //罪魁祸首, 诶这个为什么快
    auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
    for (size_t i = 0; i < info.nlist; i++) {
        listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.d);
        MPI_Bcast(listCodesBuffer[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // listCodes = vector<std::unique_ptr<float[]>>(info.nlist);
    for (size_t i = 0; i < info.nlist; i++) {
        copy_n_partial_vector(listCodesBuffer[i].get(), index->lists[i].candidate_codes.get(), info.d, info.block_dim,
                              (rank - 1) * info.block_dim, listSizes[i]);
    }
    // for (size_t i = 0; i < info.nlist; i++) {
    //     // copy_n_partial_vector(listCodesBuffer[i].get(), index->lists[i].candidate_codes.get(), info.d, info.block_dim,
    //     //                       (rank - 1) * info.block_dim, listSizes[i]);
    //     MPI_Recv(index->lists[i].candidate_codes.get(), listSizes[i] * info.block_dim , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    MPI_Bcast(index->centroid_codes.get(), info.nlist * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(index->centroid_ids.get(), info.nlist , MPI_INT64_T, 0, MPI_COMM_WORLD);

    watch.print("listCodes and listIds");

    // Search顺序
    blockSearchOrder = std::make_unique<idx_t[]>(info.blockCount);   // search block的顺序，第i个元素是第i个要进行search的blockId
    MPI_Recv(blockSearchOrder.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // cout << "Q" << rank << endl;
    // printVector(blockSearchOrder.get(), info.blockCount, BLUE);

    //提前malloc
    // std::unique_ptr<float[]> querysBuffer = std::make_unique<float[]>(presumeNq * info.d);
    index->originalQuery = std::make_unique<float[]>(presumeNq * info.d);
    this->querys = std::make_unique<float[]>(presumeNq * info.block_dim);
    listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // 最近的nprobe个聚类中心的id
    // index->heapTops = std::make_unique<float[]>(presumeNq);
    heapTops = std::make_unique<float[]>(presumeNq);
    queryCompareSize = std::make_unique<idx_t[]>(presumeNq);
    queryCompareSizePreSum = std::make_unique<idx_t[]>(presumeNq + 1);
    sendNextWorker = std::make_unique<idx_t[]>(info.blockCount);
    recvPrevWorker = std::make_unique<idx_t[]>(info.blockCount);
    // distanceHeap = std::make_unique<float[]>(presumeNq * presumeK);
    // idHeap = std::make_unique<idx_t[]>(presumeNq * presumeK);
    distanceHeap = vector<std::unique_ptr<float[]>>(info.blockCount);
    idHeap = vector<std::unique_ptr<idx_t[]>>(info.blockCount);
    for (size_t i = 0; i < info.blockCount; i++) {
        distanceHeap[i] = std::make_unique<float[]>(presumeK * presumeNq);
        idHeap[i] = std::make_unique<idx_t[]>(presumeK * presumeNq);
        init_result(METRIC_L2, presumeNq * presumeK, distanceHeap[i].get(), idHeap[i].get());
    }
    // init_result(METRIC_L2, presumeNq * presumeK, distanceHeap.get(), idHeap.get());
    
    watch.print("small malloc");
    blockDistancesSize = 2 * presumeNq / info.blockCount * info.nb;
    
    // distancesForBlocks = vector<std::unique_ptr<float[]>>(info.blockCount);
    // for (size_t i = 0; i < info.blockCount; i++) {
    //     distancesForBlocks[i] = std::make_unique<float[]>(blockDistancesSize);
    // }
    // blockSize = presumeNq / info.blockCount;
    // presumeBlockDistancesSize = presumeNq / info.blockCount * info.nb * 2;
    // cout << presumeBlockDistancesSize << "blockD" << endl;
    distanceBufferPool = std::make_unique<DistanceBufferPool>(info, this);
    // watch.print(format("node {} distancesForBlocks", rank));

    disRequests.clear();
    sendRequests.clear();
    disRequests.resize(info.blockCount);
    sendRequests.resize(info.blockCount);
    // disRequests = vector<vector<MPI_Request>>(info.blockCount);
    // sendRequests = vector<vector<MPI_Request>>(info.blockCount);
    // for(int i = 0; i < disRequests.size(); i++) {
    //     disRequests[i] = vector<MPI_Request>();
    //     sendRequests[i] = vector<MPI_Request>();
    // }
    // sendRequests = vector<MPI_Request>(info.blockCount);
    sendDistanceRequests = vector<MPI_Request>(info.blockCount);
    sendIdRequests = vector<MPI_Request>(info.blockCount);

    skipRates = vector<double>(info.blockCount, 0);

    //初始化sendNextWorker, recvPrevWorker
    MPI_Recv(sendNextWorker.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(recvPrevWorker.get(), info.blockCount, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


}
void Worker::search(bool cut) {
    this->cut = cut;

    MPI_Barrier(MPI_COMM_WORLD); //对应preSearch最后的barrier

    uniWatch = MyStopWatch(false, "uniWatch", MAG);

    uniWatch.print(format("node {} search() start", rank), false);
    // uniWatch.print(format("node {} cross barrier", rank), false);

    // nq, querys
    MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    if((double)nq * k > presumeK * presumeNq) {
        cerr << "presumeNq * k is too small" << endl;
        exit(1);
    }
    if(nq > presumeNq) {
        cerr << "presumeNq is too small" << endl;
        exit(1);
    }
    uniWatch.print(format("node {} nq k {} ", rank, k), false);
    // MPI_Bcast(querysBuffer.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(index->originalQuery.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} querys", rank), false);
    addQuerys(index->originalQuery.get(), nq);
    // addQuerys(querysBuffer.get(), nq);
    uniWatch.print(format("node {} addQuerys", rank), false);

    // query最近的nprobe个聚类中心的id
    // listidqueries = std::make_unique<idx_t[]>(nq * info.nprobe);  // 最近的nprobe个聚类中心的id
    MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} listidqueries", rank), false);

    // queryCompareSize,queryCompareSizePreSum
    MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
    // queryCompareSizePreSum = std::make_unique<size_t[]>(nq + 1);
    MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} queryCompareSize", rank), false);

    // 最大堆
    MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} heapTops", rank), false);

    // 其他初始化
    blockSize = nq / info.blockCount;
    
    // watch.print(format("Node {} Init", rank));
    uniWatch.print(format("node {} finish Receving", rank), false);


    MyStopWatch totalWatch(true, "total search watch");
    
    int searchedBlockCount = 0;
    vector<bool> isBlockSearched = vector<bool>(info.blockCount);

    //先给可以算的分配内存
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if(recvPrevWorker[blockId] == 0) {
            //直接标记使用
            distanceBufferPool->use(blockId);
        }
    }
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if(recvPrevWorker[blockId] != 0) {
            size_t sender = recvPrevWorker[blockId];
            // tag 当做blockId
            // MPI_Irecv(distancesForBlocks[blockId].get(), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId][0]);
            // MPI_Irecv(distanceBufferPool->getBuffer(blockId), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId]);
            bool suc = distanceBufferPool->IRecv(blockId, getTotalQueryCompareSize(blockId), sender, disRequests[blockId]);
            if(!suc) {
                break;
            }
            // cout << format("node({}) waiting for block({}) from node({})", rank, blockId, sender) << endl;
        } 
    }

    //不断查
    MyStopWatch watch(true);
    idx_t failCount = 0;
    while(searchedBlockCount < info.blockCount)
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if(isBlockSearched[blockId]) {
            continue;
        }
        if(recvPrevWorker[blockId] == 0) {
            // if(rank == 1) {

            // searchBlock(blockId, cut);
            searchBlock(blockId, cut);
            isBlockSearched[blockId] = true;
            searchedBlockCount++;

            watch.reset();

            // vector<size_t> unFinished;
            // vector<size_t> finished;
            // for(int i = 0; i < info.blockCount; i++) {
            //     if(isBlockSearched[i]) {
            //         int flag;
            //         MPI_Test(&sendRequests[i], &flag, MPI_STATUS_IGNORE);
            //         if(flag == false) {
            //             unFinished.push_back(i);
            //         } else {
            //             finished.push_back(i);
            //         }
            //     }
            // }
            // printVector(unFinished, RED, format("node{} unfinished", rank));
            // printVector(finished, GREEN, format("node{} finished", rank));

            if(blockSend && !shouldSendHeap(blockId)) {
                for(auto& req : sendRequests[blockId]) {
                    MPI_Wait(&req, MPI_STATUS_IGNORE);
                }
                // MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
                waitTime += watch.watch.elapsedSeconds();
                watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
            }

            // }

        } else {
            //应该检查是否应该释放缓冲区
            bool testFail = false;
            for(auto& req : disRequests[blockId]) {
                int isReceived;
                MPI_Status stat;
                MPI_Test(&req, &isReceived, &stat);
                if(!isReceived) {
                    testFail = true;
                    break;
                }
            }
            if(!testFail) {
                // cout << GREEN << format("node({}) received block({}) from node({})",rank, blockId, stat.MPI_SOURCE) << RESET << endl;
                waitTime += watch.watch.elapsedSeconds();
                watch.print(format("node {} waiting , now received block {} fail {}", rank, blockId, failCount));
                uniWatch.print(format("node {} 接收到了 block {} Test失败次数 {}", rank, blockId, failCount), false);

                searchBlock(blockId, cut);

                searchedBlockCount++;
                isBlockSearched[blockId] = true;
                failCount = 0;

                watch.reset();
                if(blockSend && !shouldSendHeap(blockId)) {
                    for(auto& req : sendRequests[blockId]) {
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }
                    // MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
                    waitTime += watch.watch.elapsedSeconds();
                    watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
                }
            } else {
                failCount++;
                usleep(100);
            }
        }
    }
    double totalTime = totalWatch.watch.elapsedSeconds();
    double otherTime = totalTime - waitTime - searchTime;
    totalWatch.print(format("Search Finished node({}), total skip:{:.1f}%, waitTime {} {:.2f}% searchTime {} {:.2f}% otherTime {} {:.2f}%", 
        rank, (double)totalSkip / totalCompare * 100, waitTime, 100 * waitTime / totalTime, searchTime, 100 * searchTime / totalTime, otherTime, 100 * otherTime / totalTime));

    if(!blockSend) {
        for(size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(!shouldSendHeap(blockId)) {
                cout << "sendRequests[blockId].size" << sendRequests[blockId].size() << endl;
                for(auto& req : sendRequests[blockId]) {
                    MPI_Wait(&req, MPI_STATUS_IGNORE);
                }
                // MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
            }
        }
    }
    // MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUS_IGNORE);
    // MPI_Waitall(sendDistanceRequests.size(), sendDistanceRequests.data(), MPI_STATUS_IGNORE);
    // MPI_Waitall(sendIdRequests.size(), sendIdRequests.data(), MPI_STATUS_IGNORE);
    

}
void Worker::searchBlock(size_t blockId, bool cut) {
    uniWatch.print(format("searchBlock 搜索开始 node({}) block({})", rank, blockId), false);
    MyStopWatch searchWatch(true, "searchBlock");


    float* distanceBuffer = distanceBufferPool->getBuffer(blockId);
    // float* distanceBuffer = distancesForBlocks[blockId].get();

    size_t queryStart = blockId * blockSize;
    idx_t totalQueryCompareSize = getTotalQueryCompareSize(blockId);
        
    // cout << RED << rank << "node search: blockId:" << blockId << " totalCompareSize:" << totalQueryCompareSize << RESET
    //      << endl;


    if (totalQueryCompareSize > blockDistancesSize) {
        cerr << "Error blockDistancesSize too small" << endl;
        exit(1);
    }
    // if(totalQueryCompareSize > INT_MAX) {
    //     cerr << "totalQueryCompareSize > INT_MAX : " << totalQueryCompareSize << endl;
    //     exit(1);
    // }

    // uniWatch.print(format("node {} index search start", rank), false);

    // Index::Param param;
    // param.mode = Index::SearchMode::DIVIDE_DIM_WORKER;
    // param.block_dim = info.block_dim;
    // param.queryCompareSize = queryCompareSize.get();
    // param.queryCompareSizePreSum = queryCompareSizePreSum.get();
    // param.queryStart = queryStart;
    // param.cut = cut;
    // param.listidqueries = listidqueries.get();

    size_t skip = 0;
    // index->search(blockSize, querys.get() + queryStart * info.block_dim, 0, distanceBuffer, NULL, 1, &param);

    

    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
//     // cout << "node " << rank << " nt = " << nt << endl;
#pragma omp parallel for num_threads(nt) reduction(+:skip)
// #pragma omp parallel for num_threads(nt)
    for (size_t q = queryStart; q < queryStart + blockSize; q++) {
        // cout << "node " << rank << " nt = " << omp_get_num_threads() << endl;
        size_t queryOffset =
            queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // 第q个查询的结果应该存的地址偏移量

        float* simi = distanceHeap[blockId].get() + k * (q - queryStart);
        idx_t* idxi = idHeap[blockId].get() + k * (q - queryStart);

        size_t curDistancePosition = 0;  // 在一个查询向量的结果内
        for (size_t i = 0; i < info.nprobe; i++) {
            idx_t ivfId = listidqueries[q * info.nprobe + i];
            for (size_t v = 0; v < listSizes[ivfId]; v++) {
                if(cut) {
                    // if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
                    if(distanceBuffer[queryOffset + curDistancePosition] == INFINITY) {
                        skip++;
                    } else {
                        float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                index->lists[ivfId].get_candidate_codes() + v * info.block_dim,
                                                                info.block_dim);
                        // cout << " " << queryOffset + curDistancePosition  << endl;
                        
                        // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                        distanceBuffer[queryOffset + curDistancePosition] += dis;
                        if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                            distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                        } else {
                            if(shouldSendHeap(blockId)) {
                                if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                                    //比堆顶
                                    heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                                }
                            } 
                        }
                    }
                } else {
                    float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                            index->lists[ivfId].get_candidate_codes() + v * info.block_dim,
                                                            info.block_dim);
                    // cout << " " << queryOffset + curDistancePosition  << endl;
                    assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                    distanceBuffer[queryOffset + curDistancePosition] += dis;
                    if(shouldSendHeap(blockId)) {
                        if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                            //比堆顶
                            // cout << format("q {} replace {} {} top {}", q, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v], simi[0]) << endl;
                            heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                        }
                    }
                }
                curDistancePosition++;
                
            }
        }
        // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
        assert(curDistancePosition == queryCompareSize[q]);
        sort_result(METRIC_L2, k, simi, idxi);
    }

    totalSkip += skip;
    skipRates[blockId] = (double)skip / totalQueryCompareSize * 100;
    totalCompare += totalQueryCompareSize;

    searchTime += searchWatch.watch.elapsedSeconds();
    searchWatch.print(format("node({}) search 主循环 block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100));

    uniWatch.print(format("worker({}) -> block({}) -> worker({}) 传输开始", rank, blockId, sendNextWorker[blockId]), false);
    MyStopWatch watch(true);
    if(shouldSendHeap(blockId)) {
        // cout << "send Heap" << blockId << endl;
        // printVector(idHeap[blockId].get(), blockSize * k, RED);
        // MPI_Isend(distanceHeap[blockId].get(), blockSize * k, MPI_FLOAT, sendNextWorker[blockId], 0, MPI_COMM_WORLD, &sendDistanceRequests[blockId]);
        // MPI_Isend(idHeap[blockId].get(), blockSize * k, MPI_INT64_T, sendNextWorker[blockId], 1, MPI_COMM_WORLD, &sendIdRequests[blockId]);
        MPI_Send(distanceHeap[blockId].get(), blockSize * k, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD);
        MPI_Send(idHeap[blockId].get(), blockSize * k, MPI_INT64_T, sendNextWorker[blockId], blockId, MPI_COMM_WORLD);


        // init_result(METRIC_L2, blockSize * k, distanceHeap.get(), idHeap.get());
    } else {


    // uniWatch.print(format("ready to perform node({}) -> block({}) -> node({})", rank, blockId, sendNextWorker[blockId]), false);
    // if(totalQueryCompareSize > INT_MAX) {
    //     cout << "> INT_MAX" << endl;
    //     idx_t sizeToSend = totalQueryCompareSize;
    //     while(sizeToSend > INT_MAX) {
    //         MPI_Request request;
    //         MPI_Isend(distanceBuffer + totalQueryCompareSize - sizeToSend, INT_MAX, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &request);
    //         sendRequests.push_back(request);
    //         sizeToSend -= INT_MAX;
    //     }
    //     MPI_Isend(distanceBuffer + totalQueryCompareSize - sizeToSend, sizeToSend, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &sendRequests[blockId]);
    // } else {
        // MPI_Isend(distanceBuffer, totalQueryCompareSize, MPI_FLOAT, sendNextWorker[blockId], blockId, MPI_COMM_WORLD, &sendRequests[blockId]);
        distanceBufferPool->ISendSplit(distanceBuffer, blockId, totalQueryCompareSize, sendNextWorker[blockId], sendRequests[blockId]);
    // }
    }
    waitTime += watch.watch.elapsedSeconds();
    watch.print(format("worker({}) -> block({}) -> worker({}) 传输时间", rank, blockId, sendNextWorker[blockId]));


    // distanceBufferPool->releaseBuffer(blockId); 不能直接release, 要检查buffer是不是已经发送完毕了


    // searchWatch.print(format("node({}) search finish block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100));

   

    // uniWatch.print(format("finish node({}) search block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100), false);

}

void BaseWorker::init(int rank) {
        MyStopWatch watch(true);

        this->rank = rank;
        // this->index = index;

        // 1.InitInfo
        MPI_Recv(&info, sizeof(InitInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // info.print();
        // IVF的ID通过startIVFid给出, 顺序排列

        index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

        watch.print("index");

        // 2.IVF的大小，IVF的向量表示
        listSizes = std::make_unique<size_t[]>(info.ivfCount);


        listCodes = vector<std::unique_ptr<float[]>>(info.ivfCount); //每个聚类对应的codes
        listIds = vector<std::unique_ptr<size_t[]>>(info.ivfCount); //每个聚类对应的ids

        watch.print("malloc");

        MPI_Recv(listSizes.get(), info.ivfCount * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        watch.print("listSize");

#pragma omp parallel for
        for (size_t i = 0; i < info.ivfCount; i++) {
            index->lists[i].reset(listSizes[i], info.d, 0);
        }

        for (size_t i = 0; i < info.ivfCount; i++) {
            listCodes[i] = std::make_unique<float[]>(listSizes[i] * info.d);
            listIds[i] = std::make_unique<size_t[]>(listSizes[i]);
        }
        watch.print("lists");

        size_t totalNb = 0;
        for(int i = 0; i < info.ivfCount; i++) {
            totalNb += listSizes[i];
        }


        for (size_t i = 0; i < info.ivfCount; i++) {
            MPI_Recv(listCodes[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            copy_n(listCodes[i].get(), listSizes[i] * info.d, index->lists[i].candidate_codes.get());
            MPI_Recv(listIds[i].get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            copy_n(listIds[i].get(), listSizes[i], index->lists[i].candidate_id.get());
            // MPI_Recv(curCodes, listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(curIds, listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // curCodes += listSizes[i] * info.d;
            // curIds += listSizes[i];
        }
        // for (size_t i = 0; i < info.ivfCount; i++) {
        //     MPI_Recv(index->lists[i].candidate_codes.get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     MPI_Recv(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // }
        watch.print("listCodes and listIds");

        MPI_Bcast(index->centroid_codes.get(), info.nlist * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(index->centroid_ids.get(), info.nlist , MPI_INT64_T, 0, MPI_COMM_WORLD);

        //提前malloc
        // int presumeNq = 1000;
        int presumeK = 100;
        this->querys = std::make_unique<float[]>(presumeNq * info.d);
        listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // 最近的nprobe个聚类中心的id
        // queryCompareSize = std::make_unique<size_t[]>(presumeNq);
        // queryCompareSizePreSum = std::make_unique<size_t[]>(presumeNq + 1);

        distances = std::make_unique<float[]>(presumeNq * presumeK);
        labels = std::make_unique<idx_t[]>(presumeNq * presumeK);
        init_result(METRIC_L2, presumeNq * presumeK, distances.get(), labels.get());

        MPI_Barrier(MPI_COMM_WORLD); //对应preSearch最后的barrier

        uniWatch = MyStopWatch(true, "uniWatch", MAG);
        uniWatch.print(format("node {} cross barrier", rank), false);

        // nq, k, querys
        MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
        if(nq * k > presumeK * presumeNq) {
            cerr << "presumeNq is too small" << endl;
            exit(1);
        }
        uniWatch.print(format("node {} nq", rank), false);
        MPI_Bcast(querys.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} querys", rank), false);

        // query最近的nprobe个聚类中心的id
        MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} listidqueries", rank), false);

        heapTops = std::make_unique<float[]>(presumeNq);
        MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
        uniWatch.print(format("node {} heapTops", rank), false);

        // queryCompareSize,queryCompareSizePreSum
        // queryCompareSize = std::make_unique<size_t[]>(nq);
        // MPI_Bcast(queryCompareSize.get(), nq, MPI_INT64_T, 0, MPI_COMM_WORLD);
        // MPI_Bcast(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, MPI_COMM_WORLD);
        // uniWatch.print(format("node {} queryCompareSize", rank), false);

        // 最大堆

        watch.print(format("Node {} Init", rank));
        uniWatch.print(format("node {} finish init", rank), false);

    }

void BaseWorker::search(bool cut) {
    this->cut = cut;
    MyStopWatch watch(true, "search watch");
    MyStopWatch totalWatch(true, "total watch");
    // uniWatch.print(format("node {} index search start", rank), false);

    // Index::Param oriParam;
    // oriParam.mode = Index::SearchMode::ORIGINAL;
    // oriParam.divideIVFVersionOriginal = true;
    // oriParam.startIVFId = info.startIVFId;
    // oriParam.ivfCount = info.ivfCount;

    // MyStopWatch watch(true);

    // index->search(nq, querys.get(), k, distances.get(), labels.get(), 1, &oriParam);

    // watch.print(format("node {} search", rank));

    // uniWatch.print(format("node {} search", rank), false);

    // MPI_Send(distances.get(), k * nq, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
    // MPI_Send(labels.get(), k * nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD); 

    // watch.print(format("node {} send", rank));
    // uniWatch.print(format("node {} send", rank), false);

    // int count = 0;
    // for(int i = 0; i < nq * info.nprobe; i++) {
    //     auto ivfId = listidqueries[i];
    //     if((ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
    //         count++;
    //     }
    // }
    // cout << format("node {} counted {}%", rank, 100.0 * count / nq / info.nprobe) << endl;

    // return ;

    uniWatch.print(format("node {} start search", rank), false);

        size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), nq);
        size_t batch_size = nq / nt;
        size_t extra = nq % nt;
        // cout << nt << endl;
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
                // end - start是查询向量的数量，queries + start * d是查询向量的起始位置，distance + start * k
                // 是结果存放的起始位置

                // cout << format("thread {} {}", omp_get_thread_num(), start) << endl;
                // MyStopWatch w;
                single_thread_search_simple(end - start, querys.get() + start * info.d, k, distances.get() + start * k, labels.get() + start * k, listidqueries.get() + start * info.nprobe, heapTops.get() + start);
                // single_thread_search_fast(end - start, querys.get() + start * info.d, k, distances.get() + start * k, labels.get() + start * k, listidqueries.get() + start * info.nprobe);
                // w.print(format("single thread {}", i));
            }
        }
// #pragma omp parallel for schedule(dynamic)
//         for(size_t q = 0; q < nq; q++) {
//             // size_t nt = omp_get_num_threads();
//             // cout << nt << endl;
//             // auto scaner = scaners[omp_get_thread_num()].get();
//             // scaner->set_query(querys.get() + q * info.d);
//             // float* simi = distances.get() + q * k;
//             // idx_t* idxi = labels.get() + q * k;
//             // float* query = querys.get() + q * info.d;
//             for (size_t i = 0; i < info.nprobe; i++) {
//                 idx_t ivfId = listidqueries[q * info.nprobe + i];
//                 // if(!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
//                 //     continue;
//                 // }
//                 idx_t index = ivfId - info.startIVFId;
//                 size_t listSize = listSizes[index];
//                 // float* codes = listCodes[index].get();
//                 // size_t* ids = listIds[index].get();
//                 // scaner->lite_scan_codes(listSize, codes, ids, distances.get() + q * k, labels.get() + q * k);
//                 for (size_t j = 0; j < listSize; j++) {
//                     //和每一个待比较向量进行比较，每一个i和一个待比较向量绑定
//                     // const float* candicate = codes + j * info.d;
//                     // float dis = calculatedEuclideanDistance(query, candicate, info.d);
//                     // if (dis < simi[0]) {
//                     //     //比堆顶
//                     //     heap_replace_top<METRIC_L2>(k, simi, idxi, dis, ids[j]);
//                     //     // simi[0] = dis;
//                     // }
//                         //比堆顶
//                     heap_replace_top<METRIC_L2>(k, distances.get(), labels.get(), 0, 0);
//                         // simi[0] = dis;
//                 }
//             }
//         }

    searchTime += watch.watch.elapsedSeconds();
    watch.print(format("node {} search", rank));

    uniWatch.print(format("node {} search", rank), false);
    MPI_Send(distances.get(), k * nq, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
    MPI_Send(labels.get(), k * nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD); 
    uniWatch.print(format("node {} send", rank), false);

    waitTime += watch.watch.elapsedSeconds();
    watch.print(format("node {} send", rank));

    double totalTime = totalWatch.watch.elapsedSeconds();
    totalWatch.print(format("Search Finished node({}), waitTime {} {:.2f}% searchTime {} {:.2f}%",
            rank, waitTime, 100 * waitTime / totalTime, searchTime, 100 * searchTime / totalTime));
            
    int count = 0;
    for(int i = 0; i < nq * info.nprobe; i++) {
        auto ivfId = listidqueries[i];
        if((ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
            count++;
        }
    }
    cout << format("node {} counted {}%", rank, 100.0 * count / nq / info.nprobe) << endl;
}

void BaseWorker::single_thread_search_fast(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries) {
    std::unique_ptr<IVFScanBase> scaner_quantizer = index->get_scanner(index->metric, OPT_NONE, info.nprobe);  // 搜索最近的聚类中心

    std::unique_ptr<float[]> centroid2queries =
        std::make_unique<float[]>(n * info.nprobe);  // n个查询向量到nprobe个聚类中心的距离
    // std::unique_ptr<idx_t[]> listidqueries = std::make_unique<idx_t[]>(n * info.nprobe);  // 最近的nprobe个聚类中心的id
    // init_result(index->metric, n * info.nprobe, centroid2queries.get(),
    //             listidqueries);  // 优先队列，存储离n个查询向量最近的nprobe个聚类中心
    // copy_n(this->listidqueries.get(), n * info.nprobe, listidqueries.get());

    // 下面四个向量都和i绑定，也就是和每一个查询绑定
    float* simi = distances;
    idx_t* idxi = labels;
    float* centroids2query = centroid2queries.get();  // 单个查询对应的距离
    idx_t* listids = listidqueries;             // 单个查询对应的IVF聚类中心id

    int calculatedCount = 0;

    for (size_t i = 0; i < n; i++) {
        // 每一个i对应一个查询
        // scaner_quantizer->set_query(queries + i * info.d);
        // // scaner->set_query(queries + i * info.d);
        // // 把和nlist个聚类中心计算距离的结果放进centroids2query这个堆里面
        // scaner_quantizer->lite_scan_codes(info.nlist, index->centroid_codes.get(),
        //                                   reinterpret_cast<const size_t*>(index->centroid_ids.get()),
        //                                   centroids2query,  // ret
        //                                   listids);         // ret , 分别对应两个堆
        // // 取前nprobe个聚类中心
        // sort_result(index->metric, info.nprobe, centroids2query, listids);
        // cout << i << endl;
        // printVector(listids, info.nprobe, BLUE);
        // printVector(this->listidqueries.get() + i * info.nprobe, info.nprobe, GREEN);

        // if (index->metric == MetricType::METRIC_L2) {
            for (size_t j = 0; j < info.nprobe; j++) {
                // 在第j个聚类中搜索所有点
                // list代表聚类
                idx_t ivfId = listids[j];
                if(!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
                    continue;
                }
                IVF& list = index->lists[listids[j] - info.startIVFId];

                // 查询点到中心的距离
                float centroid2query = centroids2query[j];
                // 聚类中的点的数量
                size_t list_size = list.get_list_size();

                // std::unique_ptr<bool[]> if_skip = std::make_unique<bool[]>(list_size);

                // size_t skip_count = 0;
                // size_t skip_count_large = 0;
                // size_t scan_begin = 0;
                // size_t scan_end = list_size;

                for (size_t v = 0; v < list_size; v++) {
                    const float* candicate = list.get_candidate_codes() + v * index->d;
                    float dis = 0;
                    dis = calculatedEuclideanDistance(queries + i * index->d, candicate, index->d);
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

                // IF_STATS {
                //     stats->skip_triangle_count += skip_count;
                //     stats->skip_triangle_large_count += skip_count_large;
                //     stats->total_count += list_size;
                // }

                // scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(),
                //                    list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
                //                    list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
                //                    list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(),
                //                    list.get_sub_farest_IP_dis(), list.get_sub_nearest_L2_id(),
                //                    list.get_sub_nearest_L2_dis(), if_skip.get(), simi, idxi, stats,
                //                    centroid_codes.get() + listids[j] * d);
            }
        // } 
        // sort_result(index->metric, k, simi, idxi);

        simi += k;
        idxi += k;
        centroids2query += info.nprobe;
        listids += info.nprobe;
    }
}
void BaseWorker::single_thread_search_simple(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, idx_t* listidqueries, float* heapTops) {
    //比fast要慢，是不是因为scaner?不是

    // cout << index->metric;
    // std::unique_ptr<IVFScanBase> scaner = std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_ENABLED>(info.d, k));  //用的是cal0
    // cout << format("d {} k {} nprobe {}", index->d, k, info.nprobe) << endl;
    // cout << "edge " << index->edge_device_enabled << "metric " << index->metric << endl;

    // std::unique_ptr<IVFScanBase> scaner = index->get_scanner(index->metric, OPT_NONE, k);  // 搜索最近的聚类中心 用的是cal非0
    

    //下面四个向量都和i绑定，也就是和每一个查询绑定
    float* simi = distances; //结果，查询向量最近的k个向量的距离
    idx_t* idxi = labels; //结果，查询向量最近的k个向量的id
    idx_t* listids = listidqueries;             // 单个查询对应的IVF聚类中心id
    size_t cutCount = 0, totalCount = 0;
    // MyStopWatch w;
    for (size_t i = 0; i < n; i++) {
        //每一个i对应一个查询
        // scaner->set_query(queries + i * info.d);
        // printVector(listids, info.nprobe, GREEN);
        for (size_t j = 0; j < info.nprobe; j++) {
            //在第j个聚类中搜索所有点

            idx_t ivfId = listids[j];
            if(!(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount)) {
                continue;
            }
            // idx_t index = ivfId - info.startIVFId;
            IVF& list = index->lists[ivfId - info.startIVFId];
            // size_t listSize = listSize[index];
            // float* codes = listCodes[index].get();
            // size_t* ids = listIds[index].get();
            size_t listSize = list.get_list_size();
            totalCount += listSize;
            float* codes = list.candidate_codes.get();
            size_t* ids = list.candidate_id.get();

            // MyStopWatch wa;
            // scaner->lite_scan_codes(listSize, codes, ids, simi, idxi);

            if(cut) {
                for (size_t v = 0; v < list.get_list_size(); v++) {
                    const float* candicate = list.get_candidate_codes() + v * index->d;
                    float dis = 0;
                    // dis = calculatedEuclideanDistance(queries + i * index->d, candicate, index->d);
                    int numCheck = 4;
                    size_t checkDim = index->d / numCheck; 
                    const float* query = queries + i * index->d;
                    for(int check = 0; check < numCheck; check++) {
                        if(check == numCheck - 1) {
                            dis += calculatedEuclideanDistance(query + checkDim * check, candicate + checkDim * check, index->d - check * checkDim);
                        } else {
                            dis += calculatedEuclideanDistance(query + checkDim * check, candicate + checkDim * check, checkDim);
                            if(dis > heapTops[i]) {
                                dis = INFINITY;
                                cutCount += numCheck - check - 1;
                                break;
                            }
                        }
                    }
                    if (dis < simi[0]) {
                        //比堆顶
                        heap_replace_top<METRIC_L2>(k, simi, idxi, dis, list.get_candidate_id()[v]);
                    }
                }
            } else {
                for (size_t v = 0; v < list.get_list_size(); v++) {
                    const float* candicate = list.get_candidate_codes() + v * index->d;
                    float dis = 0;
                    dis = calculatedEuclideanDistance(queries + i * index->d, candicate, index->d);
                    if (dis < simi[0]) {
                        //比堆顶
                        heap_replace_top<METRIC_L2>(k, simi, idxi, dis, list.get_candidate_id()[v]);
                    }
                }
            }

            // for (size_t v = 0; v < list.get_list_size(); v++) {
            //     const float* candicate = list.get_candidate_codes() + v * index->d;
            //     float dis = 0;
            //     dis = calculatedEuclideanDistance(queries + i * index->d, candicate, index->d);
            //     if (dis < simi[0]) {
            //         //比堆顶
            //         heap_replace_top<METRIC_L2>(k, simi, idxi, dis, list.get_candidate_id()[v]);
            //     }
            // }
            // if(i == 3) {
                // cout << RED << listSize << RESET << endl;
                // wa.print(format("q {} ivf {}", i, j), false);
            // }
        }
        simi += k;
        idxi += k;
        listids += info.nprobe;
        
        // w.print(format("q {}", i));
    }
    // cout << RED << format("{}%", double(cutCount) * 25 / totalCount)<< RESET << endl;
}
void GroupWorker::init(int rank, bool blockSend) {
    MyStopWatch watch(false);

    this->rank = rank;
    this->blockSend = blockSend;

    // InitInfo
    MPI_Recv(&info, sizeof(InitInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    info.print();

    index = std::make_unique<Index>(info.d, info.nlist, info.nprobe);

    // IVF的大小，IVF的向量表示
    listSizes = std::make_unique<size_t[]>(info.nlist);
    MPI_Recv(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#pragma omp parallel for
    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        index->lists[i].reset(listSizes[i], info.block_dim, 0);
    }
    watch.print("lists");
        
    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        MPI_Recv(index->lists[i].candidate_id.get(), listSizes[i] * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
    //     MPI_Recv(index->lists[i].candidate_codes.get(), listSizes[i] * info.block_dim , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    auto listCodesBuffer = vector<std::unique_ptr<float[]>>(info.nlist);
    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        listCodesBuffer[i] = std::make_unique<float[]>(listSizes[i] * info.d);
        MPI_Recv(listCodesBuffer[i].get(), listSizes[i] * info.d , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for (size_t i = info.startIVFId; i < info.startIVFId + info.ivfCount; i++) {
        copy_n_partial_vector(listCodesBuffer[i].get(), index->lists[i].candidate_codes.get(), info.d, info.block_dim,
                              (info.rankInsideTeam - 1) * info.block_dim, listSizes[i]);
    }

    groupSearchOrder = SearchOrder(info.teamCount, info.groupCount, true);
    blockSearchOrder = SearchOrder(info.teamSize, info.blockCount, true);
        // groupSearchOrder.print();
        // blockSearchOrder.print();

    //malloc内存

    index->originalQuery = std::make_unique<float[]>(presumeNq * info.d);
    this->querys = std::make_unique<float[]>(presumeNq * info.block_dim);
    listidqueries = std::make_unique<idx_t[]>(presumeNq * info.nprobe);  // 最近的nprobe个聚类中心的id
    // index->heapTops = std::make_unique<float[]>(presumeNq);
    heapTops = std::make_unique<float[]>(presumeNq);
    queryCompareSize = std::make_unique<idx_t[]>(presumeNq);
    queryCompareSizePreSum = std::make_unique<idx_t[]>(presumeNq + 1);
    // distanceHeap = vector<std::unique_ptr<float[]>>(info.blockCount);
    // idHeap = vector<std::unique_ptr<idx_t[]>>(info.blockCount);
    // for (size_t i = 0; i < info.blockCount; i++) {
    //     distanceHeap[i] = std::make_unique<float[]>(presumeK * presumeNq);
    //     idHeap[i] = std::make_unique<idx_t[]>(presumeK * presumeNq);
    //     init_result(METRIC_L2, presumeNq * presumeK, distanceHeap[i].get(), idHeap[i].get());
    // }
    distanceHeap = vector<std::unique_ptr<float[]>>(info.groupCount);
    idHeap = vector<std::unique_ptr<idx_t[]>>(info.groupCount);
    for (size_t i = 0; i < info.groupCount; i++) {
        distanceHeap[i] = std::make_unique<float[]>(presumeK * presumeNq);
        idHeap[i] = std::make_unique<idx_t[]>(presumeK * presumeNq);
        init_result(METRIC_L2, presumeNq * presumeK, distanceHeap[i].get(), idHeap[i].get());
    }
    
    blockDistancesSize = 2 * presumeNq / info.blockCount / info.groupCount * info.nb * info.nprobe / info.nlist;
    // blockDistancesSize = 2 * presumeNq / info.blockCount / info.groupCount * info.nb / info.teamSize * info.nprobe / info.nlist;
    distancesForBlocks = vector<vector<std::unique_ptr<float[]>>>(info.groupCount);
    for (size_t i = 0; i < distancesForBlocks.size(); i++) {
        distancesForBlocks[i] = vector<std::unique_ptr<float[]>>(info.blockCount);
        for (size_t j = 0; j < distancesForBlocks[i].size(); j++) {
            distancesForBlocks[i][j] = std::make_unique<float[]>(blockDistancesSize);
        }
    }


    disRequests = vector<vector<MPI_Request>>(info.blockCount);
    for(int i = 0; i < disRequests.size(); i++) {
        disRequests[i] = vector<MPI_Request>(1);
    }
    sendRequests = vector<MPI_Request>(info.blockCount);
    sendDistanceRequests = vector<MPI_Request>(info.blockCount);
    sendIdRequests = vector<MPI_Request>(info.blockCount);

    skipRates = vector<double>(info.blockCount, 0);

    // cout << "finish init" << endl;


}

void GroupWorker::receiveQuery() {

    MPI_Barrier(MPI_COMM_WORLD); //对应preSearch最后的barrier

    uniWatch = MyStopWatch(false, "uniWatch", MAG);
    uniWatch.print(format("node {} cross barrier", rank), false);
    // nq, querys
    MPI_Bcast(&nq, sizeof(nq), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, sizeof(k), MPI_BYTE, 0, MPI_COMM_WORLD);
    if((double)nq * k > presumeK * presumeNq) {
        cerr << "presumeNq * k is too small" << endl;
        exit(1);
    }
    if(nq > presumeNq) {
        cerr << "presumeNq is too small" << endl;
        exit(1);
    }
    // uniWatch.print(format("node {} nq k {} ", rank, k), false);

    MPI_Bcast(index->originalQuery.get(), nq * info.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    addQuerys(index->originalQuery.get(), nq);

    // uniWatch.print(format("node {} querys", rank), false);

    MPI_Bcast(listidqueries.get(), nq * info.nprobe, MPI_INT64_T, 0, MPI_COMM_WORLD);

    // uniWatch.print(format("node {} listidqueries", rank), false);

    // queryCompareSize,queryCompareSizePreSum
    MPI_Recv(queryCompareSize.get(), nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // queryCompareSizePreSum = std::make_unique<size_t[]>(nq + 1);
    MPI_Recv(queryCompareSizePreSum.get(), (nq + 1), MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // uniWatch.print(format("node {} queryCompareSize", rank), false);
    // printVector(queryCompareSize.get(), nq, BLUE);
    // printVector(queryCompareSizePreSum.get(), nq + 1, GREEN);

    groupSize = nq / info.groupCount;        
    blockSize = groupSize / info.blockCount;

    // cout << "finish receive query" << endl;
    uniWatch.print(format("node {} finish Receving", rank), false);

}
void GroupWorker::search(bool cut, bool minorCut) {

    this->cut = cut;
    this->minorCut = minorCut;

    MyStopWatch totalWatch(true, "Total Watch");

    // 初始最大堆
    MPI_Bcast(heapTops.get(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uniWatch.print(format("node {} 接收初始heapTops", rank), false);

    for(int i = 0; i < info.groupCount; i++) {
        size_t groupId = groupSearchOrder.workerSearchOrder[info.teamId][i];
        //如果应该接收更新的heapTops
        size_t sender = groupSearchOrder.recvPrevWorker[info.teamId][groupId];
        if(sender != 0) {
            MyStopWatch waitWatch(true, "和master之间的等待");
            MPI_Recv(heapTops.get() + groupId * groupSize, groupSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(listSizes.get(), info.nlist * sizeof(size_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(distanceHeap[groupId].get(), groupSize * k, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(idHeap[groupId].get()      , groupSize * k, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            waitTime += waitWatch.watch.elapsedSeconds();
            waitWatch.print(format("node {} 接收group {} Heap的时间", rank, groupId));
            uniWatch.print(format("node {} 接收到 group {}", rank, groupId), false);
            
        }

        searchGroup(groupId);

        reset();
    }

    // totalWatch.print(format("node {} 结束所有group搜索", rank));
    double totalTime = totalWatch.watch.elapsedSeconds();
    double otherTime = totalTime - waitTime - searchTime;
    totalWatch.print(format("node({}) 结束所有group搜索, waitTime {:.3f}s {:.2f}% searchTime {:.3f}s {:.2f}% otherTime {:.3f}s {:.2f}%", 
        rank, waitTime, 100 * waitTime / totalTime, searchTime, 100 * searchTime / totalTime, otherTime, 100 * otherTime / totalTime));

    
}
size_t GroupWorker::getSender(idx_t groupId, idx_t blockId) {
    if(blockSearchOrder.getRecvPrevWorker(info.rankInsideTeam, blockId) == 0) {
        return 0;
    }
    return blockSearchOrder.getRecvPrevWorker(info.rankInsideTeam, blockId) + (info.teamId - 1) * info.teamSize;
}
size_t GroupWorker::getReceiver(idx_t groupId, idx_t blockId) {
    if(blockSearchOrder.getSendNextWorker(info.rankInsideTeam, blockId) == 0) {
        return 0;
    }
    return blockSearchOrder.getSendNextWorker(info.rankInsideTeam, blockId) + (info.teamId - 1) * info.teamSize;
}

void GroupWorker::searchGroup(idx_t groupId) 
{
    // cout << format("node {} groupid {}", rank, groupId) << endl;
    uniWatch.print(format("node {} 开始searchGroup {}", rank, groupId), false);
    MyStopWatch groupWatch(true, "Group search watch");
   
    int searchedBlockCount = 0;
    vector<bool> isBlockSearched = vector<bool>(info.blockCount);

    //先给可以算的分配内存
    // for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
    //     if(blockSearchOrder.getRecvPrevWorker(info.rankInsideTeam, blockId) == 0) {
    //         //直接标记使用
    //         // distanceBufferPool->use(blockId);
    //     }
    // }

    //TODO FIX DISQUEST
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        size_t sender = getSender(groupId, blockId);
        if(sender != 0) {
            MPI_Irecv(distancesForBlocks[groupId][blockId].get(), getBlockQueryCompareSize(groupId, blockId), MPI_FLOAT, sender, getTag(groupId, blockId, info.blockCount), MPI_COMM_WORLD, &disRequests[blockId][0]);
            // MPI_Irecv(distanceBufferPool->getBuffer(blockId), getTotalQueryCompareSize(blockId), MPI_FLOAT, sender, blockId, MPI_COMM_WORLD, &disRequests[blockId]);
            // bool suc = distanceBufferPool->IRecv(blockId, getTotalQueryCompareSize(blockId), sender, disRequests[blockId]);
            // if(!suc) {
            //     break;
            // }
            // cout << format("node({}) waiting for block({}) from node({})", rank, blockId, sender) << endl;
        } 
    }

    //不断查
    MyStopWatch watch(true);
    idx_t failCount = 0;
    while(searchedBlockCount < info.blockCount)
    for (size_t blockId = 0; blockId < info.blockCount; blockId++) {
        if(isBlockSearched[blockId]) {
            continue;
        }
        size_t sender = getSender(groupId, blockId);
        if(sender == 0) {

            searchBlock(blockId, groupId);
            isBlockSearched[blockId] = true;
            searchedBlockCount++;

            watch.reset();
            if(blockSend && !shouldSendHeap(blockId)) {
                MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
                waitTime += watch.watch.elapsedSeconds();
                watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
            }

        } else {
            int isReceived;
            MPI_Status stat;
            bool testFail = false;
            for(auto& req : disRequests[blockId]) {
                MPI_Test(&req, &isReceived, &stat);
                if(!isReceived) {
                    testFail = true;
                    break;
                }
            }
            if(!testFail) {
                // cout << GREEN << format("node({}) received block({}) from node({})",rank, blockId, stat.MPI_SOURCE) << RESET << endl;
                waitTime += watch.watch.elapsedSeconds();
                watch.print(format("node {} waiting , now received block {} fail {}", rank, blockId, failCount));
                uniWatch.print(format("node {} 接收到了 block {} Test失败次数 {}", rank, blockId, failCount), false);

                searchBlock(blockId, groupId);

                searchedBlockCount++;
                isBlockSearched[blockId] = true;
                failCount = 0;

                watch.reset();
                if(blockSend && !shouldSendHeap(blockId)) {
                    MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
                    waitTime += watch.watch.elapsedSeconds();
                    watch.print(format("node({}) block {} MPI_Wait", rank, blockId));
                }
            } else {
                failCount++;
                usleep(100);
            }
        }
    }
    // double totalTime = groupWatch.watch.elapsedSeconds();
    // double otherTime = totalTime - waitTime - searchTime;
    // groupWatch.print(format("node({}) 搜索完了Group{}, 剪枝率:{:.1f}%, waitTime {:.3f}s {:.2f}% searchTime {:.3f}s {:.2f}% otherTime {:.3f}s {:.2f}%", 
    //     rank, groupId, (double)totalSkip / totalCompare * 100, waitTime, 100 * waitTime / totalTime, searchTime, 100 * searchTime / totalTime, otherTime, 100 * otherTime / totalTime));

    if(!blockSend) {
        for(size_t blockId = 0; blockId < info.blockCount; blockId++) {
            if(!shouldSendHeap(blockId)) {
                MPI_Wait(&sendRequests[blockId], MPI_STATUS_IGNORE);
            }
        }
    } 
}

void GroupWorker::single_thread_searchBlock(size_t n, size_t blockId, size_t groupId, float* queries, float* distanceBuffer, float* simi, idx_t* idxi, idx_t* listidqueries) {

    size_t curDistancePosition = 0;  // 在一个查询向量的结果内
    size_t skip = 0;
    for (size_t q = 0; q < n; q++) {
        for (size_t i = 0; i < info.nprobe; i++) {
            idx_t ivfId = listidqueries[q * info.nprobe + i];

            if(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount) {

                IVF& list = index->lists[ivfId];

                for (size_t v = 0; v < listSizes[ivfId]; v++) {

                    const float* codes = list.get_candidate_codes() + v * info.block_dim;

                    if(cut) {
                        // if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
                        if(distanceBuffer[curDistancePosition] == INFINITY) {
                            skip++;
                        } else {
                            float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                    codes,
                                                                    info.block_dim);

                            distanceBuffer[curDistancePosition] += dis;

                            if (distanceBuffer[curDistancePosition] > heapTops[q]) {
                                distanceBuffer[curDistancePosition] = INFINITY;
                            } else {
                                if(shouldSendHeap(blockId)) {
                                    if (distanceBuffer[curDistancePosition] < simi[0]) {
                                        //比堆顶
                                        heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                                    }
                                } 
                            }
                        }
                    } else {
                        float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                index->lists[ivfId].get_candidate_codes() + v * info.block_dim,
                                                                info.block_dim);
                        // cout << " " << queryOffset + curDistancePosition  << endl;
                        // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                        // distanceBuffer[queryOffset + curDistancePosition] += dis;
                        // if(info.blockCount != 1) {
                        //     dis += distanceBuffer[curDistancePosition];
                        // }
                        if(shouldSendHeap(blockId)) {
                            if (dis < simi[0]) {
                                //比堆顶
                                heap_replace_top<METRIC_L2>(k, simi, idxi, dis, index->lists[ivfId].get_candidate_id()[v]);
                            }
                        }
                    }
                    curDistancePosition++;
                    
                }
            }
        }
    }
}
void GroupWorker::searchBlock(size_t blockId, size_t groupId) {
    // uniWatch.print(format("node({}) 开始搜索 block({}) , group({})", rank, blockId, groupId), false);


    // float* distanceBuffer = distanceBufferPool->getBuffer(blockId);
    float* distanceBuffer = distancesForBlocks[groupId][blockId].get();

    size_t queryStart = getQueryOffset(groupId, blockId);
    idx_t totalQueryCompareSize = getBlockQueryCompareSize(groupId,blockId);
        
    // cout << RED << rank << "node search: blockId:" << blockId << " totalCompareSize:" << totalQueryCompareSize << RESET
    //      << endl;


    if (totalQueryCompareSize > blockDistancesSize) {
        
        cerr << format("Error blockDistancesSize too small {} < {}", blockDistancesSize , totalQueryCompareSize) << endl;
        exit(1);
    }

    MyStopWatch searchWatch(false, "searchBlock");

//     size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
//     size_t batch_size = blockSize / nt;
//     size_t extra = blockSize % nt;
// #pragma omp parallel for num_threads(nt)
//     for (size_t i = 0; i < nt; i++) {
//         size_t start, end;
//         if (i < extra) {
//             start = i * (batch_size + 1);
//             end = start + batch_size + 1;
//         } else {
//             start = i * batch_size + extra;
//             end = start + batch_size;
//         }
//         if (start < end) {
//             size_t q = queryStart + start;
//             float* simi = distanceHeap[groupId].get() + k * (start + blockId * blockSize);
//             idx_t* idxi = idHeap[groupId].get()       + k * (start + blockId * blockSize);
//             idx_t ivfId = listidqueries[q * info.nprobe + i];
//             size_t queryOffset = queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // 第q个查询的结果应该存的地址偏移量
//             single_thread_searchBlock(end - start, blockId, groupId, querys.get() + q * info.block_dim, distanceBuffer + queryOffset, simi, idxi, 
//                 listidqueries.get() + q * info.nprobe);
//         }
//     }
    size_t skip = 0;

    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), blockSize);
#pragma omp parallel for reduction(+:skip)
    for (size_t q = queryStart; q < queryStart + blockSize; q++) {
        size_t queryOffset =
            queryCompareSizePreSum[q] - queryCompareSizePreSum[queryStart];  // 第q个查询的结果应该存的地址偏移量

        float* query = querys.get() + q * info.block_dim;
        float* simi = distanceHeap[groupId].get() + k * (q - queryStart + blockId * blockSize);
        idx_t* idxi = idHeap[groupId].get()       + k * (q - queryStart + blockId * blockSize);
        size_t curDistancePosition = 0;  // 在一个查询向量的结果内

        for (size_t i = 0; i < info.nprobe; i++) {

            idx_t ivfId = listidqueries[q * info.nprobe + i];

            if(ivfId >= info.startIVFId && ivfId < info.startIVFId + info.ivfCount) {

                IVF& list = index->lists[ivfId];

                for (size_t v = 0; v < listSizes[ivfId]; v++) {

                    const float* codes = list.get_candidate_codes() + v * info.block_dim;

                    if(cut) {
                        // if(distancesForBlocks[blockId][queryOffset + curDistancePosition] == INFINITY) {
                        if(distanceBuffer[queryOffset + curDistancePosition] == INFINITY) {
                            skip++;
                        } else {

                            if(minorCut) {

                                float dis = distanceBuffer[queryOffset + curDistancePosition];
                                auto candicate = codes;
                                auto query = querys.get() + q * info.block_dim;
                                int numCheck = 4;
                                size_t checkDim = info.block_dim / numCheck; 

                                for(int check = 0; check < numCheck; check++) {
                                    if(check == numCheck - 1) {
                                        dis += calculatedEuclideanDistance(query + checkDim * check, candicate + checkDim * check, info.block_dim- check * checkDim);
                                    } else {
                                        dis += calculatedEuclideanDistance(query + checkDim * check, candicate + checkDim * check, checkDim);
                                        if(dis > heapTops[q]) {
                                            dis = INFINITY;
                                            // cutCount += numCheck - check - 1;
                                            break;
                                        }
                                    }
                                }

                                distanceBuffer[queryOffset + curDistancePosition] = dis;

                                if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                                    distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                                } else {
                                    if(shouldSendHeap(blockId)) {
                                        if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                                            //比堆顶
                                            heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                                        }
                                    } 
                                }
                            } else {
                                float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                                                                        codes,
                                                                        info.block_dim);

                                distanceBuffer[queryOffset + curDistancePosition] += dis;

                                if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                                    distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                                } else {
                                    if(shouldSendHeap(blockId)) {
                                        if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                                            //比堆顶
                                            heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                                        }
                                    } 
                                }
                            }
                            
                            // float dis = calculatedEuclideanDistance(querys.get() + q * info.block_dim,
                            //                                         codes,
                            //                                         info.block_dim);

                            // distanceBuffer[queryOffset + curDistancePosition] += dis;

                            // if (distanceBuffer[queryOffset + curDistancePosition] > heapTops[q]) {
                            //     distanceBuffer[queryOffset + curDistancePosition] = INFINITY;
                            // } else {
                            //     if(shouldSendHeap(blockId)) {
                            //         if (distanceBuffer[queryOffset + curDistancePosition] < simi[0]) {
                            //             //比堆顶
                            //             heap_replace_top<METRIC_L2>(k, simi, idxi, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v]);
                            //         }
                            //     } 
                            // }
                        }
                    } else {
                        float dis = calculatedEuclideanDistance(query,
                                                                index->lists[ivfId].get_candidate_codes() + v * info.block_dim,
                                                                info.block_dim);
                        // cout << " " << queryOffset + curDistancePosition  << endl;
                        // assert(queryOffset + curDistancePosition < totalQueryCompareSize);
                        // distanceBuffer[queryOffset + curDistancePosition] += dis;
                        distanceBuffer[queryOffset + curDistancePosition] += dis;
                        if(shouldSendHeap(blockId)) {
                            dis = distanceBuffer[queryOffset + curDistancePosition];
                            if (dis < simi[0]) {
                                //比堆顶
                                // cout << format("q {} replace {} {} top {}", q, distanceBuffer[queryOffset + curDistancePosition], index->lists[ivfId].get_candidate_id()[v], simi[0]) << endl;
                                heap_replace_top<METRIC_L2>(k, simi, idxi, dis, index->lists[ivfId].get_candidate_id()[v]);
                            }
                        }
                    }
                    curDistancePosition++;
                    
                }
            }
        }
        // cout << curDistancePosition << " " << queryCompareSize[q] << endl;
        // assert(curDistancePosition == queryCompareSize[q]);
        // sort_result(METRIC_L2, k, simi, idxi);
    }

    searchTime += searchWatch.watch.elapsedSeconds();
    searchWatch.print(format("node({}) 搜索主循环 block({}) group({}) skip:{:.1f}%", rank, blockId, groupId, (double)skip / totalQueryCompareSize * 100));

    totalSkip += skip;
    skipRates[blockId] = (double)skip / totalQueryCompareSize * 100;
    totalCompare += totalQueryCompareSize;


    uniWatch.print(format("worker({}) -> block({}) -> worker({}) 传输开始", rank, blockId, getReceiver(groupId, blockId)), false);
    MyStopWatch watch(true);
    int tag = GroupWorker::getTag(groupId, blockId, info.blockCount);
    if(shouldSendHeap(blockId)) {
        MPI_Send(distanceHeap[groupId].get() + blockId * blockSize * k, blockSize * k, MPI_FLOAT  , 0, tag, MPI_COMM_WORLD);
        MPI_Send(idHeap[groupId].get()       + blockId * blockSize * k, blockSize * k, MPI_INT64_T, 0, tag, MPI_COMM_WORLD);
        // init_result(METRIC_L2, blockSize * k, distanceHeap.get(), idHeap.get());
    } else {
        MPI_Isend(distanceBuffer, totalQueryCompareSize, MPI_FLOAT, getReceiver(groupId, blockId), tag, MPI_COMM_WORLD, &sendRequests[blockId]);
    }
    waitTime += watch.watch.elapsedSeconds();
    watch.print(format("worker({}) -> block({}) -> worker({}) 传输时间", rank, blockId, getReceiver(groupId, blockId)));


    // distanceBufferPool->releaseBuffer(blockId); 不能直接release, 要检查buffer是不是已经发送完毕了


    // searchWatch.print(format("node({}) search finish block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100));

   

    // uniWatch.print(format("finish node({}) search block({}) skip:{:.1f}%", rank, blockId, (double)skip / totalQueryCompareSize * 100), false);

}
}  // namespace tribase