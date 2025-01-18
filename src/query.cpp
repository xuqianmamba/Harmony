#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <mpi.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tribase.h"
#include "utils.h"

using namespace tribase;
using namespace std;


int workerMain(int rank, bool cut, SearchMode mode, bool blockSend) {
    
    if(mode == SearchMode::DIVIDE_VECTOR) {
        BaseWorker worker; 
        worker.init(rank);
        worker.search();
    } else if (mode == SearchMode::DIVIDE_DIM) {
        Worker node;
        node.init(rank, blockSend);
        // node.uniWatch.print("workerMain", false);
        node.search(cut);
        node.postSearch();
    } else {
        GroupWorker worker;
        worker.init(rank, blockSend);
        worker.receiveQuery();
        worker.search(cut);
        // node.postSearch();
    }
    return 0;
}
int main(int argc, char* argv[]) {
    
    argparse::ArgumentParser program("tribase");

    program.add_argument("--benchmarks_path")
        .help("benchmarks path");
    program.add_argument("--dataset").help("dataset name").default_value(std::string("msong"));
    program.add_argument("--input_format").help("format of the dataset").default_value(std::string("fvecs"));
    program.add_argument("--output_format").help("format of the output").default_value(std::string("bin"));
    program.add_argument("--k")
        .help("number of nearest neighbors")
        .default_value(100ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nprobes")
        .default_value(std::vector<size_t>({0ul}))
        .nargs(0, 100)
        .help("number of clusters to search")
        .scan<'u', size_t>();
    program.add_argument("--train_only").default_value(false).implicit_value(true).help("train only");
    program.add_argument("--cache").default_value(false).implicit_value(true).help("use cached index");
    program.add_argument("--metric").default_value("l2").help("metric type");
    program.add_argument("--run_faiss").default_value(false).implicit_value(true).help("run faiss");
    program.add_argument("--loop").default_value(1ul).action(
        [](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nlist").default_value(0ul).action(
        [](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--verbose").default_value(false).implicit_value(true).help("verbose");
    program.add_argument("--csv").help("csv result file").default_value(std::string(""));
    program.add_argument("--dataset_info")
        .help("only output dataset-info to csv file")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--early_stop").help("early stop").default_value(false).implicit_value(true);
    program.add_argument("--block")
        .help("number of blocks")
        .default_value(0ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--warmup_list_size")
        .help("how many vectors in a list are used to warmup heap")
        .default_value(0ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--warmup_list")
        .help("how many lists are used to warmup heap")
        .default_value(0ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--prune")
        .help("set pruning enabled")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--disableOrderOpt")
        .help("disable block search order optimization")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--inBalance")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--blockSend")
        .help("use blocking MPI_Send instead of unblocking MPI_Isend with search phase")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--fullWarmUp")
        .help("use groundtruth to warmup heap")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--group")
        .help("The number of groups the nq query vectors need to be divided into.")
        .default_value(1ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--team")
        .help("The number of teams the workers need to be divided into. Each team handles an individual part of base vectors")
        .default_value(1ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--mode")
           .help("The mode of search")
           .default_value(std::string("original"))
           .choices("vector", "group", "dim", "original"); // Only these choices are valid
    program.add_argument("--HardInBalance")
        .help("enable hard inBalance")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--HardInBalanceRatio")
        .help("control how inbalanced the search would be, 0.0 is perfectly balanced")
        .default_value(1.0f)
        .scan<'f', float>();
    program.add_argument("--HardInBalanceTeam")
        .help("number of Hard InBalance Team")
        .default_value(0ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });


    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    

    int pro;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &pro);
    
    // Get the rank (ID) of the current process
    int rank, workerCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &workerCount);  
    workerCount--;
    

    bool disableOrderOptimize = program.get<bool>("disableOrderOpt");
    bool prune = program.get<bool>("prune");
    bool run_faiss = program.get<bool>("run_faiss");
    bool inBalance = program.get<bool>("inBalance");
    bool hardInBalance = program.get<bool>("HardInBalance");
    size_t hardInBalanceTeam = program.get<size_t>("HardInBalanceTeam");
    float inBalanceRatio = program.get<float>("HardInBalanceRatio");
    if((hardInBalance && hardInBalanceTeam == 0) || (hardInBalanceTeam != 0 && !hardInBalance) || inBalanceRatio > 1.0) {
        cerr << "hardInBalance and hardInBalanceTeam" << endl;
        exit(1);
    }
    bool blockSend = program.get<bool>("blockSend");
    bool fullWarmUp = program.get<bool>("fullWarmUp");

    size_t groupCount = program.get<size_t>("group");
    size_t teamCount = program.get<size_t>("team");
    size_t teamSize = workerCount / teamCount;
    size_t blockCount = program.get<size_t>("block");

    std::string mode = program.get<std::string>("--mode");
    SearchMode searchMode;
    // if(mode == "original") {
    //     cout << YELLOW << "Mode: Original" << RESET << endl;
    //     searchMode = Index::SearchMode::ORIGINAL;
    //     if(workerCount > 0) {
    //         cerr << "worker should be 0" << endl;
    //         return 1;
    //     }
    // } else if (mode == "block") {
    //     cout << YELLOW << "Mode: Block" << RESET << endl;
    //     searchMode = Index::SearchMode::DIVIDE_DIM;
    // } else if (mode == "base") {
    //     cout << YELLOW << "Mode: Baseline" << RESET << endl;
    //     groupCount = workerCount;
    //     searchMode = Index::SearchMode::DIVIDE_IVF;
    // } else {
    //     cout << YELLOW << "Mode: Group" << RESET << endl;
    //     searchMode = Index::SearchMode::DIVIDE_GROUP;
    // }


    if(workerCount % teamCount != 0) {
        cerr << "worker should divide team count" << endl;
        return 1;
    }
    if(groupCount < teamCount) {
        cerr << "groupCount < teamCount" << endl;
        return 1;
    }
    // if(workerCount > 0) {
    //     if(groupCount > 0) {
    //         searchMode = Index::SearchMode::DIVIDE_GROUP;
    //         if(teamCount <= 0 || blockCount <= 0 || workerCount <= 0) {
    //             cerr << "provide --team and --block and workers" << endl;
    //             exit(1);
    //         }
    //     } else if(divideIVF) {
    //         searchMode = Index::SearchMode::DIVIDE_IVF;
    //     } else if(blockCount > 0) {
    //         searchMode = Index::SearchMode::DIVIDE_DIM;
    //     }
    // } else {
    //     searchMode = Index::SearchMode::ORIGINAL;
    // }
    
    // bool groupVersion = groupCount > 0; 
    // bool block_version = (blockCount > 0);
    // if (blockCount > 0 && (blockCount != 0 && workerCount == 0 || blockCount == 0 && workerCount != 0)) {
    //     throw std::invalid_argument("block count and node cound must be provided at the same time");
    // }


    if (rank != 0) {
        if(!run_faiss) {
            MPI_Barrier(MPI_COMM_WORLD);
            workerMain(rank, prune, searchMode, blockSend);
        }
    } else {
        // MyStopWatch watch(true, "queryWatch", CRAN);
        // if (job_id && node_list && num_nodes) {
        //     std::cout << "Job ID: " << job_id << std::endl;
        //     std::cout << "Nodes List: " << node_list << std::endl;
        //     std::cout << "Number of Nodes: " << num_nodes << std::endl;
        // } 

        std::cout << "Arguments passed to the program:" << std::endl;
        for (int i = 0; i < argc; i++) {
            std::cout << argv[i] << " ";
        }
        cout << CRAN << "master main, node count: " << workerCount << RESET << endl;
        
        std::vector<size_t> nprobes = program.get<std::vector<size_t>>("nprobes");
        // std::vector<std::string> opt_levels_str = program.get<std::vector<std::string>>("opt_levels");
        // std::vector<float> ratios = program.get<std::vector<float>>("ratios");

        size_t k = program.get<size_t>("k");

        // std::vector<OptLevel> opt_levels;
        // for (const auto& opt_level_str : opt_levels_str) {
        //     opt_levels.push_back(str2OptLevel(opt_level_str));
        // }
        // OptLevel added_opt_levels = OptLevel::OPT_NONE;

        std::string benchmarks_path = program.get<std::string>("benchmarks_path");
        std::string dataset = program.get<std::string>("dataset");
        std::string input_format = program.get<std::string>("input_format");
        std::string output_format = program.get<std::string>("output_format");
        std::string metric_str = program.get<std::string>("metric");
        MetricType metric;
        // size_t loop = program.get<size_t>("loop");
        size_t nlist = program.get<size_t>("nlist");
        bool verbose = program.get<bool>("verbose");
        // bool early_stop = program.get<bool>("early_stop");
        // bool block_version = program.get<bool>("block");
        // size_t nodeCount = program.get<size_t>("node");
        // bool sync = program.get<bool>("sync");
        

        std::cout << BLUE << "number of nodes : " << workerCount << RESET << std::endl;
        // if (early_stop && (ratios[0] != 1 || ratios.size() != 1)) {
        //     throw std::invalid_argument("early_stop is only allowed when ratios is 1.0");
        // }

        if (str_lower_equal(metric_str, "l2")) {
            metric = MetricType::METRIC_L2;
        } else if (str_lower_equal(metric_str, "ip")) {
            metric = MetricType::METRIC_IP;
        } else {
            throw std::runtime_error("Invalid metric type");
        }

        bool train_only = program.get<bool>("train_only");
        bool cache = program.get<bool>("cache");
        // float sub_nprobe_ratio = program.get<float>("sub_nprobe_ratio");

        // std::string inBalanceString = inBalance ? "InBalance" : "";
        std::string inBalanceString = inBalance ? "InBalance" : (hardInBalance ? "Hard" : "");
        std::string base_path = std::format("{}/{}/origin/{}_base.{}", benchmarks_path, dataset, dataset, input_format);
        std::string query_path =
            std::format("{}/{}/origin/{}_query.{}", benchmarks_path, dataset, dataset, input_format);
        std::string groundtruth_path =
            std::format("{}/{}/result/groundtruth_{}.{}", benchmarks_path, dataset, k, output_format);
            // std::format("{}/{}/result/groundtruth_{}{}.{}", benchmarks_path, dataset, k, inBalanceString, output_format);
        std::string log_path;
        std::string log_path_simple;
        std::string tmp_csv_path = program.get<std::string>("csv");
        if (tmp_csv_path.length()) {
            log_path = tmp_csv_path;
        } else {
            if (!run_faiss) {
                log_path = std::format("{}/{}/result/log.csv", benchmarks_path, dataset);
                log_path_simple = std::format("{}/{}/result/log_simple.csv", benchmarks_path, dataset);
            } else {
                // log_path = std::format("{}/{}/result/log_faiss.csv", benchmarks_path, dataset);
                log_path = std::format("{}/{}/result/log_faiss{}.csv", benchmarks_path, dataset, inBalanceString);
            }
        }

        // if (run_faiss && train_only) {
        //     throw std::invalid_argument("run_faiss && train_only is not allowed, run_faiss will not train at all.");
        // }

        // initialize base set
        size_t nb, d;
        std::unique_ptr<float[]> base = nullptr;
        std::tie(nb, d) = loadXvecsInfo(base_path);
        
        // 不用管
        if (program.get<bool>("dataset_info")) {
            auto [nq, _] = loadXvecsInfo(query_path);
            std::ofstream ofs;
            ofs.open(tmp_csv_path.data());
            if (!ofs.is_open()) {
                std::cerr << "Failed to open file: " << tmp_csv_path << std::endl;
                return 0;
            }
            ofs << "nb, nq, d" << std::endl;
            ofs << std::format("{}, {}, {}", nb, nq, d) << std::endl;
            return 0;
        }

        // init nlist, nprobe
        // sift10k, nb = 10000, nlist = 100
        if (nlist == 0) {
            nlist = static_cast<size_t>(std::sqrt(nb));
        }
        size_t warmUpSearchList = program.get<size_t>("warmup_list"); 
        size_t warmUpSearchListSize = program.get<size_t>("warmup_list_size"); 
        if(warmUpSearchList > 0 && warmUpSearchListSize == 0) {
            cout << RED << "set warmupSearchListSize to k" << RESET << endl;
            warmUpSearchListSize = k;
        }
        // size_t warmUpSearchListSize;
        // if(!program.present("--warmup_list_size")) {
        //     warmUpSearchListSize = k;
        // } else {
        //     warmUpSearchListSize = program.get<size_t>("--warmup_list_size"); 
        // }
        // size_t warmUpSearchNb;
        // if(program.present("--warmup")) {
        //     warmUpSearchNb = program.get<size_t>("warmup"); 
        // } else {
        //     size_t bigger = max(k, nlist);
        //     if(bigger % nlist == 0) {
        //         warmUpSearchNb = bigger;
        //     } else {
        //         warmUpSearchNb = (bigger / nlist + 1) * nlist;
        //     }
        // }
        if (nprobes.back() == 0) {
            nprobes.back() = nlist;
        }

        // size_t sub_nlist = std::sqrt(nb / nlist);
        // size_t sub_nprobe = std::max(static_cast<size_t>(sub_nlist * sub_nprobe_ratio), 1ul);
        // if (verbose) {
        //     std::cout << std::format("sub_nlist: {} sub_nprobe: {}", sub_nlist, sub_nprobe) << std::endl;
        // }

        // for (const OptLevel& opt_level : opt_levels) {
        //     added_opt_levels = static_cast<OptLevel>(static_cast<int>(added_opt_levels) | static_cast<int>(opt_level));
        // }
        // if (verbose) {
        //     std::cout << std::format("Added optimization levels: {}", static_cast<int>(added_opt_levels)) << std::endl;
        // }
        // nprobes.clear();
        // for (size_t val = 1; val <= nlist / 2; val *= 2) {
        //     nprobes.push_back(val);
        // }
        // nprobes.push_back(nlist);

        auto get_index_path = [&]() {
            // int target = static_cast<int>(added_opt_levels);
            // for (int i = 0; i < 8; i++) {
            //     // target is a subset of i
            //     if ((target & i) == target) {
            //         std::string index_path = std::format("{}/{}/index/index_nlist_{}_opt_{}_subNprobeRatio_{}.index",
            //                                              benchmarks_path, dataset, nlist, i, sub_nprobe_ratio);
            //         if (std::filesystem::exists(index_path)) {
            //             return index_path;
            //         }
            //     }
            // }
            return std::format("{}/{}/index/index_nlist_{}_opt_{}_subNprobeRatio_{}.index", benchmarks_path, dataset,
                               nlist, target, sub_nprobe_ratio);
        };

        auto get_faiss_index_path = [&]() {
            return std::format("{}/{}/index/faiss_index_nlist_{}.index", benchmarks_path, dataset, nlist);
        };

        std::string index_path = get_index_path();
        std::string faiss_index_path = get_faiss_index_path();
        prepareDirectory(faiss_index_path);

        // init query set
        auto [query, nq, _] = loadXvecs(query_path);

        cout << YELLOW << std::format("[dim:{}, nb:{}, nq:{}, k:{}, worker:{}]", d, nb, nq, k, workerCount) << RESET << endl; 

        if(blockCount > 0) {
            if (d % teamSize != 0) {
                cerr << RED << "Error: d % worker must be 0" << RESET << endl;
                return 1;
            }
            if (nq % (groupCount * blockCount) != 0) {
                cerr << RED << "Error: nq % block must be 0" << RESET << endl;
                return 1;
            }
            // if(divideIVF) {
            //     cerr << RED << "divideIVF and block cannot be provided together" << RESET <<endl;
            //     return 1;
            // }
        }
        srand(time(0));
        if(inBalance && !hardInBalance) {
            cout << YELLOW << "InBalance Query Set" << RESET << endl;
            for(int q = 1; q < nq; q++) {
                copy_n(query.get(), d, query.get() + q * d);
                float random_value = rand() / (float)RAND_MAX * 0.00001;
                for(int i = q * d; i < q * d + d; i++) {
                    query[i] += random_value;
                }
            }
            // for(int q = 0; q < nq; q++) {
            //     printVector(query.get() + q * d, d, BLUE);
            // }
        }



        if(workerCount > 0 && (run_faiss)) {
            cerr << RED << "Error: worker > 0" << RESET << endl;
            return 1;
        }
     

        // init groundtruth
        std::unique_ptr<idx_t[]> ground_truth_I = std::make_unique<idx_t[]>(k * nq);
        std::unique_ptr<float[]> ground_truth_D = std::make_unique<float[]>(k * nq);

        // init faiss_time file
        std::string faiss_time_path =
            std::format("{}/{}/result/faiss_result_nlist_{}.txt", benchmarks_path, dataset, nlist);
            // std::format("{}/{}/result/faiss_result_nlist_{}{}.txt", benchmarks_path, dataset, nlist, inBalanceString);
        std::vector<double> faiss_time(nprobes.size(), 0.0);
        std::ifstream faiss_time_input(faiss_time_path);
        if (faiss_time_input.is_open()) {
            size_t nprobe;
            double time;
            float recall, r2;
            while (faiss_time_input >> nprobe >> time >> recall >> r2) {
                auto it = std::find(nprobes.begin(), nprobes.end(), nprobe);
                if (it != nprobes.end()) {
                    faiss_time[std::distance(nprobes.begin(), it)] = time;
                }
            }
        } else {
            if (verbose) {
                std::cout << std::format("Faiss time file {} does not exist", faiss_time_path) << std::endl;
            }
        }

        faiss::IndexFlatL2 quantizer(d);
        std::unique_ptr<faiss::IndexIVFFlat> index_faiss = std::make_unique<faiss::IndexIVFFlat>(&quantizer, d, nlist);

        // train faiss and add base set to index_faiss
        // write to index file
        auto train_load_faiss = [&]() {
            if (!std::filesystem::exists(faiss_index_path)) {
            // if (true) {
                Stopwatch warch_faiss;
                if (verbose) {
                    std::cout << std::format("Training Faiss index") << std::endl;
                }
                if (base == nullptr) {
                    std::tie(base, nb, d) = loadXvecs(base_path);
                }
                warch_faiss.reset();
                index_faiss->train(nb, base.get());
                if (verbose) {
                    double faiss_train_elapsed = warch_faiss.elapsedSeconds(true);
                    std::cout << std::format("train: {:.2f}s", faiss_train_elapsed) << std::endl;
                    std::cout << std::format("Adding vectors to Faiss index") << std::endl;
                }
                index_faiss->add(nb, base.get());
                if (verbose) {
                    double faiss_add_elapsed = warch_faiss.elapsedSeconds(true);
                    std::cout << std::format("add: {:.2f}s", faiss_add_elapsed) << std::endl;
                }
                faiss::write_index(index_faiss.get(), faiss_index_path.c_str());
            } else {
                if (verbose) {
                    std::cout << std::format("Load faiss index from {}", faiss_index_path) << std::endl;
                }
                index_faiss.reset(dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(faiss_index_path.c_str())));
            }
        };

        // init groundtruth
        if (!std::filesystem::exists(groundtruth_path)) {
            double faiss_groundtruth_time = 0.0;
            if (verbose) {
                std::cout << std::format("Groundtruth file {} does not exist", groundtruth_path) << std::endl;
            }

            train_load_faiss();

            if (verbose) {
                std::cout << std::format("Searching Faiss index") << std::endl;
            }
            index_faiss->nprobe = nlist;
            Stopwatch stopwatch;
            // init ground_truth
            index_faiss->search(nq, query.get(), k, ground_truth_D.get(), ground_truth_I.get());
            faiss_groundtruth_time = stopwatch.elapsedSeconds();
            writeResultsToFile(ground_truth_I.get(), ground_truth_D.get(), nq, k, groundtruth_path);
            if (verbose) {
                std::cout << std::format("Groundtruth file {} created using {} s", groundtruth_path,
                                         faiss_groundtruth_time)
                          << std::endl;
            }
            if (nprobes.back() == nlist) {
                faiss_time.back() = faiss_groundtruth_time;
            }
        } else {
            if (verbose) {
                std::cout << std::format("Loading groundtruth file {}", groundtruth_path) << std::endl;
            }
            loadResults(groundtruth_path, ground_truth_I.get(), ground_truth_D.get(), nq, k);
            if (verbose) {
                std::cout << std::format("Groundtruth file loaded") << std::endl;
            }
        }

        // run_faiss train log_faiss.csv
        if (run_faiss) {
            if (verbose) {
                std::cout << std::format("Running Faiss") << std::endl;
            }
            if (!index_faiss->is_trained) {
                train_load_faiss();
            }
            std::ofstream faiss_time_output(faiss_time_path);
            if (!faiss_time_output.is_open()) {
                std::cerr << std::format("Fail to open {}\n", faiss_time_path);
            } else {
                if (verbose) {
                    std::cout << std::format("Output faiss time to {}\n", faiss_time_path);
                }
            }
            std::unique_ptr<float[]> tmp_faiss_dis = std::make_unique<float[]>(k * nq);
            std::unique_ptr<idx_t[]> tmp_faiss_labels = std::make_unique<idx_t[]>(k * nq);
            CsvWriter faiss_time_output_writer(log_path, {"dataset", "nlist", "nprobe", "time", "recall", "r2"}, true,
                                               false);
            for (size_t i = 0; i < nprobes.size(); i++) {
                index_faiss->nprobe = nprobes[i];
                if (loop > 1) {
                    index_faiss->search(nq, query.get(), k, tmp_faiss_dis.get(), tmp_faiss_labels.get());
                }
                Stopwatch stopwatch;
                for (size_t j = 0; j < loop; j++) {
                    index_faiss->search(nq, query.get(), k, tmp_faiss_dis.get(), tmp_faiss_labels.get());
                }
                float recall = calculate_recall(tmp_faiss_labels.get(), tmp_faiss_dis.get(), ground_truth_I.get(),
                                                ground_truth_D.get(), nq, k, metric);
                float r2 = calculate_r2(tmp_faiss_labels.get(), tmp_faiss_dis.get(), ground_truth_I.get(),
                                        ground_truth_D.get(), nq, k, metric);
                faiss_time[i] = stopwatch.elapsedSeconds() / loop;
                std::cout << std::format("Faiss nprobe: {} time: {} recall: {} r2: {}", nprobes[i], faiss_time[i],
                                         recall, r2)
                          << std::endl;
                faiss_time_output << std::format("{} {} {} {}", nprobes[i], faiss_time[i], recall, r2) << std::endl;
                faiss_time_output.flush();
                faiss_time_output_writer << dataset << nlist << nprobes[i] << faiss_time[i] << recall << r2
                                         << std::endl;
            }
            MPI_Finalize();
            return 0;
        }

        Index index;
        if (std::filesystem::exists(index_path) && cache) {
            // 直接使用已有的index文件，省去了train
            if (verbose) {
                std::cout << std::format("Loading index from {}", index_path) << std::endl;
            }
            index.load_index(index_path);
            if (verbose) {
                std::cout << std::format("Index loaded") << std::endl;
            }
        } else {
            // Index train, save index file
            std::tie(base, nb, d) = loadXvecs(base_path);
            nlist = static_cast<size_t>(std::sqrt(nb));
            index = Index(d, nlist, 0, metric, verbose);
            index.train(nb, base.get());
            // if (block_version) {
            //     index.add_simple(nb, base.get());
            // } else
                index.add(nb, base.get());

            if (verbose) {
                std::cout << std::format("Index trained") << std::endl;
            }
            index.save_index(index_path);
            if (verbose) {
                std::cout << std::format("Index saved to {}", index_path) << std::endl;
            }
        }

        if (train_only) {
            return 0;
        }

        

        auto doSearch = [&](auto nprobe, auto opt_level, auto ratio, auto early_stop_flag, auto f_time, float* distances, idx_t* labels, Index::Param* param) -> Stats {
           
            auto path = std::format("{}/{}/index/index_nlist_{}_{}.index", benchmarks_path, dataset,
                               nlist, to_string(param->mode));
            index.preSearch(nb, workerCount, blockCount, warmUpSearchList, warmUpSearchListSize, param, path);
            // if(!std::filesystem::exists(path)) {
            //     index.save_index(path, param->mode);
            // }
            // if (loop > 1) {
            //     index.search(nq, query.get(), k, distances, labels, ratio);
            // }
            Stopwatch stopwatch;
            Stats stats;
            // for (size_t j = 0; j < loop; j++) {
            stats = index.search(nq, query.get(), k, distances, labels, ratio);
            // }
            
            double search_time = stopwatch.elapsedSeconds() / loop;
            index.postSearch();
            float recall = calculate_recall(labels, distances, ground_truth_I.get(), ground_truth_D.get(),
                                            nq, k, metric);
            float r2 =
                calculate_r2(labels, distances, ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
            stats.simi_ratio = ratio;
            stats.nlist = nlist;
            stats.nprobe = nprobe;
            stats.query_time = search_time;
            stats.faiss_query_time = f_time;
            // stats.opt_level = opt_level;
            stats.recall = recall;
            stats.r2 = r2;
            stats.block = blockCount;
            stats.worker = workerCount;
            stats.disableOrderOptimize = disableOrderOptimize;
            // stats.divideIVF = divideIVF;
            stats.cut = prune;
            // stats.nodeList = node_list;
            stats.nb = nb;
            stats.nq = nq;
            stats.d = d;
            stats.mode = to_string(param->mode);
            stats.group = groupCount;
            stats.team = teamCount;
            stats.blockSend = blockSend;
            stats.inBalanceRatio = param->hardInBalanceRatio;
            
            if (recall == 1) {
                early_stop_flag = true;
            }
            return stats;
        };
        // search with different nprobe ,Opt level ,ratio
        for (size_t i = 0; i < nprobes.size(); i++) {
            size_t nprobe = nprobes[i];
            double f_time = faiss_time[i];
            index.nprobe = nprobe;
            bool early_stop_flag = false;
            for (const OptLevel& opt_level : opt_levels) {
                index.opt_level = opt_level;
                for (float ratio : ratios) {
                    std::unique_ptr<float[]> distancesB = std::make_unique<float[]>(nq * k);
                    std::unique_ptr<idx_t[]> labelsB = std::make_unique<idx_t[]>(nq * k);
                    std::unique_ptr<float[]> distances = std::make_unique<float[]>(nq * k);
                    std::unique_ptr<idx_t[]> labels = std::make_unique<idx_t[]>(nq * k);

                    Index::Param oriParam;
                    oriParam.mode = SearchMode::ORIGINAL;
                    Stats oriStat = doSearch(nprobe, opt_level, ratio, early_stop_flag, f_time, distances.get(), labels.get(), &oriParam);

                    std::cout << YELLOW;
                    oriStat.print();
                    std::cout << RESET;

                    // if(searchMode != Index::SearchMode::ORIGINAL) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    // }
                    // std::cout << YELLOW;

                    Index::Param param;
                    param.orderOptimize = !disableOrderOptimize;
                    param.mode = searchMode;
                    // param.period = period;
                    param.cut = prune;
                    param.fullWarmUp = fullWarmUp;
                    param.groupCount = groupCount;
                    param.teamCount = teamCount;
                    param.teamSize = teamSize;
                    param.hardInBalance = hardInBalance;
                    param.hardInBalanceTeam = hardInBalanceTeam;
                    param.hardInBalanceRatio = inBalanceRatio;
                    auto heapTops = std::make_unique<float[]>(nq); 
                    if(param.fullWarmUp) {
                        cout << YELLOW << "Full Warm Up!" << RESET << endl;
                        param.heapTops = heapTops.get();
                        for(int q = 0; q < nq; q++){
                            param.heapTops[q] = ground_truth_D[q * k + k - 1];
                            // printVector(ground_truth_D.get() + q * k, k, MAG);
                        }   
                        // printVector(param.heapTops, nq, BLUE);
                    }

                    // if (param.mode != Index::SearchMode::ORIGINAL) {
                    Stats stat = doSearch(nprobe, opt_level, ratio, early_stop_flag, f_time, distancesB.get(), labelsB.get(), &param);
                    // stat.blockVersionSpeedUpWithOriginal = 100.0 * oriStat.query_time / stat.query_time;
                    // cout << MAG << format("Speed up ratio compared to original version : {:.2f}", stat.blockVersionSpeedUpWithOriginal) << RESET << endl;
                    auto ivfCalculatedCount = vector<int>(nlist, 0);
                    for(int i = 0; i < nprobe * nq; i++) {
                        ivfCalculatedCount[index.listidqueries[i]]++;
                    }
                    double average = (double)nprobe * nq / nlist;
                    double variance = 0;
                    for(int i = 0; i < nlist; i++){
                        variance += (ivfCalculatedCount[i] - average) * (ivfCalculatedCount[i] - average);
                    }
                    variance /= nlist;
                    variance = sqrt(variance);
                    stat.variance = variance;
                    // printVector(ivfCalculatedCount, BLUE, format("方差 {} 平均{}",variance, average));
                    cout << MAG << format("Speed up ratio compared to faiss : {:.2f}", stat.blockVersionSpeedUpWithOriginal) << RESET << endl;
                    // stat.original_time = oriStat.query_time;
                    stat.original_time = stat.faiss_query_time;
                    stat.trainTime = index.trainTime;
                    stat.addTime = index.addTime;
                    stat.preSearchTime = index.preSearchTime;
                    stat.print();
                    // stat.myToCsv(log_path, true, dataset);
                    stat.myToCsv(log_path, true, dataset + inBalanceString);
                    // } 
                    // for(int i = 0; i < nq; i++) {
                        // if(diffVector(labels.get() + i * k, labelsB.get() + i * k, k)) {
                        // // if(true) {
                        //     std::cout << "Q" << i << " " << std::endl;
                        //     printVector(distances.get() + i * k, k, BLUE);
                        //     printVector(distancesB.get() + i * k, k, BLUE);
                        //     printVector(labels.get() + i * k, k, GREEN);
                        //     printVector(labelsB.get() + i * k, k, GREEN);
                        // }
                    // }
                    // std::cout << RESET;
                    // }
                }
            }
            if (early_stop_flag) {
                break;
            }
        }
    }
    MPI_Finalize();
    cout << CRAN << "return worker:" << rank << RESET << endl;
    return 0;
}