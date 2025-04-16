#!/bin/bash


# 定义所有数据集及其nprobes列表
declare -A DATASET_NPROBES=(
    ["msong"]="997 500 400 300 200 100 70 50 40 30 20 10 5 3 1"
    ["sift1m"]="1000 500 400 300 200 100 70 50 40 30 20 10 5 3 1"
    ["glove1.2m"]="1092 900 700 500 400 300 200 100 50 30 20 10 5 3 1"
    ["glove2.2m"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
    ["deep1M"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
    ["word2vec"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
    ["StarLightCurves"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
    ["HandOutlines"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
)

# 定义所有数据集
DATASETS="msong sift1m glove1.2m glove2.2m deep1M word2vec StarLightCurves HandOutlines"

# 默认运行所有数据集
SELECTED_DATASETS="$DATASETS"
EXTRA_OPTIONS=""
NODES=5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets=*)
            SELECTED_DATASETS="${1#*=}"
            shift
            ;;
        --add-option=*)
            EXTRA_OPTIONS="${EXTRA_OPTIONS} ${1#*=}"
            shift
            ;;
        --nodes=*)
            NODES="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --datasets=dataset1,dataset2   Select which datasets to run (default: all)"
            echo "  --add-option=\"OPTION\"         Add an option to all sbatch commands"
            echo "  --nodes=N                     Set number of nodes (default: 5)"
            echo "  --help                         Show this help message"
            echo ""
            echo "Available datasets: $DATASETS"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 将逗号分隔的列表转换为空格分隔
SELECTED_DATASETS=$(echo "$SELECTED_DATASETS" | tr ',' ' ')

# 运行选定的数据集
for dataset in $SELECTED_DATASETS; do
    case "$dataset" in
        msong)
            echo "===== Submitting jobs for dataset: msong ====="
            for nprobe in ${DATASET_NPROBES[$dataset]}; do
                sbatch --nodes=$NODES run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=msong --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=msong --mode base --nprobes $nprobe $EXTRA_OPTIONS
            done
            ;;
        sift1m)
            echo "===== Submitting jobs for dataset: sift1m ====="
            for nprobe in ${DATASET_NPROBES[$dataset]}; do
                sbatch --nodes=$NODES run.slurm --dataset=sift1m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=sift1m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=sift1m --mode base --nprobes $nprobe $EXTRA_OPTIONS
            done
            ;;
        glove1.2m)
            echo "===== Submitting jobs for dataset: glove1.2m ====="
            for nprobe in ${DATASET_NPROBES[$dataset]}; do
                sbatch --nodes=$NODES run.slurm --dataset=glove1.2m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=glove1.2m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=glove1.2m --mode base --nprobes $nprobe $EXTRA_OPTIONS
            done
            ;;
        glove2.2m)
            echo "===== Submitting jobs for dataset: glove2.2m ====="
            for nprobe in ${DATASET_NPROBES[$dataset]}; do
                sbatch --nodes=$NODES run.slurm --dataset=glove2.2m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=glove2.2m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=glove2.2m --mode base --nprobes $nprobe $EXTRA_OPTIONS
            done
            ;;
        deep1M)
            echo "===== Submitting jobs for dataset: deep1M ====="
            for nprobe in ${DATASET_NPROBES[$dataset]}; do
                sbatch --nodes=$NODES run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=deep1M --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=deep1M --mode base --nprobes $nprobe $EXTRA_OPTIONS
            done
            ;;
        word2vec)
            echo "===== Submitting jobs for dataset: word2vec ====="
            for nprobe in ${DATASET_NPROBES[$dataset]}; do
                sbatch --nodes=$NODES run.slurm --dataset=word2vec --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=word2vec --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=word2vec --mode base --nprobes $nprobe $EXTRA_OPTIONS
            done
            ;;
        StarLightCurves)
            echo "===== Submitting jobs for dataset: StarLightCurves ====="
            for nprobe in ${DATASET_NPROBES[$dataset]}; do
                sbatch --nodes=$NODES run.slurm --dataset=StarLightCurves --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --input_format=txt --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=StarLightCurves --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --input_format=txt --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=StarLightCurves --mode base --input_format=txt --nprobes $nprobe $EXTRA_OPTIONS
            done
            ;;
        HandOutlines)
            echo "===== Submitting jobs for dataset: HandOutlines ====="
            for nprobe in ${DATASET_NPROBES[$dataset]}; do
                sbatch --nodes=$NODES run.slurm --dataset=HandOutlines --warmup_list=10 --input_format=txt --disableOrderOpt --block=10 --mode block --blockSend --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=HandOutlines --warmup_list=10 --input_format=txt --cut --block=5 --mode group --group=2 --team=2 --nprobes $nprobe $EXTRA_OPTIONS
                sbatch --nodes=$NODES run.slurm --dataset=HandOutlines --mode base --input_format=txt --nprobes $nprobe $EXTRA_OPTIONS
            done
            ;;
        *)
            echo "Warning: Unknown dataset '$dataset', skipping..."
            ;;
    esac

    while true; do
        # Check if any jobs are still running
        squeue_output=$(squeue -u lvxg --noheader)
        
        if [[ -z "$squeue_output" ]]; then
            # No output means all jobs have finished
            break
        fi
        
        # echo $squeue_output
        # Otherwise, wait a bit and check again
        # echo "Waiting for jobs to finish..."
        sleep 10  # Check every 10 seconds
    done
done

echo "All jobs submitted successfully!"

# #!/bin/bash



# # 定义所有数据集
# DATASETS="msong sift1m glove1.2m glove2.2m deep1M word2vec StarLightCurves HandOutlines"

# # 定义所有数据集及其对应的nprobes列表
# declare -A DATASET_CONFIG=(
#     ["msong"]="997 500 400 300 200 100 70 50 40 30 20 10 5 3 1"
#     ["sift1m"]="1000 500 400 300 200 100 70 50 40 30 20 10 5 3 1"
#     ["glove1.2m"]="1092 900 700 500 400 300 200 100 50 30 20 10 5 3 1"
#     ["glove2.2m"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
#     ["deep1M"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
#     ["word2vec"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
#     ["StarLightCurves"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
#     ["HandOutlines"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
# )

# # 默认运行所有数据集
# SELECTED_DATASETS="$DATASETS"
# EXTRA_OPTIONS=""

# # 解析命令行参数
# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --datasets=*)
#             SELECTED_DATASETS="${1#*=}"
#             shift
#             ;;
#         --add-option=*)
#             EXTRA_OPTIONS="${EXTRA_OPTIONS} ${1#*=}"
#             shift
#             ;;
#         --help)
#             echo "Usage: $0 [options]"
#             echo "Options:"
#             echo "  --datasets=dataset1,dataset2   Select which datasets to run (default: all)"
#             echo "  --add-option=\"OPTION\"         Add an option to all sbatch commands"
#             echo "  --help                         Show this help message"
#             echo ""
#             echo "Available datasets: $DATASETS"
#             exit 0
#             ;;
#         *)
#             echo "Unknown option: $1"
#             exit 1
#             ;;
#     esac
# done

# # 将逗号分隔的列表转换为空格分隔
# SELECTED_DATASETS=$(echo "$SELECTED_DATASETS" | tr ',' ' ')

# # 运行选定的数据集
# for dataset in $SELECTED_DATASETS; do
#     for nprobe in "${nprobes[@]}"; do
#     case "$dataset" in
#         msong)
#             echo "===== Submitting jobs for dataset: msong ====="
#             sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=msong --mode base $EXTRA_OPTIONS
#             ;;
#         sift1m)
#             echo "===== Submitting jobs for dataset: sift1m ====="
#             sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=sift1m --mode base $EXTRA_OPTIONS
#             ;;
#         glove1.2m)
#             echo "===== Submitting jobs for dataset: glove1.2m ====="
#             sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=glove1.2m --mode base $EXTRA_OPTIONS
#             ;;
#         glove2.2m)
#             echo "===== Submitting jobs for dataset: glove2.2m ====="
#             sbatch --nodes=5 run.slurm --dataset=glove2.2m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=glove2.2m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=glove2.2m --mode base $EXTRA_OPTIONS
#             ;;
#         deep1M)
#             echo "===== Submitting jobs for dataset: deep1M ====="
#             sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=deep1M --mode base $EXTRA_OPTIONS
#             ;;
#         word2vec)
#             echo "===== Submitting jobs for dataset: word2vec ====="
#             sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=word2vec --mode base $EXTRA_OPTIONS
#             ;;
#         StarLightCurves)
#             echo "===== Submitting jobs for dataset: StarLightCurves ====="
#             sbatch --nodes=5 run.slurm --dataset=StarLightCurves --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --input_format=txt $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=StarLightCurves --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --input_format=txt $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=StarLightCurves --mode base --input_format=txt $EXTRA_OPTIONS
#             ;;
#         HandOutlines)
#             echo "===== Submitting jobs for dataset: HandOutlines ====="
#             sbatch --nodes=5 run.slurm --dataset=HandOutlines --warmup_list=10 --input_format=txt --disableOrderOpt --block=10 --mode block --blockSend $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=HandOutlines --warmup_list=10 --input_format=txt --cut --block=5 --mode group --group=2 --team=2 $EXTRA_OPTIONS
#             sbatch --nodes=5 run.slurm --dataset=HandOutlines --mode base --input_format=txt $EXTRA_OPTIONS
#             ;;
#         *)
#             echo "Warning: Unknown dataset '$dataset', skipping..."
#             ;;
#     esac

#     while true; do
#         # Check if any jobs are still running
#         squeue_output=$(squeue -u lvxg --noheader)
        
#         if [[ -z "$squeue_output" ]]; then
#             # No output means all jobs have finished
#             break
#         fi
        
#         # echo $squeue_output
#         # Otherwise, wait a bit and check again
#         # echo "Waiting for jobs to finish..."
#         sleep 10  # Check every 10 seconds
#     done
# done

# echo "All jobs submitted successfully!"

# # 定义所有数据集及其对应的nprobes列表
# declare -A DATASET_CONFIG=(
#     ["msong"]="997 500 400 300 200 100 70 50 40 30 20 10 5 3 1"
#     ["sift1m"]="1000 500 400 300 200 100 70 50 40 30 20 10 5 3 1"
#     ["glove1.2m"]="1092 900 700 500 400 300 200 100 50 30 20 10 5 3 1"
#     ["glove2.2m"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
#     ["deep1M"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
#     ["word2vec"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
#     ["StarLightCurves"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
#     ["HandOutlines"]="700 500 400 300 200 100 50 30 20 10 5 3 1 0"
# )

# # 定义每个数据集的特定参数
# declare -A DATASET_PARAMS=(
#     ["StarLightCurves"]="--input_format=txt"
#     ["HandOutlines"]="--input_format=txt"
# )

# # 默认运行所有数据集
# SELECTED_DATASETS=("${!DATASET_CONFIG[@]}")
# EXTRA_OPTIONS=""
# NODES=5

# # 解析命令行参数
# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --datasets=*)
#             IFS=',' read -ra SELECTED_DATASETS <<< "${1#*=}"
#             shift
#             ;;
#         --add-option=*)
#             EXTRA_OPTIONS="${EXTRA_OPTIONS} ${1#*=}"
#             shift
#             ;;
#         --nodes=*)
#             NODES="${1#*=}"
#             shift
#             ;;
#         --help)
#             echo "Usage: $0 [options]"
#             echo "Options:"
#             echo "  --datasets=dataset1,dataset2   Select which datasets to run (default: all)"
#             echo "  --add-option=\"OPTION\"         Add an option to all sbatch commands"
#             echo "  --nodes=N                     Set number of nodes (default: 5)"
#             echo "  --help                         Show this help message"
#             echo ""
#             echo "Available datasets: ${!DATASET_CONFIG[@]}"
#             exit 0
#             ;;
#         *)
#             echo "Unknown option: $1"
#             exit 1
#             ;;
#     esac
# done

# # 运行选定的数据集
# for dataset in "${SELECTED_DATASETS[@]}"; do
#     if [[ -z "${DATASET_CONFIG[$dataset]}" ]]; then
#         echo "Warning: Unknown dataset '$dataset', skipping..."
#         continue
#     fi
    
#     echo "===== Processing dataset: $dataset ====="
    
#     # 获取数据集特定参数
#     ds_params="${DATASET_PARAMS[$dataset]}"
    
#     # 读取nprobes列表
#     IFS=' ' read -ra nprobes <<< "${DATASET_CONFIG[$dataset]}"
    
#     for nprobe in "${nprobes[@]}"; do
#         # 为当前数据集提交三个不同的作业
        
#         # 第一个作业: block模式
#         cmd="sbatch --nodes=$NODES run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe $ds_params $EXTRA_OPTIONS"
#         eval "$cmd"
        
#         # 第二个作业: group模式
#         cmd="sbatch --nodes=$NODES run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --nprobes $nprobe $ds_params $EXTRA_OPTIONS"
#         eval "$cmd"
        
#         # 第三个作业: base模式
#         cmd="sbatch --nodes=$NODES run.slurm --dataset=$dataset --mode base --nprobes $nprobe $ds_params $EXTRA_OPTIONS"
#         eval "$cmd"
#     done

#     while true; do
#         # Check if any jobs are still running
#         squeue_output=$(squeue -u lvxg --noheader)
        
#         if [[ -z "$squeue_output" ]]; then
#             # No output means all jobs have finished
#             break
#         fi
        
#         # echo $squeue_output
#         # Otherwise, wait a bit and check again
#         # echo "Waiting for jobs to finish..."
#         sleep 10  # Check every 10 seconds
#     done
# done

# echo "All jobs submitted successfully!"

# nprobes=(997 500 400 300 200 100 70 50 40 30 20 10 5 3 1)
# dataset=msong
# for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# done

# nprobes=(1000 500 400 300 200 100 70 50 40 30 20 10 5 3 1)
# dataset=sift1m
# for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# done

# # nprobes=(518 516 500 400 300 200 100 70 50 40 30 20 10 5 3 1)
# # dataset=nuswide
# # for nprobe in "${nprobes[@]}"; do
# #     sbatch --nodes=3 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=2 --mode block --blockSend --nprobes $nprobe

# #     sbatch --nodes=3 run.slurm --dataset=$dataset --warmup_list=0 --cut --block=1 --mode group --group=2 --team=2  --nprobes $nprobe

# #     sbatch --nodes=3 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# # done

# # nprobes=(1087 700 500 400 300 200 100 50 30 20 10 5 3 1)
# # dataset=glove25
# # for nprobe in "${nprobes[@]}"; do
# #     sbatch --nodes=6 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=5 --mode block --blockSend --nprobes $nprobe

# #     sbatch --nodes=6 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=1 --mode group --group=5 --team=5  --nprobes $nprobe

# #     sbatch --nodes=6 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# # done

# nprobes=(1092 900 700 500 400 300 200 100 50 30 20 10 5 3 1)
# dataset=glove1.2m
# for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# done

# nprobes=(700 500 400 300 200 100 50 30 20 10 5 3 1 0)
# dataset=HandOutlines
# for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --disableOrderOpt --block=10 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --cut --block=5 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --input_format=txt --nprobes $nprobe 
# done


# nprobes=(700 500 400 300 200 100 50 30 20 10 5 3 1 0)
# dataset=StarLightCurves
# for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --input_format=txt --nprobes $nprobe 
# done

# # nprobes=(200 100 50 30 20 10 5 3 1 0)

# # dataset=fasion_mnist_784
# # for nprobe in "${nprobes[@]}"; do
# #     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=4 --mode block --blockSend --nprobes $nprobe

# #     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

# #     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --nprobes $nprobe 
# # done

# nprobes=(700 500 400 300 200 100 50 30 20 10 5 3 1 0)

# dataset=word2vec
# for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --nprobes $nprobe 
# done


# # nprobes=(1000 500 400 300 200 100 70 50 40 30 20 10 5 3 1)
# # dataset=sift1m
# # for nprobe in "${nprobes[@]}"; do
# #     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# #     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

# #     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# # done

# nprobes=(700 500 400 300 200 100 50 30 20 10 5 3 1 0)
# dataset=glove2.2m
# for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# done

# nprobes=(700 500 400 300 200 100 50 30 20 10 5 3 1 0)
# dataset=deep1M
# for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=4 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# done

while true; do
    # Check if any jobs are still running
    squeue_output=$(squeue -u lvxg --noheader)
    
    if [[ -z "$squeue_output" ]]; then
        # No output means all jobs have finished
        echo "All jobs have finished!"
        python3 average.py 
        python3 speedRecall.py
        break
    fi
    
    # echo $squeue_output
    # Otherwise, wait a bit and check again
    # echo "Waiting for jobs to finish..."
    sleep 10  # Check every 10 seconds
done

cut=("" "--cut")
# disableOrderOptimize=("" "--disableOrderOpt")

# cut=("" "--cut")
disableOrderOptimize=("")

# DATASET=msong
BASE_COMMAND="mpirun --bind-to none -n $SLURM_NPROCS ./release/bin/query --benchmarks_path ./benchmarks --cache --opt_levels OPT_NONE --warmup_list 10"
# for cut in "${cut[@]}"; do
#   for dop in "${disableOrderOptimize[@]}"; do

#     sbatch --nodes=2 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=8 --mode block $@
#     sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=8 --mode block $@

#     sbatch --nodes=2 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=8 --mode block $@
#     sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=8 --mode block $@

#     sbatch --nodes=2 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=4 --mode block $@
#     sbatch --nodes=5 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=4 --mode block $@

#     sbatch --nodes=2 run.slurm --dataset=fasion_mnist_784 --warmup_list=10 $cut $dop --block=8 --mode block $@
#     sbatch --nodes=5 run.slurm --dataset=fasion_mnist_784 --warmup_list=10 $cut $dop --block=8 --mode block $@
   

#     sbatch --nodes=2 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=4 --mode group --group=1 --team=1 $@
#     sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=4 --mode group --group=2 --team=2 $@

#     sbatch --nodes=2 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=4 --mode group --group=1 --team=1 $@
#     sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=4 --mode group --group=2 --team=2 $@

#     sbatch --nodes=2 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=2 --mode group --group=1 --team=1 $@
#     sbatch --nodes=5 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=2 --mode group --group=2 --team=2 $@

#     sbatch --nodes=2 run.slurm --dataset=fasion_mnist_784 --warmup_list=10 $cut $dop --block=8 --mode group --group=1 --team=1 $@
#     sbatch --nodes=5 run.slurm --dataset=fasion_mnist_784 --warmup_list=10 $cut $dop --block=4 --mode group --group=2 --team=2 $@
#     done
# done

# sbatch --nodes=2 run.slurm --dataset=msong --mode base $@
# sbatch --nodes=5 run.slurm --dataset=msong --mode base $@

# sbatch --nodes=2 run.slurm --dataset=sift1m --mode base $@
# sbatch --nodes=5 run.slurm --dataset=sift1m  --mode base $@

# sbatch --nodes=2 run.slurm --dataset=nuswide  --mode base $@
# sbatch --nodes=5 run.slurm --dataset=nuswide  --mode base $@

# sbatch --nodes=2 run.slurm --dataset=fasion_mnist_784  --mode base $@
# sbatch --nodes=5 run.slurm --dataset=fasion_mnist_784  --mode base $@
