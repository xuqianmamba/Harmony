#!/bin/bash

# 定义所有数据集
DATASETS="msong sift1m glove1.2m glove2.2m deep1M word2vec StarLightCurves HandOutlines"

# 默认运行所有数据集
SELECTED_DATASETS="$DATASETS"
EXTRA_OPTIONS=""

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
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --datasets=dataset1,dataset2   Select which datasets to run (default: all)"
            echo "  --add-option=\"OPTION\"         Add an option to all sbatch commands"
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
            sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=msong --mode base $EXTRA_OPTIONS
            ;;
        sift1m)
            echo "===== Submitting jobs for dataset: sift1m ====="
            sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=sift1m --mode base $EXTRA_OPTIONS
            ;;
        glove1.2m)
            echo "===== Submitting jobs for dataset: glove1.2m ====="
            sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=glove1.2m --mode base $EXTRA_OPTIONS
            ;;
        glove2.2m)
            echo "===== Submitting jobs for dataset: glove2.2m ====="
            sbatch --nodes=5 run.slurm --dataset=glove2.2m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=glove2.2m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=glove2.2m --mode base $EXTRA_OPTIONS
            ;;
        deep1M)
            echo "===== Submitting jobs for dataset: deep1M ====="
            sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=deep1M --mode base $EXTRA_OPTIONS
            ;;
        word2vec)
            echo "===== Submitting jobs for dataset: word2vec ====="
            sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=word2vec --mode base $EXTRA_OPTIONS
            ;;
        StarLightCurves)
            echo "===== Submitting jobs for dataset: StarLightCurves ====="
            sbatch --nodes=5 run.slurm --dataset=StarLightCurves --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --input_format=txt $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=StarLightCurves --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --input_format=txt $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=StarLightCurves --mode base --input_format=txt $EXTRA_OPTIONS
            ;;
        HandOutlines)
            echo "===== Submitting jobs for dataset: HandOutlines ====="
            sbatch --nodes=5 run.slurm --dataset=HandOutlines --warmup_list=10 --input_format=txt --disableOrderOpt --block=10 --mode block --blockSend $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=HandOutlines --warmup_list=10 --input_format=txt --cut --block=5 --mode group --group=2 --team=2 $EXTRA_OPTIONS
            sbatch --nodes=5 run.slurm --dataset=HandOutlines --mode base --input_format=txt $EXTRA_OPTIONS
            ;;
        *)
            echo "Warning: Unknown dataset '$dataset', skipping..."
            ;;
    esac
done

echo "All jobs submitted successfully!"

# dataset=msong
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 
# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base  

# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 
# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 
# sbatch --nodes=5 run.slurm --dataset=sift1m --mode base 

# sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 
# sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  
# sbatch --nodes=5 run.slurm --dataset=glove1.2m --mode base 

# dataset=glove2.2m
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base 

# dataset=deep1M
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base 

# dataset=word2vec
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base 

# dataset=StarLightCurves
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --input_format=txt

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --input_format=txt 

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --input_format=txt 

# dataset=HandOutlines
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --disableOrderOpt --block=10 --mode block --blockSend 

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --cut --block=5 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --input_format=txt  


# sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 

# sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=msong --mode base 

# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 

# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=4 --mode group --group=4 --team=4  

# sbatch --nodes=5 run.slurm --dataset=sift1m --mode base 


# sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 

# sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=deep1M --mode base 

# sbatch --nodes=9 run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend 

# sbatch --nodes=9 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=8 --mode group --group=1 --team=1  

# sbatch --nodes=9 run.slurm --dataset=deep1M --mode base 

# sbatch --nodes=17 run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=20 --mode block --blockSend 

# sbatch --nodes=17 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=5 --mode group --group=8 --team=8  

# sbatch --nodes=17 run.slurm --dataset=deep1M --mode base

while true; do
    # Check if any jobs are still running
    squeue_output=$(squeue -u lvxg --noheader)
    
    if [[ -z "$squeue_output" ]]; then
        # No output means all jobs have finished
        echo "All jobs have finished!"
        # python3 buildTime.py 
        python3 average.py
        break
    fi
    # echo $squeue_output
    # Otherwise, wait a bit and check again
    # echo "Waiting for jobs to finish..."
    sleep 10  # Check every 10 seconds
done


# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=glove2.2m --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=StarLightCurves --input_format=txt --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=HandOutlines --input_format=txt --warmup_list=10 --cut --block=5 --mode block 
# glove1.2m,deep1m,star,hand,word2vec