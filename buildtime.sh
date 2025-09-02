#!/bin/bash
# dataset=msong
# sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2
# sbatch --nodes=5 run.slurm --dataset=msong --mode base 
# sbatch --nodes=1 run.slurm --dataset=msong --run_faiss

dataset=msong
# # # for nprobe in "${nprobes[@]}"; do
#     sbatch --nodes=2 run.slurm --dataset=$dataset --warmup_list=10  --block=5 --mode block 

    sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2
    sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --minorCut
    # sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2

    # sbatch --nodes=2 run.slurm --dataset=$dataset --mode base  --warmup_list=10
    # sbatch --nodes=2 run.slurm --dataset=$dataset --mode base  --warmup_list=10 --cut

    # sbatch --nodes=5 run.slurm --dataset=$dataset --mode base  --warmup_list=10
    # sbatch --nodes=5 run.slurm --dataset=$dataset --mode base  --warmup_list=10 --cut
# # done

# sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=glove1.2m --mode base 

# dataset=glove2.2m
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base 

# dataset=deep1M
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base 

# dataset=word2vec
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base 

# dataset=StarLightCurves
# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --input_format=txt

# sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 --input_format=txt 

# sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --input_format=txt 

# dataset=HandOutlines
    # sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --disableOrderOpt --block=10 --mode block --blockSend --nprobes $nprobe

    # sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --cut --block=5 --mode group --group=2 --team=2  --nprobes $nprobe

    # sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --input_format=txt --nprobes $nprobe 

# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=40 --mode group --group=2 --team=2 

# sbatch --nodes=5 run.slurm --dataset=sift1m --mode base 

# sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=msong --mode base 

# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=4 --mode group --group=4 --team=4  

# sbatch --nodes=5 run.slurm --dataset=sift1m --mode base 

# sbatch --nodes=17 run.slurm --dataset=sift1m --warmup_list=10 --disableOrderOpt --block=16 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=17 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=2 --mode group --group=8 --team=8  

# sbatch --nodes=17 run.slurm --dataset=sift1m --mode base

# sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  

# sbatch --nodes=5 run.slurm --dataset=deep1M --mode base 

# sbatch --nodes=9 run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

# sbatch --nodes=9 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=8 --mode group --group=1 --team=1  

# sbatch --nodes=9 run.slurm --dataset=deep1M --mode base 

# sbatch --nodes=17 run.slurm --dataset=deep1M --warmup_list=10 --disableOrderOpt --block=20 --mode block --blockSend --nprobes $nprobe

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