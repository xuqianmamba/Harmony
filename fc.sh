#!/bin/bash
# ratios=(0.2)
ratios=(0.0  0.2  0.4 0.6  0.8  1.0)
for ratio in "${ratios[@]}"; do
    sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --disableOrderOpt --block=4 --mode block --blockSend --HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=0 --HardInBalanceTeamRatio=$ratio

    sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  --HardInBalance --HardInBalanceTeam=2 --HardInBalanceTeamRatio=$ratio --HardInBalanceRatio=0

    sbatch --nodes=5 run.slurm --dataset=msong --mode base --HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=0 --HardInBalanceTeamRatio=$ratio
done

# ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# dataset=sift1m
# for ratio in "${ratios[@]}"; do
#     hard="--HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=$ratio"
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2 $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base $hard
# done

# ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# dataset=glove1.2m
# for ratio in "${ratios[@]}"; do
#     hard="--HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=$ratio"
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=2 --mode group --group=2 --team=2 $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base $hard
# done

# ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# dataset=glove2.2m
# for ratio in "${ratios[@]}"; do
#     hard="--HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=$ratio"
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=2 --mode group --group=2 --team=2 $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base $hard
# done

# nprobes=(518 516 500 400 300 200 100 70 50 40 30 20 10 5 3 1)
# dataset=nuswide
# for ratio in "${ratios[@]}"; do
#     sbatch --nodes=3 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=2 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=3 run.slurm --dataset=$dataset --warmup_list=0 --cut --block=1 --mode group --group=2 --team=2  --nprobes $nprobe

#     sbatch --nodes=3 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# done

# nprobes=(1087 700 500 400 300 200 100 50 30 20 10 5 3 1)
# dataset=glove25
# for ratio in "${ratios[@]}"; do
#     sbatch --nodes=6 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=5 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=6 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=1 --mode group --group=5 --team=5  --nprobes $nprobe

#     sbatch --nodes=6 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# done

# nprobes=(1092 900 700 500 400 300 200 100 50 30 20 10 5 3 1)
# dataset=glove1.2m
# for ratio in "${ratios[@]}"; do
#     sbatch --nodes=9 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=9 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=2 --mode group --group=4 --team=4  --nprobes $nprobe

#     sbatch --nodes=9 run.slurm --dataset=$dataset --mode base --nprobes $nprobe
# done

# sbatch --nodes=2 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=1 --mode block --blockSend 

# sbatch --nodes=2 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=1 --mode group --group=1 --team=1  

# sbatch --nodes=2 run.slurm --dataset=$dataset --mode base --nprobes 

# dataset=word2vec
# for ratio in "${ratios[@]}"; do
#     hard="--HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=$ratio"
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend  $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2   $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base $hard
# done
# # nprobes=(31 20 10 8 5 4 3 1)
# dataset=deep1M
# for ratio in "${ratios[@]}"; do
#     hard="--HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=$ratio"
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend  $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2   $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base $hard
# done

# dataset=HandOutlines
# for ratio in "${ratios[@]}"; do
#     hard="--HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=$ratio"
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --disableOrderOpt --block=10 --mode block --blockSend  $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --cut --block=5 --mode group --group=2 --team=2   $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --input_format=txt  $hard
# done
# dataset=StarLightCurves
# for ratio in "${ratios[@]}"; do
    # hard="--HardInBalance --HardInBalanceTeam=2 --HardInBalanceRatio=$ratio"
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --disableOrderOpt --block=8 --mode block --blockSend  $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --cut --block=4 --mode group --group=2 --team=2   $hard

#     sbatch --nodes=5 run.slurm --dataset=$dataset --mode base --input_format=txt  $hard
# done

# nprobes=(31 20 10 8 5 4 3 1)

# dataset=StarLightCurves
# for ratio in "${ratios[@]}"; do
#     sbatch --nodes=4 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --disableOrderOpt --block=10 --mode block --blockSend --nprobes $nprobe

#     sbatch --nodes=4 run.slurm --dataset=$dataset --warmup_list=10 --input_format=txt --cut --block=5 --mode group --group=1 --team=1  --nprobes $nprobe

#     sbatch --nodes=4 run.slurm --dataset=$dataset --mode base --input_format=txt --nprobes $nprobe 
# done


# nprobes=(1000 500 400 300 200 100 70 50 40 30 20 10 5 3 1)
# dataset=sift1m
# for ratio in "${ratios[@]}"; do
#     sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend --nprobes $nprobe

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
        python3 speedFC.py
        break
    fi
    
    # echo $squeue_output
    # Otherwise, wait a bit and check again
    # echo "Waiting for jobs to finish..."
    sleep 10  # Check every 10 seconds
done