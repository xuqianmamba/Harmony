#!/bin/bash

cut=("--disableOrderOpt --blockSend" "--disableOrderOpt" "--blockSend" "" "--warmup_list=10 --cut")
disableOrderOptimize=("" "--disableOrderOpt")


# DATASET=msong
# BASE_COMMAND="mpirun --bind-to none -n $SLURM_NPROCS ./release/bin/query --benchmarks_path ./benchmarks --cache --opt_levels OPT_NONE --warmup_list 10"
for cut in "${cut[@]}"; do
    # sbatch --nodes=2 run.slurm --dataset=msong --warmup_list=10 $cut --block=8 --mode block $@
    sbatch --nodes=5 run.slurm --dataset=msong $cut --block=8 --mode block $@

    # sbatch --nodes=2 run.slurm --dataset=sift1m --warmup_list=10 $cut --block=8 --mode block $@
    sbatch --nodes=5 run.slurm --dataset=sift1m $cut --block=8 --mode block $@

    # sbatch --nodes=2 run.slurm --dataset=msong --warmup_list=10 $cut --block=4 --mode group --group=1 --team=1 $@
    sbatch --nodes=5 run.slurm --dataset=msong $cut --block=4 --mode group --group=2 --team=2 $@

    # sbatch --nodes=2 run.slurm --dataset=sift1m --warmup_list=10 $cut --block=4 --mode group --group=1 --team=1 $@
    sbatch --nodes=5 run.slurm --dataset=sift1m $cut --block=8 --mode group --group=2 --team=2 $@


done

sbatch --nodes=5 run.slurm --dataset=msong --block=4 --mode group --group=2 --team=2  --disableOrderOpt --blockSend --HardInBalance --HardInBalanceTeam=1 --HardInBalanceRatio=0.7
sbatch --nodes=5 run.slurm --dataset=sift1m --block=8 --mode group --group=2 --team=2 --disableOrderOpt --blockSend --HardInBalance --HardInBalanceTeam=1 --HardInBalanceRatio=0.7

# sbatch --nodes=2 run.slurm --dataset=msong --mode base $@
sbatch --nodes=5 run.slurm --dataset=msong --mode base $@

# sbatch --nodes=2 run.slurm --dataset=sift1m --mode base $@
sbatch --nodes=5 run.slurm --dataset=sift1m  --mode base $@

while true; do
    # Check if any jobs are still running
    squeue_output=$(squeue -u lvxg --noheader)
    
    if [[ -z "$squeue_output" ]]; then
        # No output means all jobs have finished
        echo "All jobs have finished!"
        python3 average.py 
        # python3 speedRecall.py
        break
    fi
    
    # echo $squeue_output
    # Otherwise, wait a bit and check again
    # echo "Waiting for jobs to finish..."
    sleep 10  # Check every 10 seconds
done