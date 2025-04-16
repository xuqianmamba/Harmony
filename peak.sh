#!/bin/bash
# ./memory.sh 
# dataset=sift1m
# datasets=("glove1.2m" "glove2.2m" "msong" "word2vec" "deep1M")

# for dataset in "${datasets[@]}"; do
# ./memory.sh sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --disableOrderOpt --block=8 --mode block --blockSend &

# ./memory.sh sbatch --nodes=5 run.slurm --dataset=$dataset --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  &

# ./memory.sh sbatch --nodes=5 run.slurm --dataset=$dataset --mode base &
# done

# for dataset in "${datasets[@]}"; do
# dataset=StarLightCurves
# dataset=HandOutlines
# ./memory.sh sbatch --nodes=5 run.slurm --dataset=$dataset --input_format=txt --warmup_list=10 --disableOrderOpt --block=10 --mode block --blockSend &

# ./memory.sh sbatch --nodes=5 run.slurm --dataset=$dataset --input_format=txt --warmup_list=10 --cut --block=5 --mode group --group=2 --team=2  &

# ./memory.sh sbatch --nodes=5 run.slurm --dataset=$dataset --input_format=txt --mode base &
# done


