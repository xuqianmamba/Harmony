#!/bin/bash
# dataset=msong
# sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 --cut --block=4 --mode block 

# sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=glove2.2m --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=deep1M --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=StarLightCurves --input_format=txt --warmup_list=10 --cut --block=4 --mode block 
# sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 --cut --block=4 --mode block 
sbatch --nodes=5 run.slurm --dataset=HandOutlines --input_format=txt --warmup_list=10 --cut --block=5 --mode block 
# glove1.2m,deep1m,star,hand,word2vec