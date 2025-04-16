#!/bin/bash
# dataset=("fasion_mnist_784" "nuswide" "msong" "sift1m" "glove25" "HandOutlines" "StarLightCurves" "deep100m" "spacev100m")
# dataset=("glove1.2m" "glove2.2m" "msong" "sift1m" "word2vec" "HandOutlines" "StarLightCurves" "deep1M")
# for set in "${dataset[@]}"; do
#     sbatch --nodes=1 run.slurm --dataset=$set --run_faiss 
# done

# sbatch --nodes=1 run.slurm --dataset=HandOutlines --run_faiss --verbose --input_format=txt
# sbatch --nodes=1 run.slurm --dataset=StarLightCurves --run_faiss --verbose --input_format=txt
# sbatch --nodes=1 run.slurm --dataset=msong --run_faiss --verbose --nprobes 997 500 400 300 200 100 70 50 40 30 20 10 5 3 1
# sbatch --nodes=1 run.slurm --dataset=sift1m --run_faiss --verbose --nprobes 1000 500 400 300 200 100 70 50 40 30 20 10 5 3 1
# sbatch --nodes=1 run.slurm --dataset=nuswide --run_faiss --verbose --nprobes 518 516 500 400 300 200 100 70 50 40 30 20 10 5 3 1
# sbatch --nodes=1 run.slurm --dataset=glove25 --run_faiss --verbose --nprobes 1087 700 500 400 300 200 100 50 30 20 10 5 3 1
# sbatch --nodes=1 run.slurm --dataset=glove1.2m --run_faiss --verbose --nprobes 1092 900 700 500 400 300 200 100 50 30 20 10 5 3 1
# sbatch --nodes=1 run.slurm --dataset=HandOutlines --run_faiss --verbose --input_format=txt --nprobes 700 500 400 300 200 100 50 30 20 10 5 3 1 0
# sbatch --nodes=1 run.slurm --dataset=StarLightCurves --run_faiss --verbose --input_format=txt --nprobes 700 500 400 300 200 100 50 30 20 10 5 3 1 0
# sbatch --nodes=1 run.slurm --dataset=StarLightCurves --run_faiss --verbose --input_format=txt --nprobes 31 20 10 8 5 4 3 1
# sbatch --nodes=1 run.slurm --dataset=sift10m --run_faiss --verbose --nprobes 1000 500 400 300 200 100 70 50 40 30 20 10 5 3 1
# sbatch --nodes=1 run.slurm --dataset=glove2.2m --run_faiss --verbose --nprobes 700 500 400 300 200 100 50 30 20 10 5 3 1 0
# sbatch --nodes=1 run.slurm --dataset=deep1M --run_faiss --nprobes 700 500 400 300 200 100 50 30 20 10 5 3 1 0
# sbatch --nodes=1 run.slurm --dataset=fasion_mnist_784 --run_faiss --verbose --nprobes 200 100 50 30 20 10 5 3 1 0
# sbatch --nodes=1 run.slurm --dataset=word2vec --run_faiss --verbose --nprobes 700 500 400 300 200 100 50 30 20 10 5 3 1 0

sbatch --nodes=1 run.slurm --dataset=HandOutlines --run_faiss --input_format=txt 
# sbatch --nodes=1 run.slurm --dataset=HandOutlines --run_faiss --input_format=txt --nprobes 700 500 400 300 200 100 50 30 20 10 5 3 1 0
# sbatch --nodes=1 run.slurm --dataset=StarLightCurves --run_faiss --input_format=txt 