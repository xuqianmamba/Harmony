#!/bin/bash
# dataset=("glove1.2m" "glove2.2m" "msong" "sift1m" "word2vec" "HandOutlines" "StarLightCurves" "deep1M")
datasets=(gaussian_128dim_1000000 gaussian_128dim_250000 gaussian_128dim_500000 gaussian_128dim_750000 gaussian_256dim_1000000 gaussian_256dim_250000 gaussian_256dim_500000 gaussian_256dim_750000 gaussian_512dim_1000000 gaussian_512dim_250000 gaussian_512dim_500000 gaussian_512dim_750000 gaussian_64dim_1000000 gaussian_64dim_250000 gaussian_64dim_500000 gaussian_64dim_750000)

# for set in "${datasets[@]}"; do
#     sbatch --nodes=1 run.slurm --dataset=$set --run_faiss 
# done
for set in "${datasets[@]}"; do
    sbatch --nodes=5 run.slurm --dataset=$set --warmup_list=10 --cut --block=4 --mode group --group=2 --team=2  
done

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
