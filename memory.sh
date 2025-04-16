#!/bin/bash

# datasets=("uk-2007-05@100000")  # List of datasets
# #"uk_1000000" "in-2004" "eu-2005" "indochina" "arabic-2005" "sk-2005" "it-2004" "uk-2007-05" "gsh-2015-host"

# # Function to stop the monitor script
# stop_monitor() {
#     echo "Stopping monitor script"
#     kill $monitor_pid $pid
#     exit
# }

# # Trap the SIGINT signal (Ctrl+C) to stop the monitor script
# trap stop_monitor SIGINT

# # Iterate over datasets
# for dataset in "${datasets[@]}"; do

#     # Start the compress.py script in the background and get the process ID (PID)
#     python3 Re-Pair.py "$dataset" &
#     pid=$!

#     # Monitor the VmRSS every second and output it to a text file
#     while true; do
#         # Check if the compress.py script is still running
#         if ! ps -p $pid > /dev/null; then
#             echo "compress.py has exited"
#             break
#         fi

#         timestamp=$(date "+%Y-%m-%d %H:%M:%S")
#         vmrss=$(ps -o pid=,rss= -p $pid | awk '{print $2}')
#         echo "$timestamp Dataset: $dataset, Part_num: $part_num, VmRSS: $vmrss kB" >> ../../benchmarks/$dataset/peak_memory/Re-Pair_mem.txt
#         sleep 0.0005

#         echo "Finished compress.py for dataset: $dataset, part_num: $part_num"
#         sleep 5
#     done
# done


# Define a function to stop the monitor script on SIGINT
stop_monitor() {
    echo "Stopping memory monitoring..."
    exit
}

# Trap the SIGINT signal (Ctrl+C) to stop the monitor script
trap stop_monitor SIGINT

# The job submission command
job_command=$@

PEAK="true"
export PEAK

# Submit the job and get the job ID
job_id=$(eval $job_command | awk '{print $4}')
echo "Job submitted with Job ID: $job_id"

# Wait for the job to start (optional: can add logic to check job state)
sleep 5


# Monitor the VmRSS of the job across all worker nodes
while true; do
    # Check if the job is still running
    job_status=$(squeue -j $job_id -h -o %T)
    if [[ "$job_status" != "RUNNING" ]]; then
        # Sort the memory usage data and display the top 10 entries
        sort -nrk 2 ./tmp/$job_id.memory_usage.txt | head -n 1  | sed "s/^/$job_command /" >> ./memory_usage.txt
        # echo "Job $job_id has finished or is no longer running."
        break
    fi
done

echo "Finished monitoring memory for job $job_id."
