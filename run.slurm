#!/bin/bash
#SBATCH --ntasks-per-node=1        # 每个机器上的核心数量
#SBATCH --partition=g1_xdu          # 队列分区且必须指定正确分区（会被告知）
#SBATCH --cpus-per-task=56
#SBATCH --job-name=vdb            # 作业名称（推荐修改一下，以便区分）
#SBATCH --output=./log/log.%j.out       # 正常日志输出 (%j 参数值为 jobId)
#SBATCH --error=./log/log.%j.err        # 错误日志输出 (%j 参数值为 jobId)

##############################################
# memeory total calculate #
##############################################
mkdir ./memory_total                # ben
mkdir ./tmp                # ben
nameflag=`scontrol show hostname $SLURM_JOB_NODELIST`
# echo $nameflag
jobflag=`squeue | grep $SLURM_JOB_ID | wc -l`
# echo $jobflag
job_command=$(scontrol show job $SLURM_JOB_ID | grep 'Command' | sed 's/.*Command=//')
function gather_job(){
  while [[ "$jobflag" -eq 1 ]]
  do
    # Get the list of nodes allocated for the job using scontrol
    nodes=$nameflag
    # Iterate over each node and check memory usage
    for node in $nodes; do
        # Fetch the memory usage for each worker node using ssh and ps
        # Adjust the ps command as necessary depending on how the job runs
        # echo $node
        # echo $node >> ./tmp/$SLURM_JOB_ID.memory_usage.txt
        # ssh $node "ps -eo pid,rss,comm | grep run.slurm | awk '{print \$2}'" > ./tmp/$SLURM_JOB_ID.memory_usage.txt &
        ssh -o StrictHostKeyChecking=no lvxg@$node "ps -eo rss,comm --sort=-rss" | grep query | sed "s/^/$node /" >> ./tmp/$SLURM_JOB_ID.memory_usage.txt

        # Optionally wait for the background processes to complete
    done

    # Sleep for a short period to allow for real-time monitoring
    sleep 0.5

    # Check if the job is still running
    # job_status=$(squeue -j $job_id -h -o %T)
    # if [[ "$job_status" != "RUNNING" ]]; then
    #     echo "Job $job_id has finished or is no longer running."
    #     break
    # fi
  done

  # # Once the job is finished, output the lines with the biggest memory usage
  # echo "Job $SLURM_JOB_ID has finished. Sorting memory usage..." >> ./tmp/$SLURM_JOB_ID.memory_usage.txt

  # # Sort the memory usage data and display the top 10 entries
  # sort -nrk 2 ./tmp/$SLURM_JOB_ID.memory_usage.txt | head -n 1  | sed "s/^/$job_command /" >> ./memory_usage.txt
}
if [ -n "$PEAK" ]; then
  # echo "MY_VAR is defined and non-empty"
  gather_job &
  # echo "MY_VAR is either undefined or empty"
fi


# gcc --version

##############################################
#          Software Envrironment             #
##############################################
# unset I_MPI_PMI_LIBRARY                        # 取消默认mpi库，使用intel自带
# export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0   # intel 多节点作业所需修改参数 
# module load intel/2022 intelmpi/2022           # intel 环境加载
unset I_MPI_PMI_LIBRARY
unset I_MPI_ROOT
module load openmpi/4.1.1           
export SLURM_MPI_TYPE=openmpi
# # export I_MPI_FABRICS=shm:ofi
# # export FI_PROVIDER=tcp/
# # export SLURM_MPI_TYPE=pmi2
# # export I_MPI_PMI_LIBRARY=none
# export OMPI_MCA_debug=all
# export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
##############################################
#               Run job                      #
##############################################
# echo $SLURM_NPROCS
# echo $OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
# mpirun --bind-to none -n $SLURM_NPROCS ./release/bin/query --benchmarks_path ./benchmarks --dataset $1 $3 $4 --cache --opt_levels OPT_NONE  $2  --warmup_list 10  # 选项-np指定 $SLURM_NPROCS 参数为总核心数，即nodes*tasks
mpirun --bind-to none -n $SLURM_NPROCS ./release/bin/query --benchmarks_path ./benchmarks $@  --cache --verbose --opt_levels OPT_NONE # 选项-np指定 $SLURM_NPROCS 参数为总核心数，即nodes*tasks
# mpirun --bind-to none -n $SLURM_NPROCS ./release/bin/query --benchmarks_path ./benchmarks $@  --verbose --opt_levels OPT_NONE # 选项-np指定 $SLURM_NPROCS 参数为总核心数，即nodes*tasks


# cut=("" "--cut")
# disableOrderOptimize=("" "--disableOrderOpt")

# BASE_COMMAND="mpirun --bind-to none -n $SLURM_NPROCS ./release/bin/query --benchmarks_path ./benchmarks --cache --opt_levels OPT_NONE --warmup_list 10"
# # Traverse all combinations
# for cut in "${cut[@]}"; do
#   for dop in "${disableOrderOptimize[@]}"; do
#     $BASE_COMMAND --dataset msong  $cut $dop # 选项-np指定 $SLURM_NPROCS 参数为总核心数，即nodes*tasks
#   done
# done
# mpirun --bind-to none -n $SLURM_NPROCS ./release/bin/query --benchmarks_path ./benchmarks --dataset --cache --opt_levels OPT_NONE  $2  --warmup_list 0  # 选项-np指定 $SLURM_NPROCS 参数为总核心数，即nodes*tasks
# tail -f log.$SLURM_JOB_ID.err
# mpirun -n $SLURM_NPROCS  ./release/bin/query 
# mpirun -n $SLURM_NPROCS ./a.out
# cd .. && ./parallel
