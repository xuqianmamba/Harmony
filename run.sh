#!/bin/bash
# scancel -u lvxg
# sbatch --nodes=2 run.slurm --dataset=msong --block=1 --mode block
# sbatch --nodes=2 run.slurm --dataset=msong --mode base
# sbatch --nodes=2 run.slurm --dataset=msong --group=1 --team=1 --block=1 --mode group

# sbatch --nodes=2 run.slurm --dataset=msong --block=8 --mode block
# sbatch --nodes=2 run.slurm --dataset=msong --mode base
# sbatch --nodes=2 run.slurm --dataset=msong --group=1 --team=1 --block=8 --mode group

# sbatch --nodes=5 run.slurm --dataset=sift1m --block=8 --mode block
# sbatch --nodes=5 run.slurm --dataset=sift1m --mode base
# sbatch --nodes=5 run.slurm --dataset=sift1m --group=1 --team=1 --block=8 --mode group
# sbatch --nodes=5 run.slurm --dataset=sift1m --group=2 --team=2 --block=4 --mode group

# sbatch --nodes=2 run.slurm --dataset=word2vec  --divideIVF
# sbatch --nodes=2 run.slurm --dataset=word2vec  --divideIVF
# sbatch --nodes=2 run.slurm --dataset=word2vec  --divideIVF
# sbatch --nodes=2 run.slurm --dataset=word2vec  --divideIVF
# sbatch --nodes=2 run.slurm --dataset=word2vec  --divideIVF

cut=("" "--cut")
# disableOrderOptimize=("" "--disableOrderOpt")

# cut=("" "--cut")
disableOrderOptimize=("")

# DATASET=msong
# BASE_COMMAND="mpirun --bind-to none -n $SLURM_NPROCS ./release/bin/query --benchmarks_path ./benchmarks --cache --opt_levels OPT_NONE --warmup_list 10"
# for cut in "${cut[@]}"; do
#   for dop in "${disableOrderOptimize[@]}"; do
    # $BASE_COMMAND --dataset msong  $cut $dop # 选项-np指定 $SLURM_NPROCS 参数为总核心数，即nodes*tasks

    # sbatch --nodes=2 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=8 --mode block $@
    # sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=8 --mode block $@

    # sbatch --nodes=2 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=8 --mode block $@
    # sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=8 --mode block $@

    # sbatch --nodes=2 run.slurm --dataset=word2vec --warmup_list=10 $cut $dop --block=4 --mode block $@
    # sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 $cut $dop --block=4 --mode block $@

    # sbatch --nodes=2 run.slurm --dataset=glove1.2m --warmup_list=10 $cut $dop --block=8 --mode block $@
    # sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 $cut $dop --block=8 --mode block $@

    # sbatch --nodes=2 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=4 --mode block $@
    # sbatch --nodes=5 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=4 --mode block $@

    # sbatch --nodes=2 run.slurm --dataset=sift10k --warmup_list=10 $cut $dop --block=4 --mode block $@
    # sbatch --nodes=5 run.slurm --dataset=sift10k --warmup_list=10 $cut $dop --block=4 --mode block $@

    # sbatch --nodes=2 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=4 --mode group --group=1 --team=1 $@
    # sbatch --nodes=5 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=4 --mode group --group=2 --team=2 $@

    # sbatch --nodes=2 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=4 --mode group --group=1 --team=1 $@
    # sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=4 --mode group --group=2 --team=2 $@

    # sbatch --nodes=2 run.slurm --dataset=word2vec --warmup_list=10 $cut $dop --block=2 --mode group --group=1 --team=1 $@
    # sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 $cut $dop --block=2 --mode group --group=2 --team=2 $@

    # sbatch --nodes=2 run.slurm --dataset=glove1.2m --warmup_list=10 $cut $dop --block=4 --mode group --group=1 --team=1 $@
    # sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 $cut $dop --block=4 --mode group --group=2 --team=2 $@

    # sbatch --nodes=2 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=2 --mode group --group=1 --team=1 $@
    # sbatch --nodes=5 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=2 --mode group --group=2 --team=2 $@

    # sbatch --nodes=2 run.slurm --dataset=sift10k --warmup_list=10 $cut $dop --block=2 --mode group --group=1 --team=1 $@
    # sbatch --nodes=5 run.slurm --dataset=sift10k --warmup_list=10 $cut $dop --block=2 --mode group --group=2 --team=2 $@


    # sbatch --nodes=2 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=5 --nprobes 10  $@
    # sbatch --nodes=4 run.slurm --dataset=msong --warmup_list=10 $cut $dop --block=5 --nprobes 10 $@

    # sbatch --nodes=2 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=10 --nprobes 10 $@
    # sbatch --nodes=5 run.slurm --dataset=sift1m --warmup_list=10 $cut $dop --block=10 --nprobes 10 $@

    # sbatch --nodes=2 run.slurm --dataset=word2vec --warmup_list=10 $cut $dop --block=10 --nprobes 10 $@
    # sbatch --nodes=5 run.slurm --dataset=word2vec --warmup_list=10 $cut $dop --block=10 --nprobes 10 $@

    # sbatch --nodes=2 run.slurm --dataset=glove1.2m --warmup_list=10 $cut $dop --block=10 --nprobes 10 $@
    # sbatch --nodes=5 run.slurm --dataset=glove1.2m --warmup_list=10 $cut $dop --block=10 --nprobes 10 $@

    # sbatch --nodes=2 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=4 --nprobes 5 $@
    # sbatch --nodes=5 run.slurm --dataset=nuswide --warmup_list=10 $cut $dop --block=4 --nprobes 5 $@

    # sbatch --nodes=2 run.slurm --dataset=sift10k --warmup_list=10 $cut $dop --block=4 --nprobes 1 $@
    # sbatch --nodes=5 run.slurm --dataset=sift10k --warmup_list=10 $cut $dop --block=4 --nprobes 1 $@
    
#   done
# done

sbatch --nodes=2 run.slurm --dataset=msong --mode base $@
# sbatch --nodes=5 run.slurm --dataset=msong --mode base $@

# sbatch --nodes=2 run.slurm --dataset=sift1m --mode base $@
# sbatch --nodes=5 run.slurm --dataset=sift1m  --mode base $@

# sbatch --nodes=2 run.slurm --dataset=word2vec  --mode base $@
# sbatch --nodes=5 run.slurm --dataset=word2vec --mode base $@

# sbatch --nodes=2 run.slurm --dataset=glove1.2m  --mode base $@
# sbatch --nodes=5 run.slurm --dataset=glove1.2m  --mode base $@

# sbatch --nodes=2 run.slurm --dataset=nuswide  --mode base $@
# sbatch --nodes=5 run.slurm --dataset=nuswide  --mode base $@

# sbatch --nodes=2 run.slurm --dataset=sift10k  --mode base $@
# sbatch --nodes=5 run.slurm --dataset=sift10k  --mode base $@


# sbatch --nodes=2 run.slurm --dataset=msong --divideIVF --nprobes 10 $@
# sbatch --nodes=5 run.slurm --dataset=msong --divideIVF --nprobes 10 $@

# sbatch --nodes=2 run.slurm --dataset=sift1m --divideIVF --nprobes 10 $@
# sbatch --nodes=5 run.slurm --dataset=sift1m  --divideIVF --nprobes 10 $@

# sbatch --nodes=2 run.slurm --dataset=word2vec  --divideIVF --nprobes 10 $@
# sbatch --nodes=5 run.slurm --dataset=word2vec --divideIVF --nprobes 10 $@

# sbatch --nodes=2 run.slurm --dataset=glove1.2m  --divideIVF --nprobes 10 $@
# sbatch --nodes=5 run.slurm --dataset=glove1.2m  --divideIVF --nprobes 10 $@

# sbatch --nodes=2 run.slurm --dataset=nuswide  --divideIVF --nprobes 5 $@
# sbatch --nodes=5 run.slurm --dataset=nuswide  --divideIVF --nprobes 5 $@

# sbatch --nodes=2 run.slurm --dataset=sift10k  --divideIVF --nprobes 1 $@
# sbatch --nodes=5 run.slurm --dataset=sift10k  --divideIVF --nprobes 1 $@