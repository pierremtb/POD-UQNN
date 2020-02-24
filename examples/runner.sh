#!/bin/bash
#SBATCH --account=def-soulaima
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --nodes=1        # nodes
#SBATCH --mem=48000M        # memory per node
#SBATCH --time=0-02:00      # time (DD-HH:MM)
#SBATCH --output=log-%j-%N.txt  # %N for node name, %j for jobID

# module load cuda cudnn llvm
# source ~/env/bin/activate

function run_ex() {
        cd $1
        python3 main.py
        cd ..
}

run_ex 1d_shekel
run_ex 2d_ackley
run_ex 1dt_burger
# run_ex 2d_shallowwater
