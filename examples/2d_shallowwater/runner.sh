#!/bin/bash
#SBATCH --account=def-soulaima
#SBATCH --gres=gpu:k80:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --nodes=1        # nodes
#SBATCH --mem=128000M        # memory per node
#SBATCH --time=0-02:00      # time (DD-HH:MM)
#SBATCH --output=log-%j-%N.txt  # %N for node name, %j for jobID

module load cuda cudnn llvm
source ~/env/bin/activate

function run_ex() {
        cd $1
        python pred_ext.py
}

run_ex 2d_shallowwater
