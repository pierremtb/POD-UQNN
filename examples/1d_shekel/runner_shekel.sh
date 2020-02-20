#!/bin/bash
#SBATCH --account=def-soulaima
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-02:00      # time (DD-HH:MM)
#SBATCH --output=log-%j-%N.txt  # %N for node name, %j for jobID

module load cuda cudnn llvm
source ~/env/bin/activate
python gen.py && time horovodrun -np 4 -H localhost:4 python train.py --distribute && python pred.py
