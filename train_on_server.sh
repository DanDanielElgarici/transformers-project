#!/bin/bash
#SBATCH --job-name=tubelet_p=4_o=0
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1                # Request 1 GPUi
#SBATCH --output=out_job_tubelet_p4o0.txt         # Standard output log
#SBATCH --error=err_job_tubelet_p4o0.txt          # Error log
#SBATCH --mail-user=elgarici-dan@campus.technion.ac.il
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mdm

# Run MDM training
python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512-2 --dataset humanml
