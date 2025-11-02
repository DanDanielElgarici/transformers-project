#!/bin/bash
#SBATCH --job-name=mdm_eval_tubelet_p4o0
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-user=elgarici-dan@campus.technion.ac.il
#SBATCH --mail-type=ALL

# go to project root
cd ~/transformers/project/motion-diffusion-model || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mdm

MODEL_PATH="save/my_humanml_trans_enc_512-2/model000600161.pt"
EVAL_MODE="mm_short"
GUIDANCE_PARAM=2.5
CONTEXT_LEN=60
PRED_LEN=60

echo "Starting MDM evaluation at $(date)"

python3 -m eval.eval_humanml \
  --model_path "$MODEL_PATH" \
  --dataset humanml \
  --eval_mode "$EVAL_MODE" \
  --train_platform_type NoPlatform \
  --context_len "$CONTEXT_LEN" \
  --pred_len "$PRED_LEN" \
  --guidance_param "$GUIDANCE_PARAM" \
  --use_ema

echo "Evaluation finished at $(date)"
