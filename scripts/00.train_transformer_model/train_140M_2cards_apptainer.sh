#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --export=ALL
#SBATCH --gres=gpu:h200:2
#SBATCH --distribution=nopack
#SBATCH --mem=200G
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00

module load apptainer

CONTAINER_PATH=/home/yanze039/orcd/scratch/container/container/sequence_modeling_0625.sif

      #   --cleanenv \
apptainer exec --cleanenv\
        -B /orcd/home/002/yanze039/orcd/pool/RNA_design,/tmp,/home/yanze039/orcd/scratch,/orcd/scratch/orcd/008/yanze039 \
        --nv \
        $CONTAINER_PATH \
  bash -c "cd /orcd/home/002/yanze039/orcd/pool/RNA_design/submit_gpt/train_hybrid_mamba_12_transformer_1006 && bash run.sh" \
  
# cd /orcd/home/002/yanze039/orcd/pool/RNA_design/submit_gpt/train_hybrid_mamba_12_transformer_1006
# if [ ! -f done.flag ]; then
#     echo "Resubmitting job..."
#     sbatch train_140M_2cards_apptainer.sh
# else
#     echo "Training completed."
# fi



