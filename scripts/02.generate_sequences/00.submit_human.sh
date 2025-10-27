#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:h200:1
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
  bash -c "cd /orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/02.generation && bash 00.run.human.sh" \
  