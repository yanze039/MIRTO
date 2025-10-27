
# batch_token_size=$((50 * 1000))
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# (Optional but recommended)
export PATH=/usr/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export XDG_CACHE_HOME=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache

export HF_HOME=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export TRITON_CACHE_DIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export TORCHINDUCTOR_CACHE_DIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export LITDATA_CACHE_DIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MAKEFLAGS="-j8"
export CMAKE_BUILD_PARALLEL_LEVEL=8
export XDG_CACHE_HOME=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache

export CUDA_VISIBLE_DEVICES=0
batch_size=5
batch_token_size=65536
# cd /pool001/yanze039/RNA_design/mdlm

for temperature in 1.0 0.8 0.85 0.9 0.95 1.05 1.1
do
  echo "Running generation with temperature: $temperature"

HYDRA_FULL_ERROR=1 python -u -m 00_generate \
  model=hybrid_mamba_12_layer_transformer \
  backbone=hybrid_mamba \
  trainer.num_nodes=1 \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  "training.finetune_from='/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/joint_sequence/production_model/attention/epoch-1-rerun-361232.ckpt'" \
  hydra.run.dir='/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/joint_sequence/tiny-mamba/production_12_layer_transformer_1006' \
  checkpointing.resume_from_ckpt=false \
  checkpointing.resume_ckpt_path='${.save_dir}/checkpoints/last.ckpt' \
  +temperature=$temperature \
  +protein_sequences=/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/01.embedding_human/embeddings/picked_cluster_members_layer5_none_50_n_clusters_100.yaml \
  +max_generating_length=8000 \
  +output_dir=/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/02.generation/generation_results/human/temperature_$temperature \
  +num_batches=1 \
  +batch_size=200
  
done