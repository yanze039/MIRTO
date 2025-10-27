
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
# export TMPDIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
# export TMPDIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/tmp

export CUDA_VISIBLE_DEVICES=0,1
batch_token_size=65536

HYDRA_FULL_ERROR=1 python -u -m main \
  model=hybrid_mamba_12_layer_transformer \
  backbone=hybrid_mamba \
  data.datamodule.num_workers=2 \
  data.datamodule.batch_size=null \
  data.datamodule.tokens_per_batch=${batch_token_size} \
  trainer.num_nodes=1 \
  trainer.max_epochs=10 \
  trainer.devices=2 \
  hydra.run.dir='/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/${data.name}/${model.name}/production_12_layer_transformer_1006' \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path='${.save_dir}/checkpoints/last.ckpt' \
  optim.lr=0.0002 \
  optim.min_learning_rate=0.00005 \
  optim.num_warmup_steps=10000 \


