
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

export CUDA_VISIBLE_DEVICES=0
# cd /pool001/yanze039/RNA_design/mdlm
HYDRA_FULL_ERROR=1 python -u -m 00_embed \
  "training.finetune_from='/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/joint_sequence/production_model/attention/epoch-1-rerun-361232.ckpt'" \
  +sequence_file=/home/yanze039/orcd/scratch/data/data/RefSeq_hsapiens/sequence_pairs/GRCh37_latest_rna_sequence_pairs.json \
  +max_generating_length=8000 \
  +output_file=/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/01.embedding_human/outputs_1012_attention_layer5/rna_embeddings.pt \
  +hidden_layer_idx=5