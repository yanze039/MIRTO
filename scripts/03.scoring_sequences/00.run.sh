
# batch_token_size=$((50 * 1000))
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# (Optional but recommended)
export PATH=/usr/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export XDG_CACHE_HOME=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export TMPDIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export HF_HOME=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export TRITON_CACHE_DIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export TORCHINDUCTOR_CACHE_DIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export LITDATA_CACHE_DIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MAKEFLAGS="-j8"
export CMAKE_BUILD_PARALLEL_LEVEL=8
export XDG_CACHE_HOME=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export TMPDIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/tmp

export CUDA_VISIBLE_DEVICES=0
batch_size=5
batch_token_size=65536
# cd /orcd/home/002/yanze039/orcd/pool/RNA_design/mdlm

# List of sequence files
files=(
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_h2az_utr3_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_h2az_utr5_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_h12_utr3_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_h12_utr5_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_HMGB2_utr3_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_HMGB2_utr5_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_RPL15_utr3_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_RPL15_utr5_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_RPL31_utr3_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_RPL31_utr5_sequence.txt"
# "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/benchmark_utr3.txt"
"/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/benchmark_utr5.txt"
)

for seq_file in "${files[@]}"; do
  HYDRA_FULL_ERROR=1 python -u -m 01_score \
    trainer.num_nodes=1 \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    "training.finetune_from='/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/joint_sequence/tiny-mamba/20250718/production/checkpoints/epoch=9-step=1023159.ckpt'" \
    hydra.run.dir='/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/joint_sequence/tiny-mamba/20250718/production' \
    checkpointing.resume_from_ckpt=false \
    checkpointing.resume_ckpt_path='${.save_dir}/checkpoints/last.ckpt' \
    +sequence_file=${seq_file} \
    +design_mode=utr_5
done

# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_h2az_utr3_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_h2az_utr5_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_h12_utr3_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_h12_utr5_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_HMGB2_utr3_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_HMGB2_utr5_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_RPL15_utr3_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_RPL15_utr5_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_RPL31_utr3_sequence.txt
# /pool001/yanze039/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_pool/protein_info_highly_translate_RPL31_utr5_sequence.txt
