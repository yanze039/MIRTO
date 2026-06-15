#!/bin/bash
# 1-step training smoke test for the refactored src/ tree.
#
# Must be run on a GPU node with apptainer.
# Uses:
#   - data=species_specific_dummy   (the tiny LMDB)
#   - model=transformer_small       (12 blocks × 512 hidden ≈ small)
#   - trainer.devices=1, max_steps=1
#   - hydra.run.dir=/tmp/jsm_refactor_smoketest_<ts>
# Disables: callbacks (no checkpoint writes), resume, finetune-from.
#
# Verifies: forward + loss + backward + optimizer.step + scheduler.step
# complete without errors on the refactored code path.
#
# Does NOT write to outputs/. Does NOT touch any canonical .ckpt.
#
set -euo pipefail

if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || true
fi

CONTAINER=/home/yanze039/orcd/scratch/container/container/sequence_modeling_0625.sif
REPO=/orcd/pool/007/yanze039/RNA_design/joint_sequence_modeling
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=/tmp/jsm_refactor_smoketest_${STAMP}
mkdir -p "$OUTDIR"
echo "Smoke test output dir: $OUTDIR"

# Same cache env as the real run.sh so HF/triton/torchinductor don't clutter $HOME.
export HF_HOME=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export XDG_CACHE_HOME=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export TRITON_CACHE_DIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache
export TORCHINDUCTOR_CACHE_DIR=/orcd/home/002/yanze039/orcd/pool/RNA_design/cache

apptainer exec --cleanenv \
    --env HF_HOME=$HF_HOME \
    --env XDG_CACHE_HOME=$XDG_CACHE_HOME \
    --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
    --env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
    --env HYDRA_FULL_ERROR=1 \
    -B /orcd/pool/007/yanze039/RNA_design \
    -B /orcd/home/002/yanze039 \
    -B /home/yanze039/orcd/pool/RNA_design \
    -B /tmp:/tmp \
    -B /home/yanze039/orcd/scratch \
    -B /orcd/scratch/orcd/008/yanze039 \
    --nv \
    "$CONTAINER" \
    bash -c "cd ${REPO}/src && python -u scripts/pretrain_joint_sequence.py \
        data=species_specific_dummy \
        model=transformer_small \
        data.datamodule.batch_size=2 \
        data.datamodule.tokens_per_batch=null \
        trainer.devices=1 \
        trainer.num_nodes=1 \
        +trainer.max_steps=1 \
        trainer.max_epochs=1 \
        trainer.limit_train_batches=1 \
        trainer.limit_val_batches=1 \
        trainer.num_sanity_val_steps=0 \
        trainer.val_check_interval=1.0 \
        optim.num_warmup_steps=1 \
        checkpointing.resume_from_ckpt=false \
        checkpointing.save_dir=${OUTDIR} \
        tensorboard.log_dir=${OUTDIR}/tensorboard \
        hydra.run.dir=${OUTDIR} \
        hydra.job.chdir=true"

echo ""
echo "=== Smoke test PASSED ==="
echo "Outputs (safe to delete): $OUTDIR"
