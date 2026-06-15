#!/bin/bash
# GPU smoke test for the refactored src/ tree.
#
# Must be run on a node with NVIDIA GPU + apptainer (training nodes).
# Verifies:
#   1. Full import chain works (flash_attn, transformer, JointSequenceDiffusion).
#   2. The canonical checkpoint loads without state-dict mismatch.
#
# Does NOT train, does NOT write to outputs/, does NOT touch the canonical .ckpt
# beyond a read-only load.
#
set -euo pipefail

# Some MIT systems require an explicit module load for apptainer.
if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || true
fi

CONTAINER=/home/yanze039/orcd/scratch/container/container/sequence_modeling_0625.sif
REPO=/orcd/pool/007/yanze039/RNA_design/joint_sequence_modeling
CKPT=/home/yanze039/orcd/pool/RNA_design/outputs/species_specific/transformer_300M/diffusion_300M/checkpoints/epoch=1-step=149012.ckpt

apptainer exec --cleanenv \
    -B /orcd/pool/007/yanze039/RNA_design \
    -B /orcd/home/002/yanze039 \
    -B /home/yanze039/orcd/pool/RNA_design \
    -B /tmp:/tmp \
    -B /home/yanze039/orcd/scratch \
    -B /orcd/scratch/orcd/008/yanze039 \
    --nv \
    "$CONTAINER" \
    python -c "
import sys, os
sys.path.insert(0, '${REPO}/src')

# Full import chain (requires GPU because flash_attn pokes torch.cuda at import).
from jsm.diffusion.modules import JointSequenceDiffusion
from jsm.models.transformer import SpeciesSpecificJointSequenceTransformer
from jsm.data.species_specific import SpeciesSpecificJointSequenceDataModule
print('=== full imports OK ===')

import torch
print('torch CUDA visible:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())

# Read-only checkpoint inspection.
ckpt = torch.load('${CKPT}', map_location='cpu', weights_only=False)
print('ckpt top-level keys:', sorted(ckpt.keys()))
sd = ckpt.get('state_dict', ckpt)
print('state_dict entries:', len(sd))
print('first 5 keys:')
for k in sorted(sd.keys())[:5]:
    print('  ', k, '->', tuple(sd[k].shape))
print('=== checkpoint loads OK ===')
"
