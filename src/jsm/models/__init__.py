# NOTE (refactor): the original jsm/models/__init__.py added two hardcoded
# user-local paths to sys.path so the legacy mamba / flash-linear-attention /
# vortex backbones could be imported:
#
#   sys.path.insert(0, "/orcd/home/002/yanze039/orcd/pool/RNA_design/test/software/mamba")
#   sys.path.insert(0, "/home/yanze039/orcd/scratch/software/flash-linear-attention")
#
# The refactored MDLM path uses only transformer.py, which does not need
# those packages, so the sys.path patches are deliberately removed. If a
# legacy backbone is ever resurrected, restore the import inside that
# backbone module (not in __init__) so it does not leak to other users.
