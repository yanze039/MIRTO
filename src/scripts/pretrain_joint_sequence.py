"""Diffusion (MDLM + species tag) training entry point.

Refactored from:
  /home/yanze039/orcd/pool/RNA_design/train_diffusion/submit_diffusion/
      train_diffusion_300M/pretrain_joint_sequence.py

Run from this directory:
  python -u pretrain_joint_sequence.py <hydra overrides...>

Or as a module from the src/ root:
  python -u -m scripts.pretrain_joint_sequence <hydra overrides...>

The only behavior change vs the original is:
  * sys.path / hydra config_path are derived from __file__ instead of being
    hardcoded to a user-specific absolute path. The refactored src/ tree is
    therefore relocatable.
"""
import os
import sys
import collections
import typing
from pathlib import Path

# Make `import jsm.*` resolve to the refactored src/jsm package, regardless of
# where this file is invoked from.
_SRC_ROOT = Path(__file__).resolve().parent.parent  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from jsm.diffusion.modules import JointSequenceDiffusion  # noqa: E402
import jsm.utils as utils  # noqa: E402

import fsspec  # noqa: E402
import hydra  # noqa: E402
import lightning as L  # noqa: E402
import lightning.pytorch as pl  # noqa: E402  (imported for side effects)
import omegaconf  # noqa: E402
import rich.syntax  # noqa: E402
import rich.tree  # noqa: E402
import tensorboard  # noqa: E402  (imported for side effects / availability check)
import torch  # noqa: E402
from esm.models.esmc import ESMC  # noqa: E402

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)

torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([typing.Any])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([float])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])


_CONFIG_DIR = str(_SRC_ROOT / 'diffusion_configs')


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True,
    save_cfg: bool = True,
) -> None:
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)
    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(config.checkpointing.save_dir),
            'w',
        ) as fp:
            rich.print(tree, file=fp)


def _train(config, logger):
    logger.info('Starting Training.')
    tensorboard_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=config.tensorboard.log_dir,
        name=config.tensorboard.name,
        version=config.tensorboard.version,
        default_hp_metric=False,
    )

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=tensorboard_logger,
    )

    protein_encoder = ESMC.from_pretrained('esmc_300m').to('cuda')
    protein_tokenizer = protein_encoder.tokenizer

    data_module = hydra.utils.instantiate(
        config.data.datamodule,
        protein_tokenizer=protein_tokenizer,
    )

    if config.training.finetune_from is not None:
        logger.info(
            'Finetuning from checkpoint: {}'.format(
                config.training.finetune_from))
        model = JointSequenceDiffusion.load_from_checkpoint(
            config.training.finetune_from,
            config=config,
            global_tokenizer=data_module.global_tokenizer,
            protein_tokenizer=protein_tokenizer,
            rna_vocab_size=data_module.rna_vocab_size,
            protein_vocab_size=data_module.protein_vocab_size,
            protein_encoder=protein_encoder,
        ).to(torch.bfloat16)
    else:
        logger.info('Training from scratch.')
        model = JointSequenceDiffusion(
            config=config,
            global_tokenizer=data_module.global_tokenizer,
            protein_tokenizer=protein_tokenizer,
            rna_vocab_size=data_module.rna_vocab_size,
            protein_vocab_size=data_module.protein_vocab_size,
            protein_encoder=protein_encoder,
        ).to(torch.bfloat16)

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    logger = utils.get_logger(__name__)
    _train(config, logger)


if __name__ == '__main__':
    main()
