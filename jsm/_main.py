from esm.models.esmc import ESMC
import os
import sys
import fsspec
import tensorboard
import hydra
import lightning as L
import lightning.pytorch as pl
import omegaconf
import rich.syntax
import rich.tree
import torch
print(sys.path)
# get env
print(os.environ)
import joint_sequence_modeling as jsm
import utils
from pathlib import Path
import typing
import collections
import torch

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
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


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
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
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)

def _train(config, logger):
  logger.info('Starting Training.')
  tensorboard_logger = L.pytorch.loggers.TensorBoardLogger(
    save_dir=config.tensorboard.log_dir,
    name=config.tensorboard.name,
    version=config.tensorboard.version,
    default_hp_metric=False,
  )

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=tensorboard_logger)
  
  protein_encoder = ESMC.from_pretrained("esmc_300m").to("cuda").to(torch.bfloat16)
  protein_tokenizer = protein_encoder.tokenizer
  
  
  data_module = hydra.utils.instantiate(
    config.data.datamodule,
    protein_tokenizer = protein_tokenizer
  )
  
  print("Here outside striped hyena")
  import pickle
  data_batch_file = "/pool001/yanze039/RNA_design/outputs_joint_sequence/joint_sequence/tiny-hyena/20250904/debug/error_batch.pkl"
  with open(data_batch_file, "rb") as f:
      data_batch = pickle.load(f)
  # print(data_batch)
  engine = "esmc_300m"
  with torch.no_grad():
      model = ESMC.from_pretrained(engine).to("cuda")
      embeddings = model(data_batch["protein_input_ids"])
      print(embeddings.embeddings.shape)
  exit(0)
  if config.training.finetune_from is not None:
    logger.info('Finetuning from checkpoint: {}'.format(
      config.training.finetune_from))
    model = jsm.JointSequenceModeling.load_from_checkpoint(
      config.training.finetune_from,
      config=config,
      global_tokenizer=data_module.global_tokenizer,
      protein_tokenizer=protein_tokenizer,
      rna_vocab_size=data_module.rna_vocab_size,
      protein_vocab_size=data_module.protein_vocab_size,
      protein_encoder=protein_encoder
    ).to(torch.bfloat16)
  else:
    logger.info('Training from scratch.')
    
    model = jsm.JointSequenceModeling(
      config=config,
      global_tokenizer=data_module.global_tokenizer,
      protein_tokenizer=protein_tokenizer,
      rna_vocab_size=data_module.rna_vocab_size,
      protein_vocab_size=data_module.protein_vocab_size,
      protein_encoder=protein_encoder
    ).to(torch.bfloat16)
  trainer.fit(model, 
              datamodule=data_module,
              ckpt_path=ckpt_path
  )

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  # _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  _train(config, logger)


if __name__ == '__main__':
  main()