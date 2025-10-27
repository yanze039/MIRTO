import sys
sys.path.insert(0, "/orcd/home/002/yanze039/orcd/pool/RNA_design/test/software/mamba")
sys.path.append("/orcd/home/002/yanze039/orcd/pool/RNA_design/joint_sequence_modeling")
import jsm
from jsm.modules import JointSequenceModeling 
from jsm.data.utils import Alphabet
from pathlib import Path
import os
import hydra
import lightning as L
import torch
from pathlib import Path
from esm.models.esmc import ESMC
from pytorch_lightning.loggers import CSVLogger
import os
import json 
import yaml
import tqdm


@hydra.main(version_base=None, config_path='/orcd/home/002/yanze039/orcd/pool/RNA_design/joint_sequence_modeling/configs',
            config_name='config_12attention')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    # _print_config(config, resolve=True, save_cfg=True)

    protein_encoder = ESMC.from_pretrained("esmc_300m").to("cuda")
    protein_tokenizer = protein_encoder.tokenizer

    ckpt_path = str(config.training.finetune_from)
    
    global_tokenizer = Alphabet.initialize_for_global(offset=0)
    utr_offset = len(global_tokenizer.all_toks)
    utr_alphabet = Alphabet.initialize_for_utr(offset=utr_offset)
    codon_offset = utr_offset + len(utr_alphabet.all_toks)
    codon_alphabet = Alphabet.initialize_for_codon(offset=codon_offset)
    protein_tokenizer = protein_tokenizer
    rna_vocab_size = len(global_tokenizer) + len(utr_alphabet) + len(codon_alphabet) + len(utr_alphabet)
    protein_vocab_size = len(protein_tokenizer)
    
    print(ckpt_path)
    model = JointSequenceModeling.load_from_checkpoint(
        ckpt_path,
        config=config,
        global_tokenizer=global_tokenizer,
        protein_tokenizer=protein_tokenizer,
        rna_vocab_size=rna_vocab_size,
        protein_vocab_size=protein_vocab_size,
        protein_encoder=protein_encoder
    ).to(torch.bfloat16)
    model.register_tokenizer("utr_5", utr_alphabet)
    model.register_tokenizer("utr_3", utr_alphabet)
    model.register_tokenizer("codon", codon_alphabet)
        
    sequence_file = config.sequence_file
    with open(sequence_file, 'r') as f:
        sequences = json.load(f)
    
    Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)
    for idx, (transcript_id, sequence_info) in tqdm.tqdm(enumerate(sequences.items()), total=len(sequences)):
        _output_file = Path(config.output_file).stem
        _output_file = f"{_output_file}_{idx}.pt"
        _output_path = Path(config.output_file).parent / _output_file
        
        if _output_path.exists():
            continue

        embeddings = model.encode(
            protein_sequence=sequence_info[0], 
            utr5_sequence=sequence_info[1].replace("T", "U"), 
            cds_sequence=sequence_info[2], 
            utr3_sequence=sequence_info[3].replace("T", "U"),
            sentence_level=True,
            style='concatenate',
            hidden_layer_idx=config.hidden_layer_idx
        )
        rna_embeddings = {
            transcript_id: embeddings,
        }
        
        torch.save(rna_embeddings, _output_path)

if __name__ == '__main__':
    main()




