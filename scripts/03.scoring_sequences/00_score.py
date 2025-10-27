import sys
sys.path.insert(0, "/orcd/home/002/yanze039/orcd/pool/RNA_design/test/software/mamba")
sys.path.append("/orcd/home/002/yanze039/orcd/pool/RNA_design/joint_sequence_modeling")
from pathlib import Path
import os
import hydra
import lightning as L
import torch
from pathlib import Path
from esm.models.esmc import ESMC
from pytorch_lightning.loggers import CSVLogger
from jsm.modules import JointSequenceModeling 
import jsm.utils as utils
import os
import json
from jsm.data.utils import Alphabet    
import yaml
import tqdm


@hydra.main(version_base=None, config_path='/orcd/home/002/yanze039/orcd/pool/RNA_design/joint_sequence_modeling/configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    # _print_config(config, resolve=True, save_cfg=True)

    protein_encoder = ESMC.from_pretrained("esmc_300m").to("cuda")
    protein_tokenizer = protein_encoder.tokenizer

    result_dir = Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(config.training.finetune_from)
    
    global_tokenizer = Alphabet.initialize_for_global(offset=0)
    utr_offset = len(global_tokenizer.all_toks)
    utr_alphabet = Alphabet.initialize_for_utr(offset=utr_offset)
    codon_offset = utr_offset + len(utr_alphabet.all_toks)
    codon_alphabet = Alphabet.initialize_for_codon(offset=codon_offset)
    protein_tokenizer = protein_tokenizer
    rna_vocab_size = len(global_tokenizer) + len(utr_alphabet) + len(codon_alphabet) + len(utr_alphabet)
    protein_vocab_size = len(protein_tokenizer)
    
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
    sequences = []
    with open(sequence_file, 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())
            
    example_sequence_file = "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/sequence_groups/example_sox2.yaml"
    with open(example_sequence_file, 'r') as f:
        example_sequence = yaml.safe_load(f)["SOX2"]
    
    rna_scores = {}    
    idx = 0
    
    for transcript_id, designed_utr in tqdm.tqdm(enumerate(sequences), total=len(sequences)):
        if config.design_mode == "utr_5":
            utr_3_seq = example_sequence["utr_3"].upper().replace("T", "U")
            utr_5_seq = designed_utr.upper().replace("T", "U")
        elif config.design_mode == "utr_3":
            utr_5_seq = example_sequence["utr_5"].upper().replace("T", "U")
            utr_3_seq = designed_utr.upper().replace("T", "U")
        else:
            raise ValueError("design_mode should be utr_3 or utr_5")

        score = model.score(
            protein_sequence=example_sequence["protein"].upper(),
            utr5_sequence=utr_5_seq,
            cds_sequence=example_sequence["cds"].upper(),
            utr3_sequence=utr_3_seq,
        )
        # print(embeddings.shape, embeddings.dtype)
        rna_scores[transcript_id] = {
            "score": score.item(),
            "utr_sequence": designed_utr.upper()
        }
        idx += 1
    
    file_stem = Path(sequence_file).stem
    output_file = f"/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/pick_sequence_by_ppl/data/scores/{file_stem}_rna_scores.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(rna_scores, f, default_flow_style=False)


if __name__ == '__main__':
    main()




