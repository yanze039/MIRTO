import sys
sys.path.append("/orcd/home/002/yanze039/orcd/pool/RNA_design/joint_sequence_modeling")
from pathlib import Path
import os
import hydra
import lightning as L
import torch
from pathlib import Path
from esm.models.esmc import ESMC
import os
import json
from jsm.modules import JointSequenceModeling 
import jsm.utils as utils
from jsm.data.utils import Alphabet
import yaml
import pandas as pd
import time

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

    protein_sequence_file = Path(config.protein_sequences)
    if protein_sequence_file.suffix == '.yaml':
        with open(protein_sequence_file, 'r') as f:
            sequences = yaml.safe_load(f)
            protein_sequences = {k: v['protein_sequence'].upper() if 'protein_sequence' in v else v
                     for k, v in sequences.items()}
            expected_utr_5_length = {}
            for k, v in sequences.items():
                if 'expected_utr_5_length' in v:
                    expected_utr_5_length[k] = v["expected_utr_5_length"]
                elif 'utr5_sequence' in v:
                    expected_utr_5_length[k] = len(v["utr5_sequence"])
                else:
                    expected_utr_5_length[k] = None
            expected_utr_3_length = {}
            for k, v in sequences.items():
                if 'expected_utr_3_length' in v:
                    expected_utr_3_length[k] = v["expected_utr_3_length"]
                elif 'utr3_sequence' in v:
                    expected_utr_3_length[k] = len(v["utr3_sequence"])
                else:
                    expected_utr_3_length[k] = None
            prompts = {k: v['prompt'].upper().replace("T", "U") if 'prompt' in v else None
                     for k, v in sequences.items()}
    elif protein_sequence_file.suffix == '.csv': 
        df = pd.read_csv(protein_sequence_file)
        protein_sequences = {row['gene_name']: row['protein_sequence'] for _, row in df.iterrows()}
        expected_utr_5_length = {row['gene_name']: None for _, row in df.iterrows()}
        expected_utr_3_length = {row['gene_name']: None for _, row in df.iterrows()}
    else:
        raise ValueError(f"Unsupported file format: {protein_sequence_file.suffix}")
    
    output_dir = Path(Path(config.output_dir).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for protein_name, protein_sequence in protein_sequences.items():
        output_file = output_dir / f"{protein_name}_rna_sequences.yaml"
        if output_file.exists():
            print(f"Skipping {protein_name}, output file already exists.")
            continue
        generated_results = []
        print(f"Generating RNA sequences for protein: {protein_name}")
        
        start = time.time()
        
        
        for i in range(config.num_batches):
            print(f"Generating batch {i+1}/{config.num_batches} for protein: {protein_name} with prompt: {prompts[protein_name]}")
            if len(protein_sequence) > 1000:
                print(f"protein sequence is too long {len(protein_sequence)}, skipping generation for {protein_name}.")
                continue
            result = model.generate(protein_sequence,
                    global_tokenizer=global_tokenizer,
                    codon_tokenizer=codon_alphabet,
                    utr_tokenizer=utr_alphabet,
                    max_generating_length=config.max_generating_length,
                    temperature=config.temperature,
                    progress_bar=False,
                    cg=True,
                    batch_size=config.batch_size,
                    cuda_monitor=False,
                    expected_utr_5_length=expected_utr_5_length[protein_name],
                    expected_utr_3_length=expected_utr_3_length[protein_name],
                    prompt=prompts[protein_name] if prompts[protein_name] is not None else None,
            )
            generated_results += result
            # except Exception as e:
            #     print(f"Error generating batch {i+1} for protein {protein_name}: {e}")
            #     continue
        end = time.time()
        if len(generated_results) == 0:
            print(f"No sequences generated for {protein_name}, skipping saving.")
            continue
        print(f"Time taken for {config.num_batches} batches: {end - start:.2f} seconds")

        with open(output_file, "w") as fp:
            yaml.dump(generated_results, fp, default_flow_style=False)

if __name__ == '__main__':
    main()




