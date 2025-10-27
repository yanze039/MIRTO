import sys
sys.path.insert(0, "/orcd/home/002/yanze039/orcd/pool/RNA_design/test/software/mamba")
sys.path.append("/orcd/home/002/yanze039/orcd/pool/RNA_design/joint_sequence_modeling")
import jsm
from jsm.modules import JointSequenceModeling 
from jsm.data.utils import Alphabet
from pathlib import Path
import os
from hydra import initialize_config_dir, compose
import lightning as L
import torch
from pathlib import Path
from esm.models.esmc import ESMC
import os
import json  
import yaml
import tqdm
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, default="config_12attention", help='Path to the config file')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory to save the embeddings')
    parser.add_argument('--hidden_layer_idx', type=int, default=None, help='Index of the hidden layer to extract embeddings from')
    parser.add_argument('--sequence_file', type=str, default=None, help='Path to the input sequence file in JSON format')
    parser.add_argument('--average', action='store_true', help='Whether to average attention weights across all layers and heads')
    args = parser.parse_args()
    
    """Main entry point for training."""
    jsm_path = Path(jsm.__path__[0]).parent
    # Initialize Hydra with your config directory
    with initialize_config_dir(config_dir=str(jsm_path / "configs")):
        config = compose(config_name=args.config_name)

    L.seed_everything(config.seed)

    protein_encoder = ESMC.from_pretrained("esmc_300m").to("cuda")
    protein_tokenizer = protein_encoder.tokenizer
    
    global_tokenizer = Alphabet.initialize_for_global(offset=0)
    utr_offset = len(global_tokenizer.all_toks)
    utr_alphabet = Alphabet.initialize_for_utr(offset=utr_offset)
    codon_offset = utr_offset + len(utr_alphabet.all_toks)
    codon_alphabet = Alphabet.initialize_for_codon(offset=codon_offset)
    protein_tokenizer = protein_tokenizer
    rna_vocab_size = len(global_tokenizer) + len(utr_alphabet) + len(codon_alphabet) + len(utr_alphabet)
    protein_vocab_size = len(protein_tokenizer)
    
    model = JointSequenceModeling.load_from_checkpoint(
      args.model_path,
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

    sequence_file = args.sequence_file
    with open(sequence_file, 'r') as f:
        sequences = yaml.safe_load(f)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for idx, (transcript_id, sequence_info) in tqdm.tqdm(enumerate(sequences.items()), total=len(sequences)):
        _output_path = Path(args.output_dir) / f"{transcript_id}.pt"
        
        if _output_path.exists():
            continue
        
        if (len(sequence_info['utr5_sequence']) + len(sequence_info['cds_sequence'])/3 + len(sequence_info['utr3_sequence'])) > 8000:
            print(f"Skip {transcript_id} due to length.")
            continue 
        
        print(len(sequence_info['utr5_sequence']), len(sequence_info['cds_sequence']), len(sequence_info['utr3_sequence']))
        weights = model.get_attention_weights(
            protein_sequence=sequence_info['protein_sequence'], 
            utr5_sequence=sequence_info['utr5_sequence'].upper().replace("T", "U"), 
            cds_sequence=sequence_info['cds_sequence'].upper(), 
            utr3_sequence=sequence_info['utr3_sequence'].upper().replace("T", "U"),
            hidden_layer_idx=args.hidden_layer_idx
        )
        
        if args.average:
            attension_layers = []
            for layer_idx in range(len(weights)):
                _weight = weights[layer_idx]
                _weight = (_weight - _weight.mean(dim=(2,3), keepdim=True)) / _weight.std(dim=(2,3), keepdim=True)
                _weight = _weight.mean(dim=(0, 1))
                attension_layers.append(_weight)
            weights = torch.stack(attension_layers, dim=0)  # shape [num_layers, seq_len, seq_len]
            weights = weights.mean(dim=0)  # shape [seq_len, seq_len]

        rna_embeddings = {
            transcript_id: weights,
        }
        
        torch.save(rna_embeddings, _output_path)
        
if __name__ == '__main__':
    main()




