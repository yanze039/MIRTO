import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering

def visualize_layer_attention(
    attn_weights, 
    layer_idx, 
    protein_length, 
    utr5_length, 
    cds_length, 
    utr3_length,
    output_dir='.',
    style="default"):
    """
    attn_weights: shape [1, 8, 2203, 2203]
    """
    # Average across heads: [2203, 2203]
    if style == "longrange":
        attn_avg = attn_weights.to(torch.float32)[0][-1].detach().cpu().numpy()
    else:
        attn_avg = attn_weights.to(torch.float32)[0].mean(dim=0).detach().cpu().numpy()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # plt.figure(figsize=(12, 10))
    # make zero as white
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_blue", ["white", "skyblue", "navy"]
    )
    sns.heatmap(attn_avg, cmap=cmap, cbar_kws={'label': 'Attention Weight'})
    # add lines to separate protein, utr5, cds, utr3
    plt.axvline(x=protein_length, color='r', linestyle='--', label='Protein/5\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length, color='g', linestyle='--', label='5\'UTR/CDS Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length, color='b', linestyle='--', label='CDS/3\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length + utr3_length, color='purple', linestyle='--', label='3\'UTR/End Boundary')
    plt.title(f'Layer {layer_idx} - Attention Map (Averaged across heads)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.savefig(output_dir / f'attention_layer_{layer_idx}.png')
    plt.close()
    
   
    
    # position_importance = (position_importance - position_importance.min()) / (position_importance.max() - position_importance.min())
    # # smooth the importance scores
    plt.figure(figsize=(12, 4))
    position_importance = attn_avg.mean(axis=0)
    # position_importance = np.convolve(position_importance, np.ones(10)/10, mode='same')
    plt.plot(range(len(position_importance)), position_importance)
    plt.axvline(x=protein_length, color='r', linestyle='--', label='Protein/5\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length, color='g', linestyle='--', label='5\'UTR/CDS Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length, color='b', linestyle='--', label='CDS/3\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length + utr3_length, color='purple', linestyle='--', label='3\'UTR/End Boundary')
    plt.legend()
    plt.title(f'Layer {layer_idx} - Position Importance (Sum of Attention Weights)')
    plt.xlabel('Position')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_dir / f'position_importance_layer_{layer_idx}.png')
    plt.close()
    
    plt.figure(figsize=(12, 4))
    start_of_utr_3 = int(protein_length + utr5_length + cds_length)
    print(f"Start of 3' UTR: {start_of_utr_3}")
    import pdb; pdb.set_trace()
    three_utr_position_importance = attn_avg[start_of_utr_3:].mean(axis=0)
    print(f"3' UTR position importance shape: {three_utr_position_importance}")
    # smooth the importance scores
    # three_utr_position_importance = np.convolve(three_utr_position_importance, np.ones(10)/10, mode='same')
    plt.plot(range(len(three_utr_position_importance)), three_utr_position_importance)
    plt.axvline(x=protein_length, color='r', linestyle='--', label='Protein/5\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length, color='g', linestyle='--', label='5\'UTR/CDS Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length, color='b', linestyle='--', label='CDS/3\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length + utr3_length, color='purple', linestyle='--', label='3\'UTR/End Boundary')
    plt.legend()
    plt.title(f'Layer {layer_idx} - Position Importance (Sum of Attention Weights)')
    plt.xlabel('Position')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_dir / f'three_utr_position_importance_layer_{layer_idx}.png')
    plt.close()
    
    # clustermap
    # distance on rows of the symmetric matrix
    attn_avg_symm = (attn_avg + attn_avg.T) / 2
    attn_avg_symm = (attn_avg_symm - attn_avg_symm.min()) / (attn_avg_symm.max() - attn_avg_symm.min())
    D = pdist(attn_avg_symm, metric="cosine")
    Z = linkage(D, method="average")
    Z = optimal_leaf_ordering(Z, D)  # nicer dendrogram ordering

    g = sns.clustermap(
        attn_avg_symm,
        row_linkage=Z,
        col_linkage=Z,         # <-- same linkage = symmetric ordering
        cmap="magma",
        figsize=(10,10),
        cbar_kws={"label": "Attention Weight"},
    )
    plt.suptitle(f'Layer {layer_idx} - Attention Map Clustermap', y=1.02)
    plt.savefig(output_dir / f'attention_clustermap_layer_{layer_idx}.png')
    plt.close()
    print(f"Saved attention visualizations for layer {layer_idx} in {output_dir}")
    

def visualize_all_attention(
    attn_weights, 
    protein_length, 
    utr5_length, 
    cds_length, 
    utr3_length,
    output_dir='.',
    process_data=False
    ):
    """
    attn_weights: shape [1, 8, 2203, 2203]
    """
    if process_data:
        # attn_weights is a dict, average across all layers
        attension_layers = []
        for layer_idx in range(len(attn_weights)):
            attension_layers.append(attn_weights[layer_idx])
        attn_weights = torch.stack(attension_layers, dim=0)  # shape [num_layers, 1, 8, 2203, 2203]
        # print(f"attn_weights shape: {attn_weights.shape}")
        attn_weights = attn_weights.to(torch.float32).detach().cpu().numpy()
        # average all layers and heads
        attn_avg = attn_weights.mean(axis=(0, 1, 2))  # shape [2203, 2203]
    else:
        # Average across heads: [2203, 2203]
        attn_avg = attn_weights.to(torch.float32).detach().cpu().numpy()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f'averaged_attention_scores.npy', attn_avg)
    transcript_level_position_importance = attn_avg[protein_length:].mean(axis=0)
    print(transcript_level_position_importance.shape)
    np.save(output_dir / f'transcript_level_position_importance.npy', transcript_level_position_importance)


    # plt.figure(figsize=(12, 10))
    # make zero as white
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_blue", ["white", "skyblue", "navy"]
    )
    sns.heatmap(attn_avg, cmap=cmap, cbar_kws={'label': 'Attention Weight'})
    # add lines to separate protein, utr5, cds, utr3
    plt.axvline(x=protein_length, color='r', linestyle='--', label='Protein/5\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length, color='g', linestyle='--', label='5\'UTR/CDS Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length, color='b', linestyle='--', label='CDS/3\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length + utr3_length, color='purple', linestyle='--', label='3\'UTR/End Boundary')
    plt.title(f'Attention Map (Averaged across heads)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.savefig(output_dir / f'attention_layer.png')
    plt.close()
    
   
    
    # position_importance = (position_importance - position_importance.min()) / (position_importance.max() - position_importance.min())
    # # smooth the importance scores
    plt.figure(figsize=(12, 4))
    position_importance = attn_avg.mean(axis=0)
    # position_importance = np.convolve(position_importance, np.ones(10)/10, mode='same')
    plt.plot(range(len(position_importance)), position_importance)
    plt.axvline(x=protein_length, color='r', linestyle='--', label='Protein/5\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length, color='g', linestyle='--', label='5\'UTR/CDS Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length, color='b', linestyle='--', label='CDS/3\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length + utr3_length, color='purple', linestyle='--', label='3\'UTR/End Boundary')
    plt.legend()
    plt.title(f'- Position Importance (Sum of Attention Weights)')
    plt.xlabel('Position')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_dir / f'position_importance_layer.png')
    plt.close()
    
    plt.figure(figsize=(12, 4))
    start_of_utr_3 = int(protein_length + utr5_length + cds_length)
    three_utr_position_importance = attn_avg[start_of_utr_3:].mean(axis=0)
    
    print(three_utr_position_importance[:protein_length].mean())
    print(three_utr_position_importance[protein_length:protein_length+utr5_length].mean())
    print(three_utr_position_importance[protein_length+utr5_length:protein_length+utr5_length+cds_length].mean())
    print(three_utr_position_importance[protein_length+utr5_length+cds_length:].mean())
    np.save(output_dir / f'three_utr_position_importance.npy', three_utr_position_importance)
    # smooth the importance scores
    # three_utr_position_importance = np.convolve(three_utr_position_importance, np.ones(10)/10, mode='same')
    plt.plot(range(len(three_utr_position_importance)), three_utr_position_importance)
    plt.axvline(x=protein_length, color='r', linestyle='--', label='Protein/5\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length, color='g', linestyle='--', label='5\'UTR/CDS Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length, color='b', linestyle='--', label='CDS/3\'UTR Boundary')
    plt.axvline(x=protein_length + utr5_length + cds_length + utr3_length, color='purple', linestyle='--', label='3\'UTR/End Boundary')
    plt.legend()
    plt.title(f'Position Importance (Sum of Attention Weights)')
    plt.xlabel('Position')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_dir / f'three_utr_position_importance_layer.png')
    plt.close()
    
    
    # clustermap
    # distance on rows of the symmetric matrix
    # attn_avg_symm = (attn_avg + attn_avg.T) / 2
    # attn_avg_symm = (attn_avg_symm - attn_avg_symm.min()) / (attn_avg_symm.max() - attn_avg_symm.min())
    # D = pdist(attn_avg_symm, metric="cosine")
    # Z = linkage(D, method="average")
    # Z = optimal_leaf_ordering(Z, D)  # nicer dendrogram ordering

    # g = sns.clustermap(
    #     attn_avg_symm,
    #     row_linkage=Z,
    #     col_linkage=Z,         # <-- same linkage = symmetric ordering
    #     cmap="magma",
    #     figsize=(10,10),
    #     cbar_kws={"label": "Attention Weight"},
    # )
    # plt.suptitle(f'Layer {layer_idx} - Attention Map Clustermap', y=1.02)
    # plt.savefig(output_dir / f'attention_clustermap_layer_{layer_idx}.png')
    # plt.close()
    # print(f"Saved attention visualizations for layer {layer_idx} in {output_dir}")
    
    
import yaml
data_file = "/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/04.attention_weight/data/elm_nls_instances/refseq_mrna_filtered_with_random_utr.yaml"
with open(data_file, 'r') as f:
    all_data = yaml.safe_load(f)

data_dir = '/orcd/home/002/yanze039/orcd/pool/RNA_design/outputs_joint_sequence/04.attention_weight/attention/attention_protein_signal_random_normalize'
output_dir = Path('./attention_maps/protein_signal_random_normalize')
data_dir = Path(data_dir)
for key in all_data.keys():
    print(f"{key}:")
    if not (data_dir / f'{key}.pt').exists():
        print(f"Attention data for {key} not found, skip.")
        continue
    data = torch.load(data_dir / f'{key}.pt')
    protein_length = len(all_data[key]['protein_sequence']) + 2
    utr5_length = len(all_data[key]['utr5_sequence']) + 3
    cds_length = int(len(all_data[key]['cds_sequence'])/3 + 2)
    utr3_length = len(all_data[key]['utr3_sequence']) + 3
    total_length = protein_length + utr5_length + cds_length + utr3_length
    print(f"Protein length: {protein_length}")
    print(f"5' UTR length: {utr5_length}")
    print(f"CDS length: {cds_length}")
    print(f"3' UTR length: {utr3_length}")
    print(f"Total length: {total_length}")
    _key = key
    
    # for style in ["default", "longrange"]:
    # for style in ["longrange"]:
        # output_dir = f'./attention_maps/attention_maps_{key}_{style}'
        # for i in range(12):
        # # for i in [5]:
        #     visualize_layer_attention(
        #         data[_key][i], 
        #         layer_idx=i, 
        #         output_dir=output_dir,
        #         protein_length=protein_length,
        #         utr5_length=utr5_length,
        #         cds_length=cds_length,
        #         utr3_length=utr3_length,
        #         style=style
        #         )
    sub_output_dir = output_dir / f'attention_maps_{key}'
    if (sub_output_dir / f'averaged_attention_scores.npy').exists():
        print(f"Attention visualizations for {key} already exist, skip.")
        continue
    visualize_all_attention(
        data[_key], 
        output_dir=sub_output_dir,
        protein_length=protein_length,
        utr5_length=utr5_length,
        cds_length=cds_length,
        utr3_length=utr3_length,
        )
        