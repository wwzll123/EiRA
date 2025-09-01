import os,sys
from peft import PeftModel
import numpy as np
import torch
from esm.utils.constants import esm3 as C
import biotite.sequence as seq
import biotite.sequence.align as align
from esm.sdk.api import GenerationConfig, ESMProtein
from esm.models.esm3 import ESM3
from DBP_ESM3 import EiRA_ESM3
import argparse
from evo2 import Evo2

parser = argparse.ArgumentParser()
parser.add_argument('--inform_position', type=str,default='0,1,2,3,5,6,7,8,9')
parser.add_argument('--weight_dir', type=str,default=r'E:\science\EiRA\model_check_point')
parser.add_argument('--SRC_PDB_path', type=str,default='./8exa_pdbfix.pdb')
parser.add_argument('--designed_seq_save_path', type=str, default='./designed_seq.fasta')
parser.add_argument('--design_num', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--chain', type=str, default='A')
parser.add_argument('--DNA', type=str, required=True)

config = parser.parse_args()

evo2_model = Evo2('evo2_7b')

def run_design(TnpB_protein_tensor_prompt,DNA_embedding):
    num_tokens_to_decode = min((TnpB_protein_tensor_prompt.structure == 4096).sum().item(), 20)
    TnpB_structure_generation = model.generate(input=TnpB_protein_tensor_prompt,
                                               DNA_embedding=DNA_embedding,
                                               config=GenerationConfig(
        track="structure", num_steps=num_tokens_to_decode, temperature=1
    ))

    structure_generation_protein = model.decode(TnpB_structure_generation)
    num_tokens_to_decode = min((TnpB_protein_tensor_prompt.sequence == 32).sum().item(), 20)
    TnpB_protein_tensor = model.generate(input=TnpB_structure_generation,
                                         DNA_embedding=DNA_embedding,
                                         config=GenerationConfig(
        track="sequence", num_steps=num_tokens_to_decode, temperature=0.5
    ))

    TnpB_protein_tensor.structure = None
    sequence_generation = model.generate(
        DNA_embedding=DNA_embedding,
        input=TnpB_protein_tensor,
        config=GenerationConfig(track="structure", num_steps=1, temperature=0.0),
    )

    # Decode to AA string and coordinates.
    sequence_generation_protein = model.decode(sequence_generation)
    temp_seq = seq.ProteinSequence(Tnpb_protein.sequence)
    designed_seq = seq.ProteinSequence(sequence_generation_protein.sequence)

    alignments = align.align_optimal(
        temp_seq, designed_seq, align.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-10, -1)
    )

    alignment = alignments[0]
    identity = align.get_sequence_identity(alignment)
    return designed_seq, sequence_generation_protein.plddt.mean().item(), sequence_generation_protein.ptm.item(), 100 * identity


def DNA_embedding(DNA_sequence):
    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize(DNA_sequence),
        dtype=torch.int,
    ).unsqueeze(0).to(config.device)
    layer_name = 'blocks.28.mlp.l3'
    outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
    return embeddings

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Instantiate the new model
    model = EiRA_ESM3(
        injection_layers=4,  # Inject into the last 4 layers
        dna_transformer_layers=2,  # Use 2 layers for the DNA transformer
        device=device
    ).to(device)
    lora_check_point_path = config.weight_dir+os.sep+"EiRA_checkpoint_DNAbinder_lora_ft32_DNAlen50"
    model = PeftModel.from_pretrained(model, lora_check_point_path, is_trainable=False)

    cross_atten_check_point_path = config.weight_dir+os.sep+"DNAbinder_lora_ft32_DNAlen50_cross_att_DNAtransformer_DNAbinder.pth"
    saved_weights = torch.load(cross_atten_check_point_path, map_location=device)

    model.base_model.dna_transformer.load_state_dict(saved_weights['dna_transformer'])
    model.base_model.cross_attention_injectors.load_state_dict(saved_weights['cross_attention_injectors'])

    model.eval()


    #DNA embedding
    embedding=DNA_embedding(config.DNA)

    #load a PDB file and MASK residues
    prompt_position = torch.tensor(list(map(int,config.inform_position.split(',')))).int()
    Tnpb_protein = ESMProtein.from_pdb(config.SRC_PDB_path, chain_id='A')
    TnpB_protein_tensor = model.encode(Tnpb_protein)

    mask = torch.zeros_like(TnpB_protein_tensor.sequence, dtype=torch.bool)
    mask[prompt_position] = True
    TnpB_protein_tensor.sequence[~mask] = C.SEQUENCE_MASK_TOKEN
    TnpB_protein_tensor.sequence[0], TnpB_protein_tensor.sequence[-1] = C.SEQUENCE_BOS_TOKEN, C.SEQUENCE_EOS_TOKEN
    TnpB_protein_tensor.structure[~mask] = C.STRUCTURE_MASK_TOKEN
    TnpB_protein_tensor.structure[0], TnpB_protein_tensor.structure[-1] = C.STRUCTURE_BOS_TOKEN, C.STRUCTURE_EOS_TOKEN
    TnpB_protein_tensor.coordinates[~mask] = torch.nan

    #run infer
    with torch.no_grad():
        for i in range(config.design_num):
            designed_seq, plddt, ptm, identity2temp = run_design(TnpB_protein_tensor,embedding)
            with open(config.designed_seq_save_path, 'a') as f:
                f.write(f'>design_{i+1}|plddt:{plddt:.2f}|plddt:{ptm:.2f}|identity:{identity2temp:.2f}\n{designed_seq}\n')

