import os,sys
from peft import PeftModel
import numpy as np
import torch
from esm.utils.constants import esm3 as C
import biotite.sequence as seq
import biotite.sequence.align as align
from esm.sdk.api import GenerationConfig, ESMProtein
from esm.models.esm3 import ESM3
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--inform_position', type=str,default='0,1,2,3,5,6,7,8,9')
parser.add_argument('--weight_dir', type=str,default=r'E:\science\EiRA\model_check_point')
parser.add_argument('--SRC_PDB_path', type=str,default='./8exa_pdbfix.pdb')
parser.add_argument('--designed_seq_save_path', type=str, default='./designed_seq.fasta')
parser.add_argument('--design_num', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--chain', type=str, default='A')

config = parser.parse_args()

def run_design(TnpB_protein_tensor_prompt):
    num_tokens_to_decode = min((TnpB_protein_tensor_prompt.structure == 4096).sum().item(), 20)
    TnpB_structure_generation = model.generate(TnpB_protein_tensor_prompt, GenerationConfig(
        track="structure", num_steps=num_tokens_to_decode, temperature=1
    ))

    structure_generation_protein = model.decode(TnpB_structure_generation)
    num_tokens_to_decode = min((TnpB_protein_tensor_prompt.sequence == 32).sum().item(), 20)
    TnpB_protein_tensor = model.generate(TnpB_structure_generation, GenerationConfig(
        track="sequence", num_steps=num_tokens_to_decode, temperature=0.5
    ))

    TnpB_protein_tensor.structure = None
    sequence_generation = model.generate(
        TnpB_protein_tensor,
        GenerationConfig(track="structure", num_steps=1, temperature=0.0),
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


if __name__ == '__main__':
    model = ESM3.from_pretrained("esm3_sm_open_v1", device=torch.device(config.device))
    model=PeftModel.from_pretrained(model,
                                    config.weight_dir+os.sep+'DPO_checkpoint_VanillaLora_part_data_no_repeat').to(torch.bfloat16)
    model.eval()

    #load a PDB file and MASK residues
    prompt_position = torch.tensor(list(map(int,config.inform_position.split(',')))).int()
    Tnpb_protein = ESMProtein.from_pdb(config.SRC_PDB_path, chain_id=config.chain)
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
            designed_seq, plddt, ptm, identity2temp = run_design(TnpB_protein_tensor)
            with open(config.designed_seq_save_path, 'a') as f:
                f.write(f'>design_{i+1}|plddt:{plddt:.2f}|plddt:{ptm:.2f}|identity:{identity2temp:.2f}\n{designed_seq}\n')
