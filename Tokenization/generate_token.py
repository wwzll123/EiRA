
import traceback
import argparse
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.utils.types import FunctionAnnotation
import numpy as np
import json
import re
from esm.tokenization import get_esm3_model_tokenizers, InterProQuantizedTokenizer
from tqdm import tqdm
from esm.pretrained import ESM3_structure_encoder_v0
from esm.utils import encoding
import os

parser = argparse.ArgumentParser()

parser.add_argument('--protein_list_file',
                    type=str,
                    default='./protein_list.txt')

parser.add_argument('--interpro_path',
                    type=str,
                    default='./esm3_UniDRBP40_InterPro.json')

parser.add_argument('--pdb_dir',
                    type=str,
                    default='./pdb_dir')

parser.add_argument('--ss_json_path',
                    type=str,
                    default='./esm3_DRBP_ss.json')

parser.add_argument('--target_dir',
                    type=str,
                    default='./token_dir')


config = parser.parse_args()


#model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1")

structure_encoder=ESM3_structure_encoder_v0().cuda()
tokenizer_collection=get_esm3_model_tokenizers()
function_tokenizer=InterProQuantizedTokenizer()
str_tokenizer=tokenizer_collection.structure


# interpro_path=r"F:\database\esm3_UniDRBP40_InterPro.json"
# pdb_dir=r"F:\database\high_plddt_esm3_newDRBP"
# ss_json_path=r"F:\database\esm3_DRBP_ss.json"
# target_dir=r"F:\database\new_DRBP_token"
plddt_json={}

with open(config['ss_json_path'], 'r') as f:
    ss_dicts=json.load(f)

with open(config['interpro_path'], 'r') as f:
    inter_pro_dicts=json.load(f)


def parse_string(input_str):
    # 定义正则表达式模式
    pattern = r'(\w+)\((\d+)-(\d+)\)'
    match = re.match(pattern, input_str)

    if match:
        # 提取括号前的内容
        prefix = match.group(1)
        # 提取括号内的数字，并转换为整数列表
        numbers = [int(num) for num in match.groups()[1:]]
        return prefix, numbers
    else:return None,None



def generate_token(protein_id):
    pdb_path=config['pdb_dir']+os.sep+protein_id+'.pdb'
    target_path=config['target_dir'] + os.sep + f"{protein_id}_tokens.npz"
    #if not os.path.exists(pdb_path) or os.path.exists(target_path):return
    chain = ProteinChain.from_pdb(pdb_path)
    #plddt_json[protein_id] = round(chain.confidence.mean(), 4)
    #if chain.confidence.mean()<70:return

    protein = ESMProtein.from_protein_chain(chain)
    #溶剂可及性
    protein.sasa=chain.sasa()
    #二级结构
    protein.secondary_structure=ss_dicts[protein_id].replace('-','C')
    protein.function_annotations=[]
    if protein_id in inter_pro_dicts:
        interpro_record=inter_pro_dicts[protein_id]
        if interpro_record!='':
            interpro_id_list=interpro_record.strip().split('\t')
            for one_interpro_id in interpro_id_list:
                prefix, motif_range=parse_string(one_interpro_id)
                if prefix is None or motif_range[0]>=len(protein.sequence):continue
                motif_range[1]=min(motif_range[1],len(protein.sequence))
                protein.function_annotations.append(FunctionAnnotation(label=prefix,start=motif_range[0],end=motif_range[1]))

    res = function_tokenizer.tokenize(protein.function_annotations, len(protein))
    function_tokens = function_tokenizer.encode(res, add_special_tokens=True)
    sequence_tokens = np.array(tokenizer_collection.sequence.encode(protein.sequence))
    sa_tokens = tokenizer_collection.sasa.encode(protein.sasa)
    ss_tokens = tokenizer_collection.secondary_structure.encode(protein.secondary_structure).numpy()

    coordinates, _, structure_tokens = encoding.tokenize_structure(
        coordinates=protein.coordinates.cuda(),
        structure_encoder=structure_encoder,
        structure_tokenizer=str_tokenizer,
        reference_sequence=protein.sequence,
        add_special_tokens=True,
    )

    if ss_tokens.shape[0]==sequence_tokens.shape[0]-1:
        ss_tokens=np.append(ss_tokens,ss_tokens[-1])
        ss_tokens[-2]=ss_tokens[-3]

    np.savez_compressed(target_path,
                        sequence=np.uint16(sequence_tokens),
                        structure=np.uint16(structure_tokens.cpu().detach().numpy()),
                        sa=np.uint16(sa_tokens.cpu().numpy()),
                        ss=np.uint16(ss_tokens),
                        function=np.uint16(function_tokens.cpu().numpy()),
                        coordinates=np.float32(protein.coordinates.cpu().numpy()))



if __name__ == '__main__':
    # fasta_path=r"E:\science\EiRA\database\UniDRBP40_toUniBind.fasta"
    # with open(fasta_path, 'r') as file:
    #     protein_ids = [line[1:].strip() for line in file if line.startswith('>')]
    protein_ids=np.loadtxt(config['protein_list_file'],dtype=str)
    for one_protein_id in tqdm(protein_ids):
        target_path = config['target_dir'] + os.sep + f"{one_protein_id}_tokens.npz"
        if not os.path.exists(config['pdb_dir']+os.sep+one_protein_id+'.pdb') or os.path.exists(target_path):continue
        try:
            generate_token(one_protein_id)
        except Exception as e:
            traceback.print_exc()

# with open(r'F:\database\DRBP_ESM3_plddt.json', 'w') as f:
#     json.dump(plddt_json, f)
