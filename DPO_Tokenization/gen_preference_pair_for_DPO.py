import sys,os
from esm.sdk.api import ESMProtein, GenerationConfig
from peft import PeftModel
import torch
from esm.models.esm3 import ESM3
import argparse
from collections import Counter
import numpy as np
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
import heapq
import re


parser = argparse.ArgumentParser()
parser.add_argument('--pro_id', type=str, required=True)
parser.add_argument('--lora_checkpoint_dir', type=str, required=True)
parser.add_argument('--sequence', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
config = parser.parse_args()


class Protein:
    def __init__(self, id:int, seq:str, plddt:float,ptm:float):
        self.id = id
        self.seq = seq
        self.plddt = plddt
        self.ptm = ptm

    def __lt__(self, other):
        return self.ptm < other.ptm


def analyze_string(s, n):
    # 统计字符总出现次数[1,5,7](@ref)
    total_counts = Counter(s)

    # 计算最大连续重复次数
    max_repeat_num = 0
    current_char = None
    current_streak = 0

    for char in s:
        if char == current_char:
            current_streak += 1
        else:
            current_char = char
            current_streak = 1
        max_repeat_num = max(max_repeat_num, current_streak)

    # 获取前N高频字符[1,5,7](@ref)
    top_n = dict(total_counts.most_common(n))

    return top_n, max_repeat_num

def check_is_seq_redundancy(seq:str,top_k:int,allowed_repeat_num:int,percent:float):
    top_n, max_repeat_num=analyze_string(seq,top_k)
    return sum(top_n.values())>percent * len(seq) and max_repeat_num>allowed_repeat_num

#大于0.7的win_num就是5，否则是3
#对于最低ptm的都大于0.8的（或者最大的小于0.5）就不处理了
def extract_win_loss_pair(pro_id,file_path,top_K=2,pTM_thresholed=0.5,win_num=3,loss_num=6):
    #返回win_dict和loss_dict
    win_dict,loss_dict={},{}
    # 正则表达式匹配 plddt 和 ptm 值
    plddt_pattern = re.compile(r'plddt:([\d.]+)')
    ptm_pattern = re.compile(r'ptm:([\d.]+)')

    min_heap = []
    max_heap = []
    max_ptm=0
    min_ptm=1
    # 打开文件并逐行读取
    num=1
    with open(file_path, 'r') as file:
        for line in file:
            # 如果行以 ">" 开头，表示是蛋白质样本的描述行
            if line.startswith('>'):
                pro_name = pro_id+f'_gen_{num}'
                seq=next(file).strip()
                num+=1
                # 使用正则表达式提取 plddt 和 ptm 值
                plddt_match = plddt_pattern.search(line)
                ptm_match = ptm_pattern.search(line)

                plddt=float(plddt_match.group(1))
                ptm=float(ptm_match.group(1))
                max_ptm=max(max_ptm,ptm)
                min_ptm = min(min_ptm, ptm)
                #这种情况直接丢入loss_dict
                if check_is_seq_redundancy(seq,top_K,
                                           allowed_repeat_num=7,
                                           percent=0.4) or seq.count('X')>0.08*len(seq):
                    loss_dict[pro_name]=Protein(pro_name,seq,plddt,ptm)
                    continue

                heapq.heappush(min_heap,Protein(pro_name,seq,plddt,ptm))
                heapq.heappush(max_heap,Protein(pro_name,seq,-plddt,-ptm))
    if min_ptm>0.8 or max_ptm<0.5:return {},{}
    if max_ptm>0.7:win_num,pTM_thresholed=5,0.6
    elif max_ptm > 0.6: win_num, pTM_thresholed = 4, 0.5
    for num in range(min(win_num,len(max_heap))):
        protein=heapq.heappop(max_heap)
        if -protein.ptm >=pTM_thresholed:
            win_dict[protein.id]=protein
    for num in range(min(loss_num,len(min_heap))):
        protein=heapq.heappop(min_heap)
        loss_dict[protein.id]=protein
    return win_dict,loss_dict


def check_ptm_higher_than_thre(gen_path,ptm_thre):
    if not os.path.exists(gen_path):return False
    max_ptm=0
    with open(gen_path,'r') as fi:
        for one_line in fi:
            one_line=one_line.strip()
            if one_line.startswith('>'):
                index=one_line.rfind(':')
                ptm=float(one_line[index+1:])
                max_ptm=max(max_ptm,ptm)
    return max_ptm>=ptm_thre



def generate_candidate_pair(protein_id:str,
                            input_prompt_esm_protein:ESMProtein,
                            save_dir:str,
                            model:PeftModel,
                            number_of_generations:int=30,):
    save_seq=''
    for index in range(number_of_generations):
        sequence_generation = model.generate(
            input_prompt_esm_protein,
            GenerationConfig(
                track="sequence",
                num_steps=input_prompt_esm_protein.sequence.count("_") // 12,
                temperature=1
            ),
        )

        # 生成结构
        structure_prediction:ESMProtein = model.generate(
            ESMProtein(sequence=sequence_generation.sequence, coordinates=input_prompt_esm_protein.coordinates),
            GenerationConfig(
                track="structure", num_steps=len(input_prompt_esm_protein) // 24, temperature=0.5
            ),
        )
        save_seq+='>'+protein_id+f'_gen_{index+1}'+'|plddt:'+str(round(structure_prediction.plddt.mean().item(),4))+\
                   '|ptm:'+str(round(structure_prediction.ptm.item(),4))+'\n'+sequence_generation.sequence+'\n'

    with open(save_dir+os.sep+protein_id+f"_gen.fasta", "w") as file:
        file.write(save_seq)


if __name__ == '__main__':
    model = ESM3.from_pretrained("esm3_sm_open_v1", device=torch.device('cuda:0'))
    eira_model = PeftModel.from_pretrained(model, config.lora_checkpoint_dir)
    eira_model.eval()
    protein=ESMProtein(config.sequence)
    seq_tokenizer = EsmSequenceTokenizer()

    #1. generate candidate sequences
    generate_candidate_pair(config.pro_id, protein, config.save_dir, eira_model, number_of_generations=30)

    win_dict, loss_dict = extract_win_loss_pair(config.pro_id, config.save_dir+os.sep+config.pro_id+f"_gen.fasta")

    # 写入文件
    pair_num = 1
    for one_win_id, one_win_protein in win_dict.items():
        for one_loss_id, one_loss_protein in loss_dict.items():

            if -one_win_protein.ptm - one_loss_protein.ptm < 0.1: continue  # choose和reject的ptm差值必须大于0.1
            win_token = seq_tokenizer.encode(one_win_protein.seq)[1:-1]
            loss_token = seq_tokenizer.encode(one_loss_protein.seq)[1:-1]
            seq_prompt = seq_tokenizer.encode(protein.sequence)[1:-1]
            # pdbid_0_pair
            if len(seq_prompt) == protein.coordinates.shape[0] == len(win_token) == len(loss_token):
                np.savez_compressed( config.save_dir + os.sep + config.pro_id + f'_{pair_num}_pair.npz',
                                    sequence=np.uint16(seq_prompt),
                                    coordinates=protein.coordinates.numpy(),
                                    win_token=np.uint16(win_token),
                                    loss_token=np.uint16(loss_token))
                pair_num += 1