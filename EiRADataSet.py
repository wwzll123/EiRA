import pickle
import os
import h5py
import torch
import random
import threading
import queue
import numpy as np
from esm.utils.constants import esm3 as C
from torch.utils.data import Dataset, DataLoader
from esm.utils import noise_schedules as noise


class EiRA_inference_DataSet(Dataset):
    def __init__(self, ESMProtein_dict):
        self.features = list(ESMProtein_dict.values())
        self.pro_ids = list(ESMProtein_dict.keys())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.pro_ids[idx],self.features[idx]

    def collate_fn(self,batch):
        return batch


class noise_EiRA_Dataset(Dataset):
    def __init__(self,
                 feature_dir:str,
                 idx_proid_dict:dict[str, str],
                 pro2id_dict:dict[str, int],
                 hdf5_or_npz='hdf5',
                 max_seq_len=768,
                 padding_threshold=50,
                 max_batch_num=17
                 ):


        self.max_seq_len = max_seq_len
        self.idx_proid_dict = idx_proid_dict#这是训练用的少部分蛋白质字典
        self.features = [None] * len(idx_proid_dict)
        self.feature_dir = feature_dir
        self.padding_threshold = padding_threshold
        self.hdf5_or_npz=hdf5_or_npz
        self.max_batch_num=max_batch_num
        self.pro2id_dict=pro2id_dict#这是那300多w


    def __len__(self):
        return len(self.features)

    # lazy load
    def __getitem__(self, idx):
        # 加载当前蛋白质样本的 npz 数据
        if self.features[idx] is None:
            
            if self.hdf5_or_npz == 'hdf5':
                pro_id=self.idx_proid_dict[str(idx)]
                data=self.read_token_from_hdf5(pro_id)
            elif self.hdf5_or_npz == 'npz':
                proid = self.idx_proid_dict[str(idx)]
                file_path = os.path.join(self.feature_dir, f"{proid}_tokens.npz")
                data = np.load(file_path)
            else:raise ValueError("hdf5_or_npz value error! it should be npz or hdf5!")

            self.features[idx] = data
        else:
            data = self.features[idx]  # 已加载，直接读取

        sequence = data["sequence"]
        structure = data["structure"]
        sa = data["sa"]
        ss = data["ss"]
        
        function = data["function"]
        coordinates = data["coordinates"]

        # 堆叠并合并 tokens 和 function
        tokens = np.stack([sequence, structure, sa, ss], axis=1).astype(np.uint16)
        combined = np.concatenate([tokens, function], axis=1)[1:-1]

        # 切分子序列并填充
        sub_combined_list = []
        sub_coordinates_list = []
        sub_lengths = []


        for start in range(0, combined.shape[0], self.max_seq_len):
            end = min(start + self.max_seq_len, combined.shape[0])

            subsequence_length = end - start

            # 如果差值小于设定的阈值，则跳过该子序列
            if subsequence_length < self.padding_threshold: continue

            # 填充 combined
            sub_combined = np.zeros((self.max_seq_len, 12), dtype=np.uint16)
            sub_combined[:end - start] = combined[start:end]
            # 填充PAD符号
            sub_combined[end - start:, 0] = C.SEQUENCE_PAD_TOKEN
            sub_combined[end - start:, 1] = C.STRUCTURE_PAD_TOKEN

            # 填充 coordinates
            sub_coordinates = np.full((self.max_seq_len, 37, 3), np.nan, dtype=np.float32)
            sub_coordinates[:end - start] = coordinates[start:end]

            sub_lengths.append(subsequence_length)
            sub_combined_list.append(sub_combined)
            sub_coordinates_list.append(sub_coordinates)

        return sub_combined_list, sub_coordinates_list,sub_lengths


    def random_select_rows(self,tensor, indices):
        # 获取第一维的大小
        N = tensor.size(0)
        if N<=indices.shape[0]:return tensor
        # 从原始 tensor 中提取对应的行
        selected_tensor = tensor[indices]
        return selected_tensor

    def custom_collate_fn(self, batch):

        combined, coordinates,lengths = [], [], []

        for (sub_combined, sub_coordinates,sub_lengths) in batch:
            combined.extend(sub_combined)
            coordinates.extend(sub_coordinates)
            lengths.extend(sub_lengths)

        # 先转换为 NumPy 数组，再转换为 PyTorch 张量
        combined = torch.tensor(np.array(combined), dtype=torch.int32)
        coordinates = torch.tensor(np.array(coordinates), dtype=torch.float32)
        lengths = torch.tensor(lengths, dtype=torch.int32)
        batch_mask_token,batch_mask_position=self.mask_batch_all_track_input_token(combined, lengths)
        mask_coord_tensor=self.mask_coord(coordinates,lengths)
        indices = torch.randperm(combined.shape[0])[:self.max_batch_num]
    
        return (self.random_select_rows(combined,indices),
                self.random_select_rows(mask_coord_tensor,indices),
                self.random_select_rows(lengths,indices),
                self.random_select_rows(batch_mask_token,indices),
                self.random_select_rows(batch_mask_position,indices))

    def mask_batch_all_track_input_token(self, input_token_tensor, valid_protein_len):
        SASA_UNK_TOKEN,SS8_UNK_TOKEN=2,2
        mask_tokens = [C.SEQUENCE_MASK_TOKEN, C.STRUCTURE_MASK_TOKEN, SASA_UNK_TOKEN, SS8_UNK_TOKEN,
                       C.INTERPRO_PAD_TOKEN]
        batch_size, seq_length, _ = input_token_tensor.shape
        batch_mask_token = torch.zeros((batch_size, seq_length, 12), dtype=input_token_tensor.dtype,
                                       device=input_token_tensor.device)
        batch_mask_position = torch.zeros((batch_size, seq_length, 5), dtype=torch.bool,
                                          device=input_token_tensor.device)

        for i in range(5):
            one_track_input_token = input_token_tensor[:, :, 4:] if i == 4 else input_token_tensor[:, :, i].unsqueeze(-1)
            per_sample_mask_length = self.mask_length(batch_size, valid_protein_len, i)
            one_track_masked_input_token, one_track_mask_positions = self.generate_mask_position(
                one_track_input_token,
                per_sample_mask_length,
                valid_protein_len,
                mask_tokens[i]
            )
            if i==4:
                batch_mask_token[:, :, i:] = one_track_masked_input_token
            else:
                batch_mask_token[:, :, i] = one_track_masked_input_token.squeeze(-1)
            batch_mask_position[:, :, i] = one_track_mask_positions.squeeze(-1)

        return batch_mask_token, batch_mask_position


    def get_hdf5_file_for_protein(self,protein_index, max_group_per_file=10000, hdf5_file_prefix='UniBind40_part'):
        """根据蛋白质ID返回其对应的HDF5文件名"""
        file_index = protein_index // max_group_per_file+1
        return f"{hdf5_file_prefix}_{file_index}.h5"

    def read_hdf5_from_hdf5_file(self, hdf5_file,protein_id)-> dict:
        one_pro_token_dicts={}
        with h5py.File(hdf5_file, 'r') as file:
            group = file[protein_id]
            one_pro_token_dicts['sequence'] = group["sequence"][:]
            one_pro_token_dicts["structure"] = group["structure"][:]
            one_pro_token_dicts["sa"] = group["sa"][:]
            one_pro_token_dicts["ss"] = group["ss"][:]
            one_pro_token_dicts["function"] = group["function"][:]
            one_pro_token_dicts["coordinates"] = group["coordinates"][:]
        return one_pro_token_dicts

    def read_token_from_hdf5(self, protein_id):
        protein_idx=self.pro2id_dict[protein_id]
        hdf5_file_name=self.get_hdf5_file_for_protein(protein_idx)
        hdf5_file_path=os.path.join(self.feature_dir,hdf5_file_name)
        return self.read_hdf5_from_hdf5_file(hdf5_file_path,protein_id)


    def generate_mask_position(self, input_token: torch.Tensor,
                               per_sample_mask_length: torch.Tensor,
                               valid_length: torch.Tensor,
                               mask_token: int):
        """
        对输入 token 进行随机掩码，并返回掩码后的 token 和掩码位置。

        参数:
            input_token (torch.Tensor): 输入 token，形状为 (batch_size, seq_length, token_size)。
            per_sample_mask_length (torch.Tensor): 每个样本需要掩码的 token 数量，形状为 (batch_size,)。
            valid_length (torch.Tensor): 每个样本的有效长度，形状为 (batch_size,)。
            mask_token (int): 用于替换被掩码 token 的值。

        返回:
            masked_input_token (torch.Tensor): 掩码后的 token，形状与 input_token 相同。
            mask_positions (torch.Tensor): 每个样本被掩码的位置，形状为 (batch_size, seq_length)。
        """
        batch_size, seq_length, token_size = input_token.shape

        # 初始化掩码位置张量
        mask_positions = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=input_token.device)

        # 遍历每个样本
        for i in range(batch_size):
            # 获取当前样本的有效长度和需要掩码的长度
            current_valid_length = valid_length[i].item()
            current_mask_length = per_sample_mask_length[i].item()

            # 确保需要掩码的长度不超过有效长度
            if current_mask_length > current_valid_length:
                raise ValueError(
                    f"Mask length {current_mask_length} exceeds valid length {current_valid_length} for sample {i}")

            # 在有效长度范围内随机选择需要掩码的位置
            indices = torch.randperm(current_valid_length)[:current_mask_length]
            mask_positions[i, indices] = True  # 标记需要掩码的位置

        # 将需要掩码的位置替换为 mask_token
        masked_input_token = input_token.clone()
        masked_input_token[mask_positions.unsqueeze(-1).expand(-1, -1, token_size)] = mask_token

        return masked_input_token, mask_positions.unsqueeze(-1)

    def mask_length(self,batch,valida_length,track_num):

        if track_num == 0:type='BetaUni30'
        elif track_num == 1:type='cosine'
        else:type='square_root_schedule'
        #得到每一个样本被mask掉的比例

        sample_rate=self.noise_schedule(batch,type)
        per_sample_mask_length=(sample_rate*valida_length).int()
        return per_sample_mask_length


    def mask_coord(self,coord_tensor,valida_length):
        # coord:B,L,37,3
        linear_noise=torch.rand(coord_tensor.shape[0])
        cubic_noise=noise.cubic_schedule(linear_noise)
        mask_length=(valida_length*cubic_noise).int()

        for one_coord,one_mask_length,one_valida_length in zip(coord_tensor,mask_length,valida_length):
            indices = torch.randperm(one_valida_length)[:one_mask_length]
            one_coord[indices]=torch.nan
        return coord_tensor

    def BetaUniform30_schedule(self,N,p=0.3):
        # N:batch size
        # 第一步：先产生一个布尔向量，用于决定每个样本来自哪种分布
        # choice[i] = True  表示来自 Beta(3,9)
        # choice[i] = False 表示来自 Uniform(0,1)

        choice = np.random.rand(N) < p
        # 第二步：分别采样 Beta(3,9) 和 Uniform(0,1)
        alpha, beta = 3, 9
        samples_beta = np.random.beta(alpha, beta, size=N)#beta分布
        samples_uniform = np.random.rand(N)#Uni分布
        # 第三步：根据 choice 将两种分布的结果合并
        samples = np.where(choice, samples_beta, samples_uniform)
        return torch.from_numpy(samples)


    def noise_schedule(self,batch,noise_type):
        if noise_type in noise.NOISE_SCHEDULE_REGISTRY:
            uniform_noise=torch.rand(batch)
            return noise.NOISE_SCHEDULE_REGISTRY.get(noise_type)(uniform_noise)
        else:
            return self.BetaUniform30_schedule(batch)

if __name__ == '__main__':
    # 加载 npz 文件并预处理
    with open(r"E:\EiRA\data\EiRA_test_prompt_DBP276.pkl", 'rb') as fi:
        ESMpro_list=list(pickle.load(fi).values())
    dataset=EiRA_inference_DataSet(ESMpro_list)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0,collate_fn=dataset.collate_fn)
    for batch in dataloader:
        print(batch)
        print(len(batch))
        for protein in batch:
            print(protein)