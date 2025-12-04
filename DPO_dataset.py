import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from esm.utils.constants import esm3 as C


class EiRA_IRPO_Dataset(Dataset):
    def __init__(self,
                 feature_dir:str,
                 idx_proid_dict:dict[str, str],
                 max_seq_len=1028,
                 padding_threshold=50,
                 max_batch_num=20
                 ):

        self.max_seq_len = max_seq_len
        self.idx_proid_dict = idx_proid_dict # {'0':'pdbid_1_pair.npz','1':'pdbid_2_pair.npz',...........}
        self.features = [None] * len(idx_proid_dict)
        self.feature_dir = feature_dir
        self.padding_threshold = padding_threshold
        self.max_batch_num=max_batch_num

    def __len__(self):
        return len(self.features)

    # lazy load
    # token文件的命名方式为 pdbid_1_pair,pdbid_2_pair 意思是该蛋白质的第n个偏好对
    def __getitem__(self, idx):
        # 加载当前蛋白质样本的 npz 数据
        if self.features[idx] is None:
            file_path = self.feature_dir+os.sep+self.idx_proid_dict[str(idx)]
            data = np.load(file_path)
            self.features[idx] = data
        else:
            data = self.features[idx]  # 已加载，直接读取

        sequence_prompt = data["sequence"] #L
        coordinates_prompt = data["coordinates"] #L*37*3
        win_seq_token=data["win_token"] #L
        loss_seq_token=data["loss_token"] #L

        # 堆叠并合并 sequence tokens 和 structure tokens
        tokens = np.stack([sequence_prompt, win_seq_token,loss_seq_token], axis=1)

        # 切分子序列并填充
        sub_combined_list = []
        sub_coordinates_list = []
        sub_lengths = []

        for start in range(0, tokens.shape[0], self.max_seq_len):
            end = min(start + self.max_seq_len, tokens.shape[0])

            subsequence_length = end - start

            # 如果差值小于设定的阈值，或者全部是MASK，没有prompt,则跳过该子序列
            if subsequence_length < self.padding_threshold or np.sum(sequence_prompt!=C.SEQUENCE_MASK_TOKEN)==0: continue

            # 填充 combined
            sub_combined = np.zeros((self.max_seq_len,3))
            sub_combined[:end - start] = tokens[start:end]
            # 填充PAD符号
            sub_combined[end - start:, :] = C.SEQUENCE_PAD_TOKEN

            # 填充 coordinates
            sub_coordinates = np.full((self.max_seq_len, 37, 3), np.nan, dtype=np.float32)
            sub_coordinates[:end - start] = coordinates_prompt[start:end]

            sub_lengths.append(subsequence_length)
            sub_combined_list.append(sub_combined)
            sub_coordinates_list.append(sub_coordinates)

        return sub_combined_list, sub_coordinates_list,sub_lengths


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
        indices=torch.randperm(combined.shape[0])[:self.max_batch_num]
        return (combined[indices],coordinates[indices],lengths[indices])


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



if __name__ == '__main__':
    fea_dir=r'E:\science\EiRA\data\IRPO_data\preference_pair'
    files=os.listdir(fea_dir)
    pro_id_dicts={str(one):one_file for one,one_file in enumerate(files)}
    data_set=EiRA_IRPO_Dataset(feature_dir=fea_dir,idx_proid_dict=pro_id_dicts)
    data_loader=DataLoader(data_set,
               batch_size=20,
               shuffle=True,
               collate_fn=data_set.custom_collate_fn,
               num_workers=2,
               drop_last=True)

    for i,batch in enumerate(data_loader):
        fea_token,coordinates,lengths=batch
        print(fea_token.shape)
        print(coordinates.shape)
        if i==5:break