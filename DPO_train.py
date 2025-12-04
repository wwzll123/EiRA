import os
import argparse
import numpy as np
import traceback
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from bitsandbytes.optim import AdamW8bit
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from peft import PeftModel
from DPO_dataset import EiRA_IRPO_Dataset
from DPO_loss import DPO_Loss,SeqMaskedCrossEntropyLoss_repeat_penalty



# torchrun --nproc_per_node=5 --master_port=29512 IRPO_train.py --gpu 0,1,2,3,4,5

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
parser = argparse.ArgumentParser()
parser.add_argument('--feature_dir', type=str, default='../../autodl-tmp/vanila_lora_pair',
                    help='Directory containing preference pair data')

parser.add_argument('--save_path', type=str, default='../../autodl-tmp/res')
parser.add_argument("--gpu", type=str, default="0", help='Visible GPU devices')
parser.add_argument("--batch_size", type=int, default=8, help='Batch size per GPU')
parser.add_argument("--epochs", type=int, default=5, help='Number of training epochs')
parser.add_argument("--fine_tuning_num", type=int, default=16, help='Number of transformer layers to fine-tune')
parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate')
parser.add_argument("--alpha", type=float, default=0.5, help='Alpha weight for DPO loss')
parser.add_argument("--beta", type=float, default=0.1, help='Beta parameter for DPO loss')

config = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

# DDP初始化
local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', init_method='env://')
device = torch.device("cuda", local_rank)

os.makedirs(config.save_path, exist_ok=True)

total_layers = 48
fine_tuning_num = config.fine_tuning_num
batch_size = config.batch_size
epochs = config.epochs
model_type = 'EiRA_DPO_sft_ft32_part_data'
lora_checkpoint_path = '/root/autodl-tmp/check_point/EiRA_checkpoint_vanilla_lora_ft32_repeat_penalty'
max_test_seq_len =2000 # 测试集最大序列长度
context_length = 1028
print_interval = 500



def get_wrapped_model(model, model_path, update=False):
    model = PeftModel.from_pretrained(model, model_path)

    if not update:
        # 冻结参考模型的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model
        # 48-16=32
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        if 'base_model.model.transformer.norm.weight' in name: param.requires_grad = True
        if 'output_heads' in name and 'sequence_head' not in name: param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad and local_rank == 0:
            print(name, end='--->')
            print(param.requires_grad)

    model.print_trainable_parameters()
    return model



def main():
    """主训练函数"""
    try:

        # 加载训练数据
        files = os.listdir(config.feature_dir)
        preference_pairs = [file for file in files if file.endswith('_pair.npz')]
        pro_id_dicts = {str(i): file for i, file in enumerate(preference_pairs)}

        print(f'len of tr_set:{len(pro_id_dicts)}')

        train_dataset = EiRA_IRPO_Dataset(
            feature_dir=config.feature_dir,
            idx_proid_dict=pro_id_dicts,
            max_seq_len=context_length,
            padding_threshold=50,
            max_batch_num=batch_size
        )

        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            prefetch_factor=10,
            collate_fn=train_dataset.custom_collate_fn,
            num_workers=10,
            pin_memory=True,
            drop_last=True
        )

        if local_rank == 0:
            print('DataLoader loaded successfully')

        # 加载模型
        if local_rank == 0:
            print('Loading ESM3 model...')
        policy_model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
        ref_model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)

        # 获取包装后的模型
        policy_model = get_wrapped_model(policy_model, model_path=lora_checkpoint_path, update=True)
        ref_model = get_wrapped_model(ref_model, model_path=lora_checkpoint_path)
        # 创建优化器

        optimizer = AdamW8bit(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=4e-4, betas=(0.9, 0.95),
                              weight_decay=0.0005, amsgrad=True)

        # 添加DDP
        policy_model = torch.nn.parallel.DistributedDataParallel(
            policy_model,
            device_ids=[local_rank],
            output_device=local_rank
        )

        # 创建损失函数
        dpo_losser = DPO_Loss(alpha=config.alpha, beta=config.beta).to(device)
        sft_losser=SeqMaskedCrossEntropyLoss_repeat_penalty()

        if local_rank == 0:
            print('Beginning training!')


        # 训练循环
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            if local_rank == 0:
                print(f'Epoch: [{epoch}], beginning')

            epoch_loss = 0
            policy_model.train()
            ref_model.eval()

            for i, batch in enumerate(train_dataloader):
                if i % print_interval == 0 and local_rank == 0:
                    print(f'Rank {local_rank}, epoch: [{epoch}], batch: [{i + 1}]')

                # 解包批次数据
                combined, coordinates, lengths = batch

                # 获取获胜序列和失败序列的token
                prompt_input = combined[:, :, 0].to(device).long()  # 提示token
                win_tokens = combined[:, :, 1].to(device).long()  # 获胜token
                loss_tokens = combined[:, :, 2].to(device).long()  # 失败token

                # 创建mask
                mask = torch.zeros_like(prompt_input, dtype=torch.float, device=device)
                for batch_idx, length in enumerate(lengths):
                    mask[batch_idx, :length] = 1.0

                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 获取政策模型输出
                    policy_outputs = policy_model(sequence_tokens=prompt_input, structure_coords=coordinates.cuda())
                    # 计算参考模型输出（不计算梯度）
                    with torch.no_grad():
                        ref_outputs = ref_model(sequence_tokens=prompt_input, structure_coords=coordinates.cuda())

                    # 获取logits
                    policy_logits = policy_outputs.sequence_logits
                    ref_logits = ref_outputs.sequence_logits

                    # 计算DPO损失
                    dpo_loss = dpo_losser(
                        policy_logits=policy_logits,
                        ref_logits=ref_logits,
                        winner_ids=win_tokens,
                        loser_ids=loss_tokens,
                        mask=mask
                    )
                    sft_loss=sft_losser(policy_logits,win_tokens,mask_position=mask)
                    loss=dpo_loss+0.2*sft_loss
                    #print(f'dpo_loss:{dpo_loss.item()},sft_loss:{sft_loss.item()*0.2}')
                    epoch_loss += loss.item()

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, policy_model.parameters()),
                                               max_norm=1.0)

                # 更新参数
                optimizer.step()

            if local_rank == 0:
                print(f"Rank {local_rank}, Epoch {epoch}, Loss: {epoch_loss / len(train_dataloader):.6f}")
                

            # 清理GPU缓存
            torch.cuda.empty_cache()
        policy_model.module.save_pretrained(config.save_path + f'/Checkpoint_{model_type}')


    except Exception as e:
        print(f"Error: {e}")
        with open('DPO_train_error.txt', 'a') as f:
            traceback.print_exc(file=f)

if __name__ == "__main__":
    main()
