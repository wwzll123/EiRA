import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast
import pickle
import re
import gc
import traceback
from peft import LoraConfig, get_peft_model,AdaLoraConfig,LoKrConfig
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.models.esm3 import ESM3
import EiRADataSet
import EiRA_Loss
import argparse
import torch.distributed as dist
from bitsandbytes.optim import AdamW8bit
import logging

# torchrun --nproc_per_node=7 --master_port=29512 pretrain_with_noise.py --gpu 0,1,2,3,4,5,6,7

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
parser = argparse.ArgumentParser()
parser.add_argument('--fasta_list', type=str, default='./UniBind40_token_remove_repeat.txt')
parser.add_argument('--token_dir', type=str, default='/root/autodl-fs/zww/UniBind40_token')
parser.add_argument('--save_path', type=str, default='/root/autodl-tmp/res')
parser.add_argument("--gpu", type=str, default="4", help='Local process rank.')
parser.add_argument("--batch_size", type=int, default=20, help='Batch size.')
parser.add_argument("--fine_tuning_num", type=int, default=16, help='fine_tuning_num.')
parser.add_argument("--epochs", type=int, default=5, help='epochs')
parser.add_argument("--prefetch_factor", type=int, default=30, help='prefetch_factor.')
parser.add_argument("--num_workers", type=int, default=16, help='num_workers')


config = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
#domain_region_path = config.domain_region_path
token_path = config.token_dir

os.makedirs(config.save_path, exist_ok=True)

fine_tuning_num = int(config.fine_tuning_num)
total_layers = 48
batch_size = int(config.batch_size)
epochs = int(config.epochs)
temp=1
ft_type='lora'
model_type = f'{ft_type}_ft{fine_tuning_num}_repeat_penalty'
max_test_seq_len = 2000
max_batch_num = batch_size

logging.basicConfig(
    filename=f'./EiRA_train_{model_type}.log',  # 日志文件的名字
    filemode='a',  # 文件模式：'a' 追加（默认），'w' 覆盖
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', init_method='env://')
device = torch.device("cuda", local_rank)


pro_list = np.loadtxt(config.fasta_list, dtype=str)

idx_proid_dict = {str(i): pro_id for i, pro_id in enumerate(pro_list)}
pro2id_dict={one:i for i,one in enumerate(pro_list)}


def train():
    if local_rank == 0: logging.info('begin to load ESM3')
    # 选择模型和 tokenizer
    model = ESM3.from_pretrained("esm3_sm_open_v1",device=device)

    # LoRA or 全量
    model, optimizer = get_wrapped_model_optimizer(model, fine_tuning_num, model_type=ft_type)
    # 这里需要指定所有的linear层
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=False)
    EiRA_train_dataset = EiRADataSet.noise_EiRA_Dataset(config.token_dir,
                                                  idx_proid_dict,
                                                  pro2id_dict=pro2id_dict,#330w条蛋白质映射回idx
                                                  hdf5_or_npz='npz',
                                                  max_batch_num=max_batch_num,
                                                  max_seq_len=768)


    train_sampler = DistributedSampler(EiRA_train_dataset)
    dataloader = DataLoader(
        EiRA_train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=EiRA_train_dataset.custom_collate_fn,
        prefetch_factor=int(config.prefetch_factor),
        num_workers=int(config.num_workers),
        pin_memory=True,
        drop_last=True
    )

    if local_rank == 0: logging.info('DataLoader load Over')


    fun_losser, seq_losser, str_losser, ss_losser, sa_losser = EiRA_Loss.get_loss_set(weight_factor=1.3)

    model = model.to(device)

    if local_rank == 0: logging.info('begin training!')

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        if local_rank == 0: logging.info(f'epoch:[{epoch}], begin')
        epoch_loss = 0
        model.train()

        for i, batch in enumerate(dataloader):
            if i % 2500 == 0 and local_rank == 0:
                logging.info(f'Rank {local_rank}, epoch:[{epoch}], begin, batch:[{i}]')
                print(f'Rank {local_rank}, epoch:[{epoch}], begin, batch:[{i}]')
            # batch_token:batch_size*context_len*12

            batch_token, batch_coords, batch_lengths, masked_token_batch, all_track_mask_position_tensor = batch
            batch_token, batch_coords = batch_token.long(), batch_coords.float()

            optimizer.zero_grad()
            masked_token_batch, batch_coords = masked_token_batch.to(device).long(), batch_coords.to(device)
            with autocast(dtype=torch.bfloat16):  # 启用自动混合精度
                outputs = model(
                    sequence_tokens=masked_token_batch[:, :, 0].squeeze(),
                    structure_tokens=masked_token_batch[:, :, 1].squeeze(),
                    ss8_tokens=masked_token_batch[:, :, 3].squeeze(),
                    sasa_tokens=masked_token_batch[:, :, 2].squeeze(),
                    function_tokens=masked_token_batch[:, :, 4:].squeeze(),
                    structure_coords=batch_coords,  # 只需要backbone的坐标,batch_size*context_length*3*3
                )

                batch_token = batch_token.to(device)
                # 将target lab中的special token变成0
                str_position = batch_token[:, :, 1] >= 4096
                batch_token[:, :, 1][str_position] = 0

                #batch*seq_length*5
                seq_loss = seq_losser(
                    outputs.sequence_logits, batch_token[:, :, 0],
                    all_track_mask_position_tensor[:,:,0],
                    [])

                str_loss = str_losser(
                    outputs.structure_logits, batch_token[:, :, 1],
                    all_track_mask_position_tensor[:, :, 1],
                    [])

                ss_loss = ss_losser(
                    outputs.secondary_structure_logits, batch_token[:, :, 3],
                    all_track_mask_position_tensor[:, :, 3],
                    [])

                sa_loss = sa_losser(
                    outputs.sasa_logits, batch_token[:, :, 2],
                    all_track_mask_position_tensor[:,:, 2],
                    [])

                fun_loss = fun_losser(
                    outputs.function_logits,
                    batch_token[:, :, 4:],
                    all_track_mask_position_tensor[:,:,4])

                loss = seq_loss + 0.5*str_loss + 0.01*ss_loss + 0.01*sa_loss + 0.01*fun_loss
                epoch_loss += loss.item()


            loss.backward()
            optimizer.step()
            #model.module.base_model.update_and_allocate(i)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)

        torch.cuda.empty_cache()
        gc.collect()

        logging.info(f"Rank {local_rank}, Epoch {epoch}, Loss: {loss.item()}")

    if dist.get_rank() == 0:
        model.module.save_pretrained(config.save_path + f'/EiRA_checkpoint_{model_type}')


def get_wrapped_model_optimizer(model, fine_tune_layer_num, model_type='base'):
    if local_rank == 0: logging.info('Loaded ESM3 model Over, begin frozen param.')
    if model_type == 'base':
        for name, param in model.named_parameters():
            if 'encoder' in name: param.requires_grad = False
            if name.startswith('transformer.blocks'):
                match = re.search(r'\.([0-9]+)\.', name)
                layer_num = int(match.group(1))
                if layer_num < total_layers - fine_tune_layer_num: param.requires_grad = False
            if 'output_heads.residue_head' in name: param.requires_grad = False
        optimizer = AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4, betas=(0.9, 0.95),
                              weight_decay=0.0005, amsgrad=True)

        if local_rank == 0: logging.info('Updatabel parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad and local_rank == 0: logging.info(name)
    elif model_type == 'lora':
        update_num = total_layers - fine_tune_layer_num
        target_layers = [f'transformer.blocks.{i}.attn.layernorm_qkv.1' for i in range(update_num, total_layers)] + \
                        [f'transformer.blocks.{i}.attn.out_proj' for i in range(update_num, total_layers)] + \
                        [f'transformer.blocks.{i}.ffn.1' for i in range(update_num, total_layers)] + \
                        [f'transformer.blocks.{i}.ffn.3' for i in range(update_num, total_layers)] + \
                        ['output_heads.sequence_head.0', 'output_heads.sequence_head.3'] + \
                        ['output_heads.structure_head.0', 'output_heads.structure_head.3'] + \
                        ['output_heads.ss8_head.0', 'output_heads.ss8_head.3'] + \
                        ['output_heads.saas_head.0', 'output_heads.sasa_head.3'] + \
                        ['output_heads.function_head.0', 'output_heads.function_head.3']

        #LoRA 配置
        lora_config = LoraConfig(
            r=8,
            # task_type=TaskType.TOKEN_CLS,
            lora_alpha=32,
            lora_dropout=0.2,
            target_modules=target_layers,
            bias="lora_only",
            use_dora=True
        )

        model = get_peft_model(model, lora_config).to(torch.bfloat16)
        if local_rank == 0:model.print_trainable_parameters()

        logging.info('Transfer LoRA model Over')

        # QLoRA+优化器
        optimizer = AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4, betas=(0.9, 0.95),
                              weight_decay=0.0005, amsgrad=True)
    else:
        raise ValueError(f'Model type {model_type} not supported!')
    return model, optimizer


try:
    train()
except RuntimeError as e:
    print(f"RuntimeError: {e}")
    logging.error("An error occurred: %s", e, exc_info=True)
    traceback.print_exc()