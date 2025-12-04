import numpy as np
import torch
import torch.nn as nn
from esm.utils.constants import esm3 as C



class SeqMaskedCrossEntropyLoss_repeat_penalty(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=C.SEQUENCE_PAD_TOKEN, weight_factor=1.3, 
                 repetition_threshold=7, 
                 repetition_penalty=2.0):
        """
        初始化带重复惩罚的交叉熵损失函数
        
        Args:
            ignore_index: 忽略的标签索引
            weight_factor: 域间隔的权重因子
            repetition_threshold: 触发惩罚的连续重复token阈值
            repetition_penalty: 固定的惩罚系数
        """
        super(SeqMaskedCrossEntropyLoss_repeat_penalty, self).__init__(ignore_index=ignore_index)
        self.weight_factor = weight_factor
        self.repetition_threshold = repetition_threshold
        self.repetition_penalty = repetition_penalty

    def detect_repetitions(self, preds, mask_position):
        """
        检测连续区间中的相同token，并构建惩罚矩阵
        
        Args:
            preds: shape (batch_size, protein_length, num_classes) - 预测的logits
            mask_position: shape (batch_size, protein_length) - 需要计算损失的位置掩码
            
        Returns:
            penalty_matrix: 与mask_position相同形状的惩罚矩阵
        """
        batch_size, protein_length, _ = preds.shape
        
        # 获取每个位置最可能的token id
        most_likely_tokens = torch.argmax(preds, dim=2)  # shape: (batch_size, protein_length)
        
        # 初始化惩罚矩阵，默认无惩罚
        penalty_matrix = torch.ones_like(mask_position, dtype=torch.float)
        
        # 对每个batch处理
        for b in range(batch_size):
            tokens = most_likely_tokens[b]  # (protein_length,)
            
            # 识别连续区间的边界
            tokens_shifted = torch.cat([tokens[1:], torch.tensor([-1], device=tokens.device)])
            change_indices = torch.nonzero(tokens_shifted != tokens).squeeze(-1)
            
            # 处理每个连续区间
            start_idx = 0
            for end_idx in change_indices:
                # 计算当前区间长度
                interval_length = end_idx - start_idx + 1
                
                # 如果区间长度超过阈值，对整个区间应用固定惩罚
                if interval_length >= self.repetition_threshold:
                    # 对整个区间内的位置应用固定惩罚系数
                    penalty_matrix[b, start_idx:end_idx+1] = self.repetition_penalty
                
                # 更新下一个区间的起始位置
                start_idx = end_idx + 1
        
        # 只对mask_position为1的位置应用惩罚
        penalty_matrix = torch.where(mask_position > 0, penalty_matrix, torch.ones_like(penalty_matrix))
        
        return penalty_matrix

    def forward(self, preds, targets, mask_position, domain_intervals):
        """
        Args:
            preds: Tensor of shape (batch_size, protein_length, num_classes) - Predicted logits for num_classes (4096).
            targets: Tensor of shape (batch_size, protein_length) - Integer class labels.
            mask_position: [batch_size, seq_length]
            domain_intervals: List of lists with domain residue intervals for each protein in the batch.
                              These intervals will have increased loss weight.

        Returns:
            loss: Computed cross-entropy loss for masked positions with repetition penalty.
        """
        batch_size, protein_length, num_classes = preds.shape

        # Permute preds to (batch_size, num_classes, protein_length) for CrossEntropyLoss
        preds_permuted = preds.permute(0, 2, 1)  # Now preds has shape (batch_size, num_class, protein_length)

        # Initialize a weight matrix for valid positions
        weight_matrix = mask_position.float().to(preds.device)  # Shape (batch_size, protein_length)

        # 应用domain intervals权重
        for batch_idx, domain_intervals_batch in enumerate(domain_intervals):
            for start, end in domain_intervals_batch:
                # Increase the weight in domain intervals by factor a
                weight_matrix[batch_idx, start-1:end] *= self.weight_factor
        
        # 检测重复并构建惩罚矩阵
        repetition_penalty = self.detect_repetitions(preds, mask_position).to(preds.device)
        
        # 将重复惩罚应用到权重矩阵
        weight_matrix = weight_matrix * repetition_penalty
        
        # Calculate the base cross-entropy loss
        loss = super().forward(preds_permuted, targets)  # Apply nn.CrossEntropyLoss with permuted preds
        
        # Apply the weight matrix to the loss
        loss = (loss * weight_matrix).sum()

        # Sum the weighted loss and return
        return loss / ((weight_matrix > 0).sum())



class SeqAndStrMaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=0, weight_factor=1.3):
        super(SeqAndStrMaskedCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.weight_factor = weight_factor  # Default weight factor (for domain_intervals)

    def forward(self, preds, targets, mask_position, domain_intervals):
        """
        Args:
            preds: Tensor of shape (batch_size, protein_length, num_classes) - Predicted logits for num_classes (4096).
            targets: Tensor of shape (batch_size, protein_length) - Integer class labels.
            mask_position: [batch_size, seq_length]
            domain_intervals: List of lists with domain residue intervals for each protein in the batch.
                              These intervals will have increased loss weight.

        Returns:
            loss: Computed cross-entropy loss for masked positions.
        """
        batch_size, protein_length, num_classes = preds.shape

        # Permute preds to (batch_size, num_classes, protein_length) for CrossEntropyLoss
        preds = preds.permute(0, 2, 1)  # Now preds has shape (batch_size, num_class, protein_length)

        # Initialize a weight matrix for valid positions
        weight_matrix =mask_position.float().to(preds.device)  # Shape (batch_size, protein_length)

        for batch_idx, domain_intervals_batch in enumerate(domain_intervals):
            for start, end in domain_intervals_batch:
                # Increase the weight in domain intervals by factor a
                weight_matrix[batch_idx, start-1:end] *= self.weight_factor
        # Calculate the base cross-entropy loss
        loss = super().forward(preds, targets)  # Apply nn.CrossEntropyLoss with permuted preds and original targets
        #2*63
        # Apply the weight matrix to the loss
        loss = (loss * weight_matrix).sum()

        # Sum the weighted loss and return
        return loss/((weight_matrix>0).sum())


class FunctionMaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=0, *args, **kwargs):
        super(FunctionMaskedCrossEntropyLoss, self).__init__(ignore_index=ignore_index,*args, **kwargs)

    def forward(self, pred, lab, mask_position):
        # pred: batch,context_lenth,8,num_class
        # lab: batch, context_len, 8
        # mask_position: batch, seq_len, 1表示该位置被mask掉
        self.reduction = 'none'

        # 初始化权重矩阵
        weight = torch.zeros(pred.size(0), pred.size(1), 8, device=pred.device)

        # 将被mask的位置的权重系数置为1
        mask_position = mask_position.unsqueeze(-1).expand(-1, -1, 8).bool()  # 扩展为 (batch, seq_len, 1)

        weight[mask_position] = 1.0

        # 调整pred的维度：batch,num_class,seq_len,token_num
        pred = pred.permute(0, 3, 1, 2)

        # 计算未加权的损失
        unweight_loss = super().forward(pred, lab)#batch*seq_len*8

        # 计算加权损失
        weighted_loss = (unweight_loss * weight).sum() / weight.sum()

        return weighted_loss


class Sequence_track_loss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=C.SEQUENCE_PAD_TOKEN):
        super(Sequence_track_loss, self).__init__(
            ignore_index=ignore_index,
            reduction='none'  # 先不做汇总，得到逐元素loss
        )

    def forward(self, preds, targets, valid_length):
        """
        Args:
            preds:   (B, L, C) —— 预测logits
            targets: (B, L)    —— 真实标签
            valid_length: (B,) —— 每条序列的有效长度
        """
        # 1) 调整 preds 形状到 (B, C, L)
        #    CrossEntropyLoss 默认认为第二维 (dim=1) 是 num_classes
        preds = preds.permute(0, 2, 1)  # -> (B, C, L)

        # 2) 调用父类的 forward，得到形状 (B, L) 的逐元素 loss
        ce_loss_2d = super(Sequence_track_loss, self).forward(preds, targets)
        # ce_loss_2d shape: (B, L)

        # 3) 构造 mask，针对 valid_length，形状同 (B, L)
        device = preds.device
        B, L = targets.shape
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        mask = (idx < valid_length.unsqueeze(1)).float()               # [B, L]
        # 4) 只对有效位置做损失
        masked_loss = ce_loss_2d * mask  # [B, L]
        total_valid = mask.sum()         # 有效位置总数

        # 5) 按有效位置取平均 (也可取 sum 或其他聚合方式)
        final_loss = masked_loss.sum() / (total_valid + 1e-8)
        return final_loss


def get_loss_set(weight_factor):
    fun_loss=FunctionMaskedCrossEntropyLoss(ignore_index=C.INTERPRO_PAD_TOKEN)
    seq_loss=SeqMaskedCrossEntropyLoss_repeat_penalty(ignore_index=C.SEQUENCE_PAD_TOKEN,weight_factor=weight_factor)
    str_loss = SeqAndStrMaskedCrossEntropyLoss(ignore_index=C.STRUCTURE_PAD_TOKEN,weight_factor=weight_factor)
    ss_loss = SeqAndStrMaskedCrossEntropyLoss(ignore_index=C.SS8_PAD_TOKEN,weight_factor=weight_factor)
    sa_loss = SeqAndStrMaskedCrossEntropyLoss(ignore_index=C.SASA_PAD_TOKEN,weight_factor=weight_factor)
    return fun_loss, seq_loss, str_loss, ss_loss, sa_loss

if __name__ == '__main__':
    batch_size = 2
    seq_len = 5
    num_class = 260
    num_categories = 8

    # pred: batch, seq_len, 8, num_class
    pred = torch.randn(batch_size, seq_len, num_categories, num_class)

    # lab: batch, seq_len, 8
    lab = torch.randint(0, num_class, (batch_size, seq_len, num_categories))

    # mask_position: batch, seq_len
    mask_position = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=torch.float32)

    # 初始化损失函数
    criterion = FunctionMaskedCrossEntropyLoss()

    # 计算加权损失
    loss = criterion(pred, lab, mask_position)

    # 打印结果
    print("Pred shape:", pred.shape)
    print("Lab shape:", lab.shape)
    print("Mask position:\n", mask_position)
    print("Weighted loss:", loss.item())

