import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from esm.utils.constants import esm3 as C


class SeqMaskedCrossEntropyLoss_repeat_penalty(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=C.SEQUENCE_PAD_TOKEN, weight_factor=1.3, 
                 repetition_threshold=5, 
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

    def forward(self, preds, targets, mask_position):
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

        
        # 检测重复并构建惩罚矩阵
        repetition_penalty = self.detect_repetitions(preds, mask_position)
        
        # 将重复惩罚应用到权重矩阵
        weight_matrix = weight_matrix * repetition_penalty
        
        # Calculate the base cross-entropy loss
        loss = super().forward(preds_permuted, targets)  # Apply nn.CrossEntropyLoss with permuted preds
        
        # Apply the weight matrix to the loss
        loss = (loss * weight_matrix).sum()

        # Sum the weighted loss and return
        return loss / ((weight_matrix > 0).sum())


# 假设 IterativeRPO 类已定义
class DPO_Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def get_log_prob(self, logits, target_ids, mask=None):
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            gathered *= mask
        return gathered.sum(-1)  # 关键修改：sum代替mean

    def forward(self, policy_logits, ref_logits, winner_ids, loser_ids, mask=None):
        ref_logits = ref_logits.detach()

        # 计算对数概率（已改为求和）
        policy_winner = self.get_log_prob(policy_logits, winner_ids, mask)
        policy_loser = self.get_log_prob(policy_logits, loser_ids, mask)
        ref_winner = self.get_log_prob(ref_logits, winner_ids, mask)
        ref_loser = self.get_log_prob(ref_logits, loser_ids, mask)

        # DPO损失（增加批次维度归一化）
        log_ratio_winner = policy_winner - ref_winner
        log_ratio_loser = policy_loser - ref_loser
        dpo_loss = -F.logsigmoid(self.beta * (log_ratio_winner - log_ratio_loser)).sum()
        return dpo_loss
