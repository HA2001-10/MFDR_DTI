import torch
import torch.nn as nn
import torch.nn.functional as F


class DWLoss(nn.Module):
    def __init__(self, weight_loss, DEVICE):
        super(DWLoss, self).__init__()
        self.main_loss = nn.CrossEntropyLoss(weight=weight_loss)
        self.aux_loss = nn.CrossEntropyLoss(weight=weight_loss)
        self.DEVICE = DEVICE

        # 初始化可学习的权重参数，使用sigmoid约束在[0.1, 1]范围内
        self.weight_combined = nn.Parameter(torch.tensor(0.5))
        self.weight_fusion = nn.Parameter(torch.tensor(0.5))
        self.weight_bert = nn.Parameter(torch.tensor(0.5))

    def _sigmoid_scale(self, x):
        """将权重缩放至[0.1, 1]范围内"""
        return 0.9 * torch.sigmoid(x) + 0.1

    def forward(self, outputs, labels):
        # 解包模型输出
        final_pred, pred_combined, pred_fusion, pred_bert = outputs
        labels = labels.to(self.DEVICE)

        # 计算各损失
        loss_main = self.main_loss(final_pred, labels)
        loss_combined = self.aux_loss(pred_combined, labels)
        loss_fusion = self.aux_loss(pred_fusion, labels)
        loss_bert = self.aux_loss(pred_bert, labels)

        # 获取缩放后的权重
        w_combined = self._sigmoid_scale(self.weight_combined)
        w_fusion = self._sigmoid_scale(self.weight_fusion)
        w_bert = self._sigmoid_scale(self.weight_bert)

        # 计算总损失
        total_loss = loss_main + w_combined * loss_combined + w_fusion * loss_fusion + w_bert * loss_bert

        return total_loss