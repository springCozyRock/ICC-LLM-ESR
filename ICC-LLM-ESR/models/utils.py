# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import pickle

# 新增：聚类信息处理类
class ClusterHandler(nn.Module):
    """处理聚类信息的工具类，用于获取聚类中心和计算聚类损失"""
    def __init__(self, dataset, hidden_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.load_cluster_info(dataset)
        
        # 定义聚类中心映射层（8维→d维）
        self.cluster_projection = nn.Linear(8, hidden_size)
        # 冻结映射层参数
        for param in self.cluster_projection.parameters():
            param.requires_grad = False

    def load_cluster_info(self, dataset):
        """加载聚类标签和中心，并转换为张量"""
        data_dir = f'data/{dataset}/handled/'
        # 读取item聚类标签（含噪声）
        with open(f'{data_dir}/item_cluster_labels.pkl', 'rb') as f:
            self.item_cluster_labels = torch.tensor(pickle.load(f), dtype=torch.long, device=self.device)
        # 读取8维聚类中心
        with open(f'{data_dir}/cluster_centers_8d.pkl', 'rb') as f:
            cluster_centers_8d = pickle.load(f)
        # 将中心数据转换为PyTorch张量
        num_clusters = len(cluster_centers_8d)
        self.cluster_centers_8d = torch.zeros(num_clusters, 8, device=self.device)
        for c_id, center in cluster_centers_8d.items():
            self.cluster_centers_8d[c_id] = torch.tensor(center, dtype=torch.float32, device=self.device)

    def get_cluster_center(self, item_ids):
        """获取item对应的聚类中心（映射到d维）"""
        # 获取聚类标签
        cluster_labels = self.item_cluster_labels[item_ids]
        # 提取非噪声item的8维中心
        non_noise_mask = cluster_labels != -1 #过滤噪声物品（标签为-1）
        non_noise_indices = torch.where(non_noise_mask)[0]
        non_noise_labels = cluster_labels[non_noise_indices]
        # 仅对非噪声物品：获取8维中心，并映射到d维
        cluster_centers_8d = self.cluster_centers_8d[non_noise_labels]
        cluster_centers_d = self.cluster_projection(cluster_centers_8d)
        # 创建全零张量（与协作嵌入维度一致）
        batch_size = item_ids.size(0)
        centers_d = torch.zeros(batch_size, self.hidden_size, device=item_ids.device)
        # 将非噪声item的中心填入结果，为噪声物品返回零向量
        centers_d[non_noise_indices] = cluster_centers_d
        return centers_d, non_noise_mask

    def calculate_cluster_loss(self, item_ids, item_embeddings):
        """计算非噪声item的聚类约束损失"""
        # 空输入处理
        if item_ids.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 获取item对应的聚类中心（d维）和非噪声掩码
        centers_d, non_noise_mask = self.get_cluster_center(item_ids)
        
        # 仅计算非噪声item的损失
        if non_noise_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 计算非噪声item的L2损失
        non_noise_embeddings = item_embeddings[non_noise_mask]
        non_noise_centers = centers_d[non_noise_mask]
        cluster_loss = torch.mean(torch.norm(non_noise_embeddings - non_noise_centers, p=2, dim=1)**2)
        
        return cluster_loss

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    


class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    


class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention



class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output



