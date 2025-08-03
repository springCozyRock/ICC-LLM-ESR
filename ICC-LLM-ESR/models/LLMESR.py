# here put the import lib
import torch
import torch.nn as nn
from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.utils import Contrastive_Loss2, ClusterHandler  # 新增导入



class LLMESR_SASRec(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg
        self.gamma = args.gamma  # 聚类约束强度

        # 初始化聚类处理器
        self.cluster_handler = ClusterHandler(args.dataset, args.hidden_size, device)

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        # 计算聚类约束损失
        item_ids = seq[seq > 0] # 有效物品ID
        item_embeddings = self.id_item_emb(item_ids) # 获取物品嵌入
        cluster_loss = self.cluster_handler.calculate_cluster_loss(item_ids, item_embeddings)
        loss += self.gamma * cluster_loss

        loss += self.alpha * align_loss

        return loss
    


class LLMESR_GRU4Rec(DualLLMGRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg
        self.gamma = args.gamma  # 聚类约束强度超参数

        # 初始化聚类处理器
        self.cluster_handler = ClusterHandler(args.dataset, args.hidden_size, device)

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        # 计算聚类约束损失
        item_ids = seq[seq > 0]
        item_embeddings = self.id_item_emb(item_ids)
        cluster_loss = self.cluster_handler.calculate_cluster_loss(item_ids, item_embeddings)
        loss += self.gamma * cluster_loss

        loss += self.alpha * align_loss

        return loss



class LLMESR_Bert4Rec(DualLLMBert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg
        self.gamma = args.gamma  # 聚类约束强度超参数

        # 初始化聚类处理器
        self.cluster_handler = ClusterHandler(args.dataset, args.hidden_size, device)

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.reg = Contrastive_Loss2()


        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        # 计算聚类约束损失（适配Bert4Rec的特殊处理）
        cluster_loss = self.calculate_cluster_loss(seq)

        # 整合所有损失
        loss += self.gamma * cluster_loss  # 聚类约束损失
        loss += self.alpha * align_loss
        
        return loss
    
    # def calculate_cluster_loss(self, seq):
    #         """适配Bert4Rec的聚类损失计算（处理掩码和填充）"""
    #         # Bert4Rec的序列中可能包含掩码标记（如0或特殊值），需过滤有效item ID
    #         # 假设seq中>0的为有效item ID（与原代码保持一致）
    #         valid_mask = (seq > 0) & (seq != self.mask_token_id)  # 排除掩码标记
    #         item_ids = seq[valid_mask]

    #         if item_ids.numel() == 0:
    #             return torch.tensor(0.0, device=seq.device)
            
    #         # 获取Bert4Rec的协作嵌入（E_co，对应id_item_emb）
    #         item_embeddings = self.id_item_emb(item_ids)
            
    #         # 调用聚类处理器计算损失
    #         cluster_loss = self.cluster_handler.calculate_cluster_loss(item_ids, item_embeddings)
            
    #         return cluster_loss

    def calculate_cluster_loss(self, seq):
            #1
            valid_mask = (seq > 0) & (seq != self.mask_token)        # 过滤掉 PAD=0 和 MASK
            item_ids   = seq[valid_mask]                             # 1-D tensor
            if item_ids.numel() == 0:                                # 这一批全是 PAD/MASK
                return torch.tensor(0.0, device=seq.device)
            # ---------- 2) 再过滤聚类表里没有的 item——id ----------
            # 防止越界导致 device-side assert
            max_id_in_cluster = self.cluster_handler.item_cluster.size(0) - 1
            item_ids = item_ids[item_ids <= max_id_in_cluster]

            if item_ids.numel() == 0:                                # 都被过滤掉
                return torch.tensor(0.0, device=seq.device)
            # ---------- 3) 查嵌入并计算聚类损失 ----------
            item_embeddings = self.id_item_emb(item_ids)
            cluster_loss    = self.cluster_handler.calculate_cluster_loss(item_ids, item_embeddings)
            return cluster_loss


