# here put the import lib
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    """设置所有随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # only add when conv in your model


def get_n_params(model):
    '''Get the number of parameters of model'''
    """计算模型参数量 - 用于模型复杂度分析"""
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_n_params_(parameter_list):
    '''Get the number of parameters of model'''
    """计算指定参数列表的参数量 - 用于分析模块复杂度"""
    pp = 0
    for p in list(parameter_list):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def unzip_data(data, aug=True, aug_num=0):

    res = []
    """
    序列数据增强处理 - 生成增强序列
    :param data: 原始用户序列 {user: [item1, item2, ...]}
    :param aug: 是否进行增强
    :param aug_num: 增强次数
    :return: 增强后的序列列表
    """
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)
            # 为每个用户生成多个子序列
            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i]) # 截取前缀序列
    else:
        # 直接返回完整序列
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)

    return res


def unzip_data_with_user(data, aug=True, aug_num=0):
    """
    带用户ID的序列增强 - 用于自蒸馏中用户标识
    返回格式: (序列列表, 用户ID列表)
    """

    # 实现类似unzip_data，但额外返回用户ID
    # 这对论文中的检索增强自蒸馏模块至关重要

    res = []
    users = []
    user_id = 1
    
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
                users.append(user_id)

            user_id += 1

    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)
            users.append(user_id)
            user_id += 1

    return res, users


def concat_data(data_list):

    res = []

    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
    
    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])

    else:

        raise ValueError

    return res


def concat_aug_data(data_list):

    res = []

    train = data_list[0]
    valid = data_list[1]

    for user in train:

        if len(valid[user]) == 0:
            res.append([train[user][0]])
        
        else:
            res.append(train[user]+valid[user])

    return res


def concat_data_with_user(data_list):

    res = []
    users = []
    user_id = 1

    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
            users.append(user_id)
            user_id += 1
    
    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])
            users.append(user_id)
            user_id += 1

    else:

        raise ValueError

    return res, users


def filter_data(data, thershold=5):
    '''Filter out the sequence shorter than threshold'''
    res = []

    for user in data:

        if len(user) > thershold:
            res.append(user)
        else:
            continue
    
    return res



def random_neq(l, r, s=[]):    # 在l-r之间随机采样一个数，这个数不能在列表s中
    """在[l, r)范围内随机采样不在列表s中的数 - 负采样核心函数"""
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t



def metric_report(data_rank, topk=10):
    """
    基础评估指标计算 - 计算整体NDCG@10和HR@10
    :param data_rank: 正样本在推荐列表中的排名数组
    :return: 指标字典
    """
    NDCG, HT = 0, 0
    
    for rank in data_rank: # rank是正样本在推荐列表中的位置

        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return {'NDCG@10': NDCG / len(data_rank),
            'HR@10': HT / len(data_rank)}



# def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):

#     if args is not None:
#         ts_short = args.ts_short
#         ts_long = args.ts_long
#     else:
#         ts_short = 10
#         ts_long = 20

#     NDCG_s, HT_s = 0, 0
#     NDCG_m, HT_m = 0, 0
#     NDCG_l, HT_l = 0, 0
#     count_s = len(data_len[data_len<ts_short+aug_len])
#     count_l = len(data_len[data_len>=ts_long+aug_len])
#     count_m = len(data_len) - count_s - count_l

#     for i, rank in enumerate(data_rank):

#         if rank < topk:

#             if data_len[i] < ts_short+aug_len:
#                 NDCG_s += 1 / np.log2(rank + 2)
#                 HT_s += 1
#             elif data_len[i] < ts_long+aug_len:
#                 NDCG_m += 1 / np.log2(rank + 2)
#                 HT_m += 1
#             else:
#                 NDCG_l += 1 / np.log2(rank + 2)
#                 HT_l += 1

#     return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0, # avoid division of 0
#             'Short HR@10': HT_s / count_s if count_s!=0 else 0,
#             'Medium NDCG@10': NDCG_m / count_m if count_m!=0 else 0,
#             'Medium HR@10': HT_m / count_m if count_m!=0 else 0,
#             'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
#             'Long HR@10': HT_l / count_l if count_l!=0 else 0,}


def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):
    """
    按用户序列长度分组评估 - 核心长尾用户评估
    :param data_rank: 正样本排名数组
    :param data_len: 对应的用户序列长度数组
    :param aug_len: 数据增强增加的长度
    :return: 分组指标字典
    
    对应论文中的表1的"Tail User"和"Head User"列
    """
    # 设置长度阈值（短序列用户和长序列用户）
    if args is not None:
        ts_user = args.ts_user
    else:
        ts_user = 10
    
    # 初始化计数器和指标
    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    count_s = len(data_len[data_len<ts_user+aug_len]) # 短序列用户数
    count_l = len(data_len[data_len>=ts_user+aug_len]) # 长序列用户数

    for i, rank in enumerate(data_rank):

        if rank < topk:

            if data_len[i] < ts_user+aug_len:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            else:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0, # avoid division of 0
            'Short HR@10': HT_s / count_s if count_s!=0 else 0,
            'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Long HR@10': HT_l / count_l if count_l!=0 else 0,}


def metric_pop_report(data_rank, pop_dict, target_items, topk=10, aug_pop=0, args=None):
    """
    Report the metrics according to target item's popularity
    item_pop: the array of the target item's popularity
    """

    """
    按物品流行度分组评估 - 核心长尾物品评估
    :param pop_dict: 物品ID到流行度的映射字典
    :param target_items: 目标物品ID数组
    :param aug_pop: 数据增强调整值
    :return: 分组指标字典
    
    对应论文中的表1的"Tail Item"和"Head Item"列
    """

    if args is not None:
        ts_tail = args.ts_item
    else:
        ts_tail = 20

    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    item_pop = pop_dict[target_items.astype("int64")]
    count_s = len(item_pop[item_pop<ts_tail+aug_pop])
    count_l = len(item_pop[item_pop>=ts_tail+aug_pop])

    for i, rank in enumerate(data_rank):

        if i == 0:  # skip the padding index
            continue

        if rank < topk:

            if item_pop[i] < ts_tail+aug_pop:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            else:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Tail NDCG@10': NDCG_s / count_s if count_s!=0 else 0,
            'Tail HR@10': HT_s / count_s if count_s!=0 else 0,
            'Popular NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Popular HR@10': HT_l / count_l if count_l!=0 else 0,}



def metric_len_5group(pred_rank, 
                      seq_len, 
                      thresholds=[5, 10, 15, 20], 
                      topk=10):
    """
    五分组用户长度评估 - 用于论文图4的精细分析
    :param thresholds: 分组阈值
    :return: (HR数组, NDCG数组, 各组数量)
    """
    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    for i, rank in enumerate(pred_rank):

        target_len = seq_len[i]
        if rank < topk:

            if target_len < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_len < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_len < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            elif target_len < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    count[0] = len(seq_len[seq_len>=0]) - len(seq_len[seq_len>=thresholds[0]])
    count[1] = len(seq_len[seq_len>=thresholds[0]]) - len(seq_len[seq_len>=thresholds[1]])
    count[2] = len(seq_len[seq_len>=thresholds[1]]) - len(seq_len[seq_len>=thresholds[2]])
    count[3] = len(seq_len[seq_len>=thresholds[2]]) - len(seq_len[seq_len>=thresholds[3]])
    count[4] = len(seq_len[seq_len>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count



def metric_pop_5group(pred_rank, 
                      pop_dict, 
                      target_items, 
                      thresholds=[10, 30, 60, 100], 
                      topk=10):

    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    for i, rank in enumerate(pred_rank):

        target_pop = pop_dict[int(target_items[i])]
        if rank < topk:

            if target_pop < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_pop < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_pop < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            elif target_pop < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    pop = pop_dict[target_items.astype("int64")]
    count[0] = len(pop[pop>=0]) - len(pop[pop>=thresholds[0]])
    count[1] = len(pop[pop>=thresholds[0]]) - len(pop[pop>=thresholds[1]])
    count[2] = len(pop[pop>=thresholds[1]]) - len(pop[pop>=thresholds[2]])
    count[3] = len(pop[pop>=thresholds[2]]) - len(pop[pop>=thresholds[3]])
    count[4] = len(pop[pop>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count



def seq_acc(true, pred):

    true_num = np.sum((true==pred))
    total_num = true.shape[0] * true.shape[1]

    return {'acc': true_num / total_num}


def load_pretrained_model(pretrain_dir, model, logger, device):

    logger.info("Loading pretrained model ... ")
    checkpoint_path = os.path.join(pretrain_dir, 'pytorch_model.bin')

    model_dict = model.state_dict()

    # To be compatible with the new and old version of model saver
    try:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    except:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)

    # filter out required parameters
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    # 打印出来，更新了多少的参数
    logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    model.load_state_dict(model_dict)

    return model


def record_csv(args, res_dict, path='log'):
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    res_dict["model_name"] = model_name
    # columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)



def record_group(args, res_dict, path='log'):
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    res_dict["model_name"] = model_name
    # columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)
