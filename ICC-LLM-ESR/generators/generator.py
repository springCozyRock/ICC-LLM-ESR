# here put the import lib
import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from generators.data import SeqDataset, SeqDatasetAllUser, Seq2SeqDatasetAllUser
from utils.utils import unzip_data, concat_data

# 用于序列推荐任务的数据生成器类 Generator 及其子类，主要用于处理用户交互序列数据并生成 PyTorch DataLoader
class Generator(object):

    def __init__(self, args, logger, device):
        # 初始化参数
        self.args = args

        self.aug_file = args.aug_file
        self.inter_file = args.inter_file
        self.dataset = args.dataset
        # 文件路径相关参数
        self.num_workers = args.num_workers 
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device
        self.aug_seq = args.aug_seq

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    
    def _load_dataset(self):
        '''Load train, validation, test dataset'''

        usernum = 0
        itemnum = 0
        User = defaultdict(list)    # default value is a blank list
        user_train = {}
        user_valid = {}
        user_test = {}
        # assume user/item index starting from 1
        f = open('./data/%s/handled/%s.txt' % (self.dataset, self.inter_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        
        self.user_num = usernum
        self.item_num = itemnum

        for user in tqdm(User):
            nfeedback = len(User[user]) - self.args.aug_seq_len
            #nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        
        self.train = user_train
        self.valid = user_valid
        self.test = user_test


    # 生成训练数据加载器 
    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = SeqDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    

        return train_dataloader

    # 生成验证数据加载器
    def make_evalloader(self, test=False):

        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])

        else:
            eval_dataset = concat_data([self.train, self.valid])

        self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg)
        eval_dataloader = DataLoader(self.eval_dataset,
                                    sampler=SequentialSampler(self.eval_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader

    
    def get_user_item_num(self):

        return self.user_num, self.item_num
    
    # 获取物品流行度
    # 该方法用于获取物品流行度，返回一个numpy数组，数组中每个元素表示对应物品的流行度。
    # 它首先调用concat_data函数对训练数据、验证数据和测试数据进行合并，然后创建一个流行度数组，最后返回该数组。
    # 参数：
    # - self: 当前对象的引用
    # - self.train: 训练数据
    # - self.valid: 验证数据
    # - self.test: 测试数据
    # 返回值：
    # - pop: 物品流行度数组
    # 
    def get_item_pop(self):
        """get item popularity according to item index. return a np-array"""
        all_data = concat_data([self.train, self.valid, self.test])
        pop = np.zeros(self.item_num+1) # item index starts from 0
        
        for items in all_data:
            pop[items] += 1

        return pop
    

    def get_user_len(self):
        """get sequence length according to user index. return a np-array"""
        all_data = concat_data([self.train, self.valid])
        lens = []

        for user in all_data:
            lens.append(len(user))

        return np.array(lens)


# 用于序列推荐任务的数据生成器类 GeneratorAllUser 及其子类，主要用于处理用户交互序列数据并生成 PyTorch DataLoader
class GeneratorAllUser(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = SeqDatasetAllUser(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader

    
# 用于序列推荐任务的数据生成器类 Seq2SeqGeneratorAllUser 及其子类，主要用于处理用户交互序列数据并生成 PyTorch DataLoader
class Seq2SeqGeneratorAllUser(Generator):
    
    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = Seq2SeqDatasetAllUser(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader
    

    