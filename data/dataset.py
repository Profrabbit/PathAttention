from torch.utils.data import Dataset
import os
import pickle as pkl
from tqdm import tqdm
import json

import torch


class PathAttenDataset(Dataset):
    '''
    json file:
    {'content':[[],[],[]],'paths':{():[],():[]},'target':[,]}
    #  每一个样本的size应该是
    #  max_code_length * sub_token_length
    #  max_code_length * max_code_length 的矩阵 矩阵上每一位表示了边的次序  那么变得个数就应该是2倍的个数 这个部分可以在dataset处理
    #  max_path_num * max_path_length 其中这里边是包含先后顺序的
    #  max_code_length  for transformer mask  -> max_code_length*max_code_length
    #  max_path_num for rnn_mask
    #  max_target_len  表示方法名  计算loss和acc都是足够的
    # 同时也需要考虑batch
    '''

    def __init__(self, args, s_vocab, t_vocab, type):
        self.on_memory = args.on_memory
        self.dataset_dir = os.path.join('./data', args.dataset, 'data')
        self.s_vocab = s_vocab
        self.t_vocab = t_vocab
        self.args = args
        assert type in ['train', 'test', 'valid']
        if self.on_memory:
            self.pkl_path = os.path.join(self.dataset_dir, type + '.pkl')
            with open(self.pkl_path, 'rb') as f:
                self.data = pkl.load(f)
                self.corpus_line = len(self.data)
        else:
            self.json_path = os.path.join(self.dataset_dir, type + '.json')
            self.corpus_line = 0
            with open(self.json_path, 'r') as f:
                for _ in f:
                    self.corpus_line += 1
            self.file = open(self.json_path, 'r')

    def __len__(self):
        return self.corpus_line

    def __getitem__(self, item):
        data = self.get_corpus_line(item)
        sample = self.process(data)
        return {key: value if torch.is_tensor(value) else torch.tensor(value) for key, value in sample.items()}

    def process(self, data):

        def abs(length):
            return length if length >= 0 else 0

        '''
        
        :param data: 
        :return: 
        # f_source: max_target_len 
        # f_target: max_target_len
        # content_:max_code_length,sub_token_length
        # content_mask_: max_code_length,sub_token_length 用来合并content_中的sub token
        # path_map_: max_code_length,max_code_length 其中每一位表示了这个位置上的path idx
        # paths_:2*max_path_num,max_path_length  因为关系有前后关系
        # paths_mask_: 2*max_path_num
        # content_length_mask: max_code_length 每一位上表示该位置上的subtoken有多少个
        '''
        target = data['target']
        paths = data['paths']
        content = data['content']

        f_ = [self.t_vocab.find(sub_token) for sub_token in target]
        f_source = [self.t_vocab.sos_index] + f_
        f_source = f_source[:self.args.max_target_len] + [self.t_vocab.pad_index] * abs(
            self.args.max_target_len - len(f_source))
        # f_source: max_target_len
        f_target = f_ + [self.t_vocab.eos_index]
        f_target = f_target[:self.args.max_target_len] + [self.t_vocab.pad_index] * abs(
            self.args.max_target_len - len(f_target))
        # f_target: max_target_len
        content_ = []
        for tokens in content:
            l_ = len(tokens)
            list_ = []
            for token in tokens:
                list_.append(self.s_vocab.find(token))
            list_ = list_[:self.args.sub_token_length] + [self.s_vocab.pad_index] * abs(self.args.sub_token_length - l_)
            content_.append(list_)

        content_ = content_[:self.args.max_code_length] + [[self.s_vocab.pad_index] * self.args.sub_token_length] * abs(
            self.args.max_code_length - len(content_))
        # content_:max_code_length*sub_token_length
        content_mask_ = [min(len(tokens), self.args.sub_token_length) * [1] + abs(
            self.args.sub_token_length - len(tokens)) * [0] for tokens in content][
                        :self.args.max_code_length] + [[0] * self.args.sub_token_length] * abs(
            self.args.max_code_length - len(content))
        # content_mask_: max_code_length*sub_token_length

        paths_map_ = [[-1 for _ in range(self.args.max_code_length)] for _ in range(self.args.max_code_length)]
        paths_map_ = torch.tensor(paths_map_)

        temp_paths = []
        for key, value in paths:
            l, r = key
            if l >= self.args.max_code_length or r >= self.args.max_code_length:
                continue
            else:
                temp_paths.append([key, value])
        paths = temp_paths
        paths = paths[:self.args.max_path_num]
        paths_ = []
        paths_mask_ = []
        for idx, [key, value] in enumerate(paths):
            l, r = key
            paths_map_[l, r] = idx * 2
            paths_.append(
                value[:self.args.max_path_length] + [self.args.path_embedding_num] * abs(
                    self.args.max_path_length - len(value)))
            paths_mask_.append(len(value) if len(value) < self.args.max_path_length else self.args.max_path_length)
            paths_map_[r, l] = idx * 2 + 1
            r_value_ = list(reversed(value))
            paths_.append(
                r_value_[:self.args.max_path_length] + [self.args.path_embedding_num] * abs(
                    self.args.max_path_length - len(r_value_)))
            paths_mask_.append(
                len(r_value_) if len(r_value_) < self.args.max_path_length else self.args.max_path_length)
        assert len(paths_) <= self.args.max_path_num * 2
        assert len(paths_mask_) <= self.args.max_path_num * 2
        paths_ = paths_ + [[self.args.path_embedding_num] * self.args.max_path_length] * (
                self.args.max_path_num * 2 - len(paths_))
        paths_mask_ = paths_mask_ + [1] * (self.args.max_path_num * 2 - len(paths_mask_))  # 1 not 0 for pack pad

        return {'f_source': f_source, 'f_target': f_target, 'content': content_, 'content_mask': content_mask_,
                'path_map': paths_map_, 'paths': paths_, 'paths_mask': paths_mask_}

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.data[item]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.json_path, 'r')
                line = self.file.__next__()
            data = json.loads(line)
            return data
