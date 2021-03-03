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
    #  (max_path_num*2) * max_path_length 其中这里边是包含先后顺序的
    #  max_code_length  for transformer mask  -> max_code_length*max_code_length
    #  max_path_num for rnn_mask
    #  max_target_len  表示方法名  计算loss和acc都是足够的
    # 同时也需要考虑batch
    '''

    def __init__(self, args, s_vocab, t_vocab, type):
        self.on_memory = args.on_memory
        self.dataset_dir = os.path.join('./data', args.dataset)
        self.s_vocab = s_vocab
        self.t_vocab = t_vocab
        self.args = args
        self.type = type
        assert type in ['train', 'test', 'valid']
        if self.on_memory:
            self.json_path = os.path.join(self.dataset_dir, type + '.json')
            with open(self.json_path, 'r') as f:
                self.data = f.readlines()
            self.corpus_line = len(self.data)
        else:
            self.json_path = os.path.join(self.dataset_dir, type + '.json')
            self.corpus_line = 0
            with open(self.json_path, 'r') as f:
                for _ in f:
                    self.corpus_line += 1
            self.file = open(self.json_path, 'r')
        if self.args.tiny_data > 0:
            self.corpus_line = self.args.tiny_data

    def __len__(self):
        return self.corpus_line

    def __getitem__(self, item):
        assert item < self.corpus_line
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
        '''

        def decoder_process(data):
            target = data['target']
            f_ = [self.t_vocab.find(sub_token) for sub_token in target]
            f_source = [self.t_vocab.sos_index] + f_
            f_source = f_source[:self.args.max_target_len] + [self.t_vocab.pad_index] * abs(
                self.args.max_target_len - len(f_source))
            # f_source: max_target_len
            f_target = f_ + [self.t_vocab.eos_index]
            f_target = f_target[:self.args.max_target_len] + [self.t_vocab.pad_index] * abs(
                self.args.max_target_len - len(f_target))
            # f_target: max_target_len

            return f_source, f_target

        def content_process(data):
            content = data['content']

            content_ = [self.s_vocab.find(token) for token in content][:self.args.max_code_length] + [
                self.s_vocab.pad_index] * abs(self.args.max_code_length - len(content))
            # content_: max_code_length
            content_mask_ = [1 for _ in content][:self.args.max_code_length] + [0] * abs(
                self.args.max_code_length - len(content))
            # content_mask_: max_code_length
            # --------------------------
            # for tokens in content:
            #     l_ = len(tokens)
            #     list_ = []
            #     for token in tokens:
            #         list_.append(self.s_vocab.find(token))
            #     list_ = list_[:self.args.sub_token_length] + [self.s_vocab.pad_index] * abs(
            #         self.args.sub_token_length - l_)
            #     content_.append(list_)
            # content_ = content_[:self.args.max_code_length] + [
            #     [self.s_vocab.pad_index] * self.args.sub_token_length] * abs(
            #     self.args.max_code_length - len(content_))
            # content_:max_code_length*sub_token_length
            # content_mask_ = [min(len(tokens), self.args.sub_token_length) * [1] + abs(
            #     self.args.sub_token_length - len(tokens)) * [0] for tokens in content][
            #                 :self.args.max_code_length] + [[0] * self.args.sub_token_length] * abs(
            #     self.args.max_code_length - len(content))
            # content_mask_: max_code_length*sub_token_length
            return content_, content_mask_

        def path_process(data):
            paths = data['paths']  # [[,,,,],[,,,,,]]
            paths_map = data['paths_map']  # [[l,r],idx] => {idx:[l,r,l,r]}

            # 1) use max_path_num to filter paths
            paths = paths[:self.args.max_path_num]
            # 2) use filtered paths and max_code_length to filter paths_map
            # TODO add random select path
            paths_map_ = [[self.args.max_path_num for _ in range(self.args.max_code_length)] for _ in
                          range(self.args.max_code_length)]
            paths_map_ = torch.tensor(paths_map_)

            for key, value in paths_map.items():
                assert len(value) % 2 == 0
                if int(key) >= self.args.max_path_num: continue
                for i in range(0, len(value), 2):
                    l, r = value[i], value[i + 1]
                    if l >= self.args.max_code_length or r >= self.args.max_code_length:
                        continue
                    paths_map_[l, r] = int(key)

            paths_mask_ = []
            paths_ = []
            for path in paths:
                paths_.append(
                    path[:self.args.max_path_length] + [self.args.path_embedding_num] * abs(
                        self.args.max_path_length - len(path)))  # use path node num as padding idx of path
                paths_mask_.append(len(path) if len(path) < self.args.max_path_length else self.args.max_path_length)

            assert len(paths_) <= self.args.max_path_num
            assert len(paths_mask_) <= self.args.max_path_num
            paths_ = paths_ + [[self.args.path_embedding_num] * self.args.max_path_length] * (
                    self.args.max_path_num - len(paths_))
            paths_mask_ = paths_mask_ + [1] * (
                    self.args.max_path_num - len(paths_mask_))  # 1 not 0 for pack pad, should avoid noise
            return paths_map_, paths_, paths_mask_

        f_source, f_target = decoder_process(data)
        content_, content_mask_ = content_process(data)
        paths_map_, paths_, paths_mask_ = path_process(data)
        return {'f_source': f_source, 'f_target': f_target, 'content': content_, 'content_mask': content_mask_,
                'path_map': paths_map_, 'paths': paths_, 'paths_mask': paths_mask_}

    def get_corpus_line(self, item):
        if self.on_memory:
            data = json.loads(self.data[item])
            return data
        else:
            if item == 0:
                self.file.close()
                self.file = open(self.json_path, 'r')
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.json_path, 'r')
                line = self.file.__next__()
            data = json.loads(line)
            return data
