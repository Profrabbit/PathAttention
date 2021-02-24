import pickle
import os
import json

PAD, UNK, EOS, SOS = '<PAD>', '<UNK>', '<EOS>', '<SOS>'


class UniTextVocab(object):
    def __init__(self, args):
        '''
        :param args: json dict:{str:num}
        self.vocab: {str:idx}
        :param type: source or target
        '''

        self.args = args
        self.vocab = dict()
        self.type = 'uni'
        self.__special__()
        print('Get Uni Vocab')
        self.dataset_dir = os.path.join('./data', args.dataset, 'data')
        with open(os.path.join(self.dataset_dir, 'source' + '_vocab.json'), 'r') as f:
            source_vocab_dict = json.load(f)
        with open(os.path.join(self.dataset_dir, 'target' + '_vocab.json'), 'r') as f:
            target_vocab_dict = json.load(f)
        all_vocab_dict = dict()
        for key, value in source_vocab_dict.items():
            if key not in all_vocab_dict:
                all_vocab_dict[key] = value
            else:
                all_vocab_dict[key] += value

        for key, value in target_vocab_dict.items():
            if key not in all_vocab_dict:
                all_vocab_dict[key] = value
            else:
                all_vocab_dict[key] += value

        ordered_list = sorted(all_vocab_dict.items(), key=lambda item: item[1], reverse=True)

        for key, value in ordered_list:
            if value < self.args.vocab_threshold:
                break
            self.vocab[key] = len(self.vocab)
        print('{} Vocab length == {}'.format('Uni', len(self.vocab)))
        self.re_vocab = dict()
        for key, value in self.vocab.items():
            self.re_vocab[value] = key

    def find(self, sub_token):
        return self.vocab.get(sub_token, self.unk_index)

    def has_key(self, key):
        if key in self.vocab:
            return True
        else:
            return False

    def __special__(self):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.vocab[PAD] = self.pad_index
        self.vocab[UNK] = self.unk_index
        self.vocab[EOS] = self.eos_index
        self.vocab[SOS] = self.sos_index
        self.special_index = [self.pad_index, self.eos_index, self.sos_index, self.unk_index]

    def re_find(self, idx):
        return self.re_vocab.get(idx, UNK)

    def __len__(self):
        return len(self.vocab)


class TextVocab(object):
    def __init__(self, args, type):
        '''
        :param args: json dict:{str:num}
        self.vocab: {str:idx}
        :param type: source or target
        '''

        self.args = args
        self.vocab = dict()
        self.__special__()
        assert type in ['source', 'target']
        print('Get {} Vocab'.format(type))
        self.dataset_dir = os.path.join('./data', args.dataset, 'data')
        with open(os.path.join(self.dataset_dir, type + '_vocab.json'), 'r') as f:
            vocab_dict = json.load(f)
        self.sum_tokens = 0
        for key, value in vocab_dict.items():
            self.sum_tokens += value
        ordered_list = sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)
        temp_sum = 0
        self.type = type
        if self.type == 'source':
            self.vocab_portion = self.args.s_vocab_portion
        else:
            self.vocab_portion = self.args.t_vocab_portion
        for key, value in ordered_list:
            temp_sum += value
            if temp_sum / self.sum_tokens > self.vocab_portion:
                break
            self.vocab[key] = len(self.vocab)
        print('{} Vocab length == {}'.format(type, len(self.vocab)))
        self.re_vocab = dict()
        for key, value in self.vocab.items():
            self.re_vocab[value] = key

    def find(self, sub_token):
        return self.vocab.get(sub_token, self.unk_index)

    def has_key(self, key):
        if key in self.vocab:
            return True
        else:
            return False

    def __special__(self):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.vocab[PAD] = self.pad_index
        self.vocab[UNK] = self.unk_index
        self.vocab[EOS] = self.eos_index
        self.vocab[SOS] = self.sos_index
        self.special_index = [self.pad_index, self.eos_index, self.sos_index, self.unk_index]

    def re_find(self, idx):
        return self.re_vocab.get(idx, UNK)

    def __len__(self):
        return len(self.vocab)
