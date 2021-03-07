import torch
import numpy as np

line = [1, 2, 3]
a = torch.tensor(np.array(line[1:]).astype(np.float))
b = torch.tensor(np.append(np.array(line[1:]), np.array([0] * 2)).astype(np.float))
print(b)
all_vocab_dict = {'a': 1, 'b': 2, 'c': 5, 'd': 10}
ordered_list = sorted(all_vocab_dict.items(), key=lambda item: item[1], reverse=True)
print(ordered_list)
vocab = dict()
for key, value in ordered_list:
    if value < 5:
        break
    vocab[key] = len(vocab)
print(vocab)