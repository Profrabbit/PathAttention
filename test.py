import torch
f_len = 2
tgt_mask = torch.ones(f_len, f_len).tril_()
print(tgt_mask == 0)