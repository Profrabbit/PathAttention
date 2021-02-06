import torch

attn_mask = torch.tril(torch.ones((8, 8))) == 0
# print(attn_mask)
inputs = torch.ones((8, 2, 6))
mha = torch.nn.MultiheadAttention(6, 2)  # hidden_dim=6, head_num=2

key_padding_mask = torch.tensor([[1] * 7 + [0]] * 2) == 0
print(key_padding_mask)

outputs, weights = mha(inputs, inputs, inputs, key_padding_mask=key_padding_mask)  # Q, K, V, attn_mask for causality
print(outputs)
print(weights)
