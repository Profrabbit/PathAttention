import torch

# attn_mask = torch.tril(torch.ones((8, 8))) == 0
# # print(attn_mask)
# inputs = torch.ones((8, 2, 6))
# mha = torch.nn.MultiheadAttention(6, 2)  # hidden_dim=6, head_num=2
#
# key_padding_mask = torch.tensor([[1] * 7 + [0]] * 2) == 0
# print(key_padding_mask)
#
# outputs, weights = mha(inputs, inputs, inputs, key_padding_mask=key_padding_mask)  # Q, K, V, attn_mask for causality
# print(outputs)
# print(weights)

a = torch.zeros(2, 3, 3, 2)
b = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]]).unsqueeze(0).repeat(2, 1, 1)  # 2,2,4 bs,n,dim
c = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 1, 1]]).unsqueeze(0).repeat(2, 1, 1)  # 2,3,3 bs,a,b

# 2   3, 2 ,4
# 2   3 ,3, 4

output = torch.gather(b.unsqueeze(1).expand(-1, 3, -1, -1), 2, c.unsqueeze(-1).expand(-1, -1, -1, 4))
# print(output)

w = torch.rand(5, 8)
index = torch.randint(0, 5, size=(10, 20))  # want 10,20,8
# print(w)
t = torch.gather(w.expand(10, -1, -1), 1, index.unsqueeze(2).expand(-1, -1, 8))

# 5，8 ->  10,5,8
# 10,20 -> 10,20,8
# print(t.shape)
targets = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
smoothing = 0.2
_targets = torch.empty(size=(targets.size(0), 3),
                       device=targets.device) \
    .fill_(smoothing / (3 - 1)) \
    .scatter_(1, targets.data, 1. - smoothing)
print(_targets)


def label_smoothing(data):
    K = data.shape[-1]
    return (data * (1 - 0.2) + 0.2 / K).detach()


print(label_smoothing(targets))
