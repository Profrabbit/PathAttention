from torch import nn
from .embedding import LeftEmbedding, RightEmbedding, PathEmbedding
from .encoder import Encoder
import torch


class Model(nn.Module):
    def __init__(self, args, s_vocab, t_vocab):
        super().__init__()
        self.left_embedding = LeftEmbedding(args, s_vocab)
        self.right_embedding = RightEmbedding(args, t_vocab)
        self.path_embedding = PathEmbedding(args)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=args.hidden, nhead=args.attn_heads,
                                                        dim_feedforward=args.ff_fold * args.hidden,
                                                        dropout=args.dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=args.decoder_layers)
        self.encoder = Encoder(args)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.relation = args.relation
        self.args = args
        if self.args.uni_vocab:
            self.right_embedding.embedding.weight = self.left_embedding.embedding.weight
            if args.embedding_size != args.hidden:
                self.right_embedding.in_.weight = self.left_embedding.in_.weight
        if self.args.weight_tying:
            # assert not self.args.pretrain
            self.right_embedding.out.weight = self.right_embedding.embedding.weight

    def encode(self, data):
        content = data['content']
        content_mask = data['content_mask']
        path_map = data['path_map']
        paths = data['paths']
        paths_mask = data['paths_mask']
        named = data['named']
        content_ = self.left_embedding(content, named)
        # bs, max_code_length, hidden

        if self.relation:
            paths_ = self.path_embedding(paths, paths_mask)
        # bs,(max_path_num+1),hidden
        else:
            paths_ = None

        # mask = torch.count_nonzero(content, dim=-1)  # 1.7.0
        # bs, max_code_length,sub_token_length -> bs, max_code_length

        mask_ = (content_mask > 0).unsqueeze(1).repeat(1, content_mask.size(1), 1).unsqueeze(1)
        # bs, 1,max_code_length,max_code_length

        memory = self.encoder(content_, mask_, paths_, path_map)
        # bs, max_code_length, hidden
        return memory, (content_mask == 0)

    def decode(self, memory, f_source, memory_key_padding_mask):
        '''

        :param tgt_key_padding_mask:
        :param memory_key_padding_mask:
        :param memory: # bs, max_code_length, hidden
        :param f_source: # bs,max_target_len
        :return:
        '''
        f_source_ = self.right_embedding(f_source)
        # bs,max_target_len,hidden

        f_len = f_source.shape[-1]
        tgt_mask = (torch.ones(f_len, f_len).tril_() == 0).to(memory.device)
        memory_key_padding_mask = memory_key_padding_mask.to(memory.device)
        tgt_key_padding_mask = (f_source == 0).to(memory.device)
        feature = self.decoder(f_source_.permute(1, 0, 2), memory.permute(1, 0, 2), tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        # tgt_mask：用作构建三角矩阵 L*L
        # memory_mask：len(tgt)*len(mem) 通常应该不会使用
        # tgt_key_padding_mask：bs*len(tgt) 用作tgt端的self attn的padding 但是因为padding的loss不会计算 所以应该也不用使用 保险一点还是用上吧
        # memory_key_padding_mask：用作tgt和mem之间attn时 attn传入的key padding mask 用来确定哪些key在被attn时被忽略 bs*len(mem)
        # True代表会被mask
        feature = feature.permute(1, 0, 2)
        # bs,max_target_len,hidden

        out = self.softmax(self.right_embedding.prob(feature))
        # bs,max_target_len,vocab_size
        return out

    def forward(self, data):
        f_source = data['f_source']
        memory, memory_key_padding_mask = self.encode(data)
        out = self.decode(memory, f_source, memory_key_padding_mask)
        return out
