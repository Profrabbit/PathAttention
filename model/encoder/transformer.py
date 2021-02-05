import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, activation):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout,
                                                    activation=activation)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, content, paths, mask):
        '''

        :param content: bs, max_code_length, hidden
        :param paths: bs,max_code_length,max_code_length,hidden
        :param mask: bs, 1,max_code_length,max_code_length
        :return:
        '''
        x = self.input_sublayer(content, lambda _x: self.attention.forward(_x, _x, _x, paths, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class Encoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, args):
        super().__init__()
        self.hidden = args.hidden
        self.n_layers = args.layers
        self.attn_heads = args.attn_heads
        self.dropout = args.dropout
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.hidden * 4
        self.activation = args.activation

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, self.dropout, self.activation) for _ in
             range(self.n_layers)])

    def forward(self, content, paths, mask):
        '''

        :param content: bs, max_code_length, hidden
        :param paths: bs,max_code_length,max_code_length,hidden
        :param mask: bs, 1,max_code_length,max_code_length
        :return:
        '''
        for transformer in self.transformer_blocks:
            content = transformer(content, paths, mask)
        return content
