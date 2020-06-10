import math
import torch
import copy
import torch.nn as nn

class Encoder(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, h, length, d_model, dropout):
        super(Encoder, self).__init__()
        self.multi_headed_attention = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_model * 4)
        self.sublayer = clones(SublayerConnection(length, d_model, dropout), 2)

    def forward(self, x, y, mask=None):
        "Follow Figure 1 (left) for connections."
        y = self.sublayer[0](y, lambda y: self.multi_headed_attention(x, y, y, mask))
        return self.sublayer[1](y, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # (3 + 1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x = self.attention(query, key, value,causality=True, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, causality=True, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        if causality:
            scores = torch.tril(scores, diagonal=0, out=None)
        p_attn = self.softmax(scores)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, length, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape = [length, d_model])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])