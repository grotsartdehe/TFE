import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import argparse
import numpy as np

# classes utilitaires

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.HIDDEN_SIZE, args.HIDDEN_SIZE)
        self.linear_k = nn.Linear(args.HIDDEN_SIZE, args.HIDDEN_SIZE)
        self.linear_q = nn.Linear(args.HIDDEN_SIZE, args.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(args.HIDDEN_SIZE, args.HIDDEN_SIZE)

        self.dropout = nn.Dropout(args.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.MULTI_HEAD,
            int(self.args.HIDDEN_SIZE / self.args.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.MULTI_HEAD,
            int(self.args.HIDDEN_SIZE / self.args.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.MULTI_HEAD,
            int(self.args.HIDDEN_SIZE / self.args.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.HIDDEN_SIZE,
            mid_size=args.FF_SIZE,
            out_size=args.HIDDEN_SIZE,
            dropout_r=args.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)






class AttFlat(nn.Module):
    def __init__(self, args):
        super(AttFlat, self).__init__()
        self.args = args
        self.FLAT_GLIMPSES = 1

        self.mlp = MLP(
            in_size=args.HIDDEN_SIZE,
            mid_size=args.HIDDEN_SIZE,
            out_size=self.FLAT_GLIMPSES,
            dropout_r=args.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            args.HIDDEN_SIZE * self.FLAT_GLIMPSES,
            args.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted



class Block(nn.Module):
    def __init__(self, args):
        super(Block, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.DROPOUT_R)
        self.norm1 = LayerNorm(args.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(args.DROPOUT_R)
        self.norm2 = LayerNorm(args.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        y = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y

# -------------------------
# ---- Main Net Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        self.enc1 = nn.ModuleList([Block(args) for _ in range(args.LAYER)])
        self.enc2 = nn.ModuleList([Block(args) for _ in range(args.LAYER)])

        # Flatten to vector
        self.attflat_img = AttFlat(args)
        self.attflat_lang = AttFlat(args)

        # Classification layers
        self.proj_norm = LayerNorm(args.FLAT_OUT_SIZE)
        self.proj = nn.Linear(args.FLAT_OUT_SIZE, 2)


    def forward(self, x, y):

        # Transformer encoder
        for enc in self.enc1:
            y = enc(x, None)

        for enc in self.enc2:
            y = enc(y, None)

        # Flatten to vector
        x_flat = self.attflat_lang(
            x,
            None,
        )

        y_flat = self.attflat_img(
            y,
            None,
        )

        # Classification layers
        proj_feat = x_flat + y_flat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


args = argparse.Namespace()
args.LAYER = 4
args.HIDDEN_SIZE = 512
args.FLAT_OUT_SIZE = 512
args.FF_SIZE = 2048
args.MULTI_HEAD = 8
args.DROPOUT_R = 0.1



net = Net(args)
x = np.zeros((10, 50, 512), dtype=np.float32) # batch x N x Features
y = np.zeros((10, 20, 512), dtype=np.float32) # batch x N x Features
x = torch.from_numpy(x)
y = torch.from_numpy(y)
out = net(x,y)
print(out)
print(out.shape) # [10,2]

