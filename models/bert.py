import math
import copy
from operator import pos

import numpy as np
import torch
import transformers
from torch.nn import init
from transformers import DistilBertModel, AlbertModel
from torch import nn, device
from transformers.modeling_distilbert import TransformerBlock


def instantiate_transformer(transformer_name, vocab_size, output_hidden_states=False):
    if transformer_name == 'albert':
        transformer = AlbertModel.from_pretrained("albert-base-v2",
                                                  output_hidden_states=output_hidden_states)
    elif transformer_name == 'distilbert':
        transformer = DistilBertModel.from_pretrained("distilbert-base-uncased",
                                                      output_hidden_states=output_hidden_states)
    elif transformer_name == 'distilroberta':
        transformer = transformers.RobertaModel.from_pretrained('distilroberta-base', output_hidden_states=output_hidden_states)
    else:
        raise Exception('No such Transformer')
    transformer.resize_token_embeddings(vocab_size)
    return transformer


def get_neg_indexes(i):
    A = torch.arange(i).unsqueeze(0).repeat(i, 1).numpy()
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    A = strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)
    return A


def submult(c_g_h, r_g_h):
    return torch.cat([c_g_h, r_g_h, c_g_h - r_g_h, c_g_h * r_g_h, (c_g_h - r_g_h) * (c_g_h - r_g_h)],
                     -1)  # b(5e)


def submult_ablate(c_g_h, r_g_h):
    return torch.cat([c_g_h, r_g_h, c_g_h - r_g_h, c_g_h * r_g_h, (c_g_h - r_g_h) * (c_g_h - r_g_h)],
                     -1)  # b(5e)

class BiEncoder(nn.Module):
    def __init__(self, transformer_name, vocab_size, pad):
        super().__init__()
        self.bert1 = instantiate_transformer(transformer_name, vocab_size)
        self.PAD = pad

    def forward_transformer(self, c, r):
        c_attn_mask = ~c.eq(self.PAD)
        r_attn_mask = ~r.eq(self.PAD)

        c_x = self.bert1(c, attention_mask=c_attn_mask)
        r_x = self.bert1(r, attention_mask=r_attn_mask)

        return c_x, r_x


class BiEncoderDot(BiEncoder):
    def __init__(self, transformer_name, vocab_size, PAD):
        super().__init__(transformer_name, vocab_size, PAD)

    def forward(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0][:, 0, :]
        r_x = r_x[0][:, 0, :]

        o_x = torch.einsum('be,be->b', [c_x, r_x])
        return o_x

    def forward_with_negatives(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0][:, 0, :]
        r_x = r_x[0][:, 0, :]

        b_size = c_x.shape[0]
        A = get_neg_indexes(b_size)

        neg_candidates = r_x[A[:b_size, :b_size - 1].reshape(-1)].reshape(b_size, b_size - 1, -1)  # B(B-1)H

        o_x_p = torch.einsum('be,be->b', [c_x, r_x])
        o_x_n = torch.einsum('be,bde->bd', [c_x, neg_candidates])
        o_x = torch.cat([o_x_p, o_x_n.reshape(-1)])

        labels = torch.zeros_like(o_x)
        labels[:b_size] = 1

        return o_x, labels

class BiEncoderDot2(BiEncoder):
    def __init__(self, transformer_name, vocab_size, PAD):
        super().__init__(transformer_name, vocab_size, PAD)

    def forward(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0][:, 0, :]
        r_x = r_x[0][:, 0, :]

        o_x = torch.einsum('be,be->b', [c_x, r_x])
        return o_x

    def forward_with_negatives(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0][:, 0, :]
        r_x = r_x[0][:, 0, :]

        b_size = c_x.shape[0]

        o_x = torch.einsum('be,de->bd', [c_x, r_x]).view(-1 )
        device = self.parameters().__next__().device
        labels = torch.eye(b_size).view(-1).to(device)

        return o_x, labels


class BiEncoderBil2(BiEncoder):
    def __init__(self, transformer_name, vocab_size, PAD):
        super().__init__(transformer_name, vocab_size, PAD)
        self.weight = nn.Parameter(torch.zeros((768, 768)), requires_grad=True)
        bound = 1 / math.sqrt(768)
        init.uniform_(self.weight, -bound, bound)

    def forward(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0][:, 0, :]
        r_x = r_x[0][:, 0, :]

        o_x = torch.einsum('be,ef,bf->b', [c_x, self.weight, r_x])
        return o_x

    def forward_with_negatives(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0][:, 0, :]
        r_x = r_x[0][:, 0, :]

        b_size = c_x.shape[0]

        o_x = torch.einsum('be,ef,df->bd', [c_x, self.weight, r_x]).view(-1)
        device = self.parameters().__next__().device
        labels = torch.eye(b_size).view(-1).to(device)

        return o_x, labels

class BiEncoderSubMult(BiEncoder):
    def __init__(self, transformer_name, vocab_size, PAD):
        super().__init__(transformer_name, vocab_size, PAD)
        self.final = nn.Sequential(
            nn.Linear(768 * 5, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )

    def forward(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0][:, 0, :]
        r_x = r_x[0][:, 0, :]

        a = c_x
        b = r_x
        m = torch.cat([a, b, a - b, a * b, (a - b) * (b - a)], -1)
        m = self.final(m).squeeze(-1)
        return m

    def forward_with_negatives(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0][:, 0, :]
        r_x = r_x[0][:, 0, :]

        b_size = c_x.shape[0]
        A = get_neg_indexes(b_size)

        neg_candidates = r_x[A[:b_size, :b_size - 1].reshape(-1)].reshape(b_size, b_size - 1, -1)  # B(B-1)H

        a = c_x.repeat(c_x.shape[0], 1)  # (BB)E
        b = torch.cat([r_x, neg_candidates.reshape(-1, neg_candidates.shape[-1])], 0)  # (BB)E
        m = torch.cat([a, b, a - b, a * b, (a - b) * (b - a)], -1)
        m = self.final(m).squeeze(-1)

        labels = torch.zeros_like(m)
        labels[:b_size] = 1

        return m, labels

class BiEncoderSubMult2(BiEncoder):
    def __init__(self, transformer_name, vocab_size, PAD):
        super().__init__(transformer_name, vocab_size, PAD)
        self.final = nn.Sequential(
            nn.Linear(768 * 5, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )
        self.pad = PAD

    def forward(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_0 = c_x[0][:, 0, :]
        r_0 = r_x[0][:, 0, :]

        m_bi = submult(c_0, r_0)
        o_bi = self.final(m_bi).squeeze(-1)

        scores = o_bi

        return scores

    def forward_with_negatives(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_0 = c_x[0][:, 0, :]
        r_0 = r_x[0][:, 0, :]


        m_bi = submult(c_0.unsqueeze(1).repeat_interleave(r.shape[0], 1),
                       r_0.unsqueeze(0).repeat_interleave(c.shape[0], 0))
        o_bi = self.final(m_bi).squeeze(-1).view(-1)

        scores = o_bi

        b_size = c.shape[0]
        device = self.parameters().__next__().device
        labels = torch.eye(b_size).view(-1).to(device)

        return scores, labels




class BiEncoderRNN2(nn.Module):
    def __init__(self, _, vocab_size, PAD):
        super().__init__()
        dim = 200
        self.emb = nn.Embedding(vocab_size, dim)
        self.pad = PAD

        self.rnn = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True, bias=True)
        self.final = nn.Bilinear(dim, dim, 1, bias=False)

        self.final = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def encode(self, c, r):
        c_emb = self.emb(c)
        r_emb = self.emb(r)

        c_hs = self.rnn(c_emb)[0]
        r_hs = self.rnn(r_emb)[0]

        c_mask = c.eq(self.pad)
        r_mask = r.eq(self.pad)

        c_hs = c_hs.masked_fill(c_mask.unsqueeze(-1), 0)
        r_hs = r_hs.masked_fill(r_mask.unsqueeze(-1), 0)

        c_len = (~c_mask).sum(1, keepdim=True)
        r_len = (~r_mask).sum(1, keepdim=True)

        c_0 = c_hs[:, 0, :]
        r_0 = r_hs[:, 0, :]

        c_max = c_hs.max(1)[0]
        r_max = r_hs.max(1)[0]

        c_avg = c_hs.sum(1) / c_len
        r_avg = r_hs.sum(1) / r_len

        c_xx = torch.cat([c_0, c_max, c_avg], -1)  # BX
        r_xx = torch.cat([r_0, r_max, r_avg], -1)  # BX

        return c_xx, r_xx

    def forward_with_negatives(self, c, r):

        c_xx, r_xx = self.encode(c, r)

        scores = torch.einsum('be,de->bd', [c_xx, r_xx]).view(-1)

        b_size = c.shape[0]
        device = self.parameters().__next__().device
        labels = torch.eye(b_size).view(-1).to(device)

        return scores, labels

    def forward(self, c, r, ):
        c_xx, r_xx = self.encode(c, r)

        scores = torch.einsum('be,be->b', [c_xx, r_xx]).view(-1)

        return scores






class PolyEncoder2(BiEncoder):
    def __init__(self, transformer_name, vocab_size, PAD):
        super().__init__(transformer_name, vocab_size, PAD)

        self.codes = nn.Parameter(torch.zeros(16, 768), requires_grad=True)
        bound = 1 / math.sqrt(self.codes.size(1))
        init.uniform_(self.codes, -bound, bound)

    def forward(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0]
        r_x = r_x[0][:, 0, :]

        w_ci = torch.einsum('bse,me->bsm', [c_x, self.codes])
        w_ci = w_ci.masked_fill(c.eq(self.PAD).unsqueeze(-1), -1e10)
        y_ci = torch.einsum('bse,bsm->bme', [c_x, w_ci.softmax(-2)])
        w = torch.einsum('bme,be->bm', [y_ci, r_x])
        c_x = torch.einsum('bme,bm->be', [y_ci, w.softmax(-1)])

        o_x = torch.einsum('be,be->b', [c_x, r_x])
        return o_x

    def forward_with_negatives(self, c, r):
        c_x, r_x = self.forward_transformer(c, r)

        c_x = c_x[0]
        r_x = r_x[0][:, 0, :]

        b_size = c_x.shape[0]

        w_ci = torch.einsum('bse,me->bsm', [c_x, self.codes])
        w_ci = w_ci.masked_fill(c.eq(self.PAD).unsqueeze(-1), -1e10)
        y_ci = torch.einsum('bse,bsm->bme', [c_x, w_ci.softmax(-2)])
        w = torch.einsum('bme,de->bdm', [y_ci, r_x])
        c_x = torch.einsum('bme,bdm->bde', [y_ci, w.softmax(-1)])

        o_x = torch.einsum('bde,de->bd', [c_x, r_x]).reshape(-1)
        device = self.parameters().__next__().device
        labels = torch.eye(b_size).view(-1).to(device)

        return o_x, labels


class MolyEncoderAggP2(nn.Module):
    def __init__(self, transformer_name, vocab_size, pad):
        super().__init__()
        self.bert1 = instantiate_transformer(transformer_name, vocab_size, True)
        self.bert1.resize_token_embeddings(vocab_size)
        self.pad = pad
        self.final_ff = nn.Sequential(
            nn.Linear(768 * 5 * (3), 768),
            nn.ReLU(),
            nn.Linear(768, 1))
        self.inf = 1e12
        # self.w = nn.Parameter(torch.zeros(768, 768), requires_grad=True)
        # bound = 1 / math.sqrt(self.w.size(1))
        # init.uniform_(self.w, -bound, bound)
        # self.rnn = nn.GRU(768, 768 // 2, batch_first=True, bidirectional=True)
        self.pre_rnn = nn.Sequential(nn.Linear(768 * 5, 768), nn.ReLU())
        self.layers_to_consider = 1

    def absmax(self, a, dim):
        # return torch.gather(a, dim, a.abs().argmax(dim, keepdim=True)).squeeze(dim)
        return a.max(dim)[0]

    # def avg(self, a, dim, l):
    #     a.sum(dim) /

    def forward(self, c, r):
        c_attn_mask = ~c.eq(self.pad)
        r_attn_mask = ~r.eq(self.pad)

        c_x, c_lhs = self.bert1(c, attention_mask=c_attn_mask)
        r_x, r_lhs = self.bert1(r, attention_mask=r_attn_mask)

        c_lhs = torch.stack(c_lhs, 0)  # LBME
        r_lhs = torch.stack(r_lhs, 0)  # LBNE

        scores = []
        for i in range(-self.layers_to_consider, 0):
            r_hs = r_lhs[i]
            c_hs = c_lhs[i]
            att = torch.einsum('bme,bne->bmn', [c_hs, r_hs]) / math.sqrt(768)
            # c_att = torch.einsum('bme,ef,bf->bm', [c_hs, self.w, r_h])
            mask = ~torch.einsum('bm,bn->bmn', [~c.eq(self.pad), ~r.eq(self.pad)])
            att = att.masked_fill(mask, -self.inf)
            c_hat = torch.einsum('bne,bmn->bme', [r_hs, att.softmax(-1)])
            r_hat = torch.einsum('bme,bmn->bne', [c_hs, att.softmax(-2)])

            c_bar = submult(c_hat, c_hs)
            r_bar = submult(r_hat, r_hs)

            c_bar = self.pre_rnn(c_bar)
            r_bar = self.pre_rnn(r_bar)

            c_x = c_bar
            r_x = r_bar

            # c_x = self.rnn(c_bar)[0]
            # r_x = self.rnn(r_bar)[0]

            c_x = c_x.masked_fill(c.eq(self.pad).unsqueeze(-1), 0)
            r_x = r_x.masked_fill(r.eq(self.pad).unsqueeze(-1), 0)

            c_avg = c_x.sum(1) / c.eq(self.pad).logical_not().long().sum(-1).unsqueeze(-1)
            r_avg = r_x.sum(1) / r.eq(self.pad).logical_not().long().sum(-1).unsqueeze(-1)

            c_x = torch.cat([c_x[:, 0], c_avg, self.absmax(c_x, 1)], -1)
            r_x = torch.cat([r_x[:, 0], r_avg, self.absmax(r_x, 1)], -1)

            m_vs = submult(c_x, r_x)

            m_o = self.final_ff(m_vs).squeeze(-1)
            scores.append(m_o)

        scores = torch.stack(scores, 0).sum(0)

        return scores

    def forward_with_negatives(self, c, r):
        c_attn_mask = ~c.eq(self.pad)
        r_attn_mask = ~r.eq(self.pad)

        c_x, c_lhs = self.bert1(c, attention_mask=c_attn_mask)
        r_x, r_lhs = self.bert1(r, attention_mask=r_attn_mask)

        c_lhs = torch.stack(c_lhs, 0)  # LBME
        r_lhs = torch.stack(r_lhs, 0)  # LBNE

        scores = []
        for i in range(-self.layers_to_consider, 0):
            r_hs = r_lhs[i]
            c_hs = c_lhs[i]
            att = torch.einsum('bme,dne->bdmn', [c_hs, r_hs]) / math.sqrt(768)
            # c_att = torch.einsum('bme,ef,bf->bm', [c_hs, self.w, r_h])
            mask = ~torch.einsum('bm,dn->bdmn', [~c.eq(self.pad), ~r.eq(self.pad)])
            att = att.masked_fill(mask, -self.inf)
            c_hat = torch.einsum('dne,bdmn->bdme', [r_hs, att.softmax(-1)])
            r_hat = torch.einsum('bme,bdmn->bdne', [c_hs, att.softmax(-2)])

            b_size = c.shape[0]

            c_bar = submult(c_hat, c_hs.unsqueeze(1).repeat_interleave(b_size, dim=1))
            r_bar = submult(r_hat, r_hs.unsqueeze(0).repeat_interleave(b_size, dim=0))

            c_bar = self.pre_rnn(c_bar)
            r_bar = self.pre_rnn(r_bar)

            c_x = c_bar
            r_x = r_bar

            # c_x = self.rnn(c_bar.view(b_size * b_size, -1, 768))[0].view(b_size, b_size, -1, 768)
            # r_x = self.rnn(r_bar.view(b_size * b_size, -1, 768))[0].view(b_size, b_size, -1, 768)

            c_x = c_x.masked_fill(c.eq(self.pad).unsqueeze(1).unsqueeze(-1), 0)
            r_x = r_x.masked_fill(r.eq(self.pad).unsqueeze(0).unsqueeze(-1), 0)

            c_avg = c_x.sum(2) / c.eq(self.pad).logical_not().long().sum(-1).unsqueeze(1).unsqueeze(-1)
            r_avg = r_x.sum(2) / r.eq(self.pad).logical_not().long().sum(-1).unsqueeze(0).unsqueeze(-1)

            c_x = torch.cat([c_x[:, :, 0], c_avg, self.absmax(c_x, 2)], -1)
            r_x = torch.cat([r_x[:, :, 0], r_avg, self.absmax(r_x, 2)], -1)

            m_vs = submult(c_x, r_x)  # bd(5e)

            m_o = self.final_ff(m_vs).squeeze(-1).view(-1)
            scores.append(m_o)

        scores = torch.stack(scores, 0).sum(0)

        device = self.parameters().__next__().device
        labels = torch.eye(b_size).view(-1).to(device)

        return scores, labels

