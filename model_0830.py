import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SAP import SelfAttentionPooling
from torch.nn import GroupNorm
from torch_geometric.nn import TransformerConv



# ===========================================================================================
#                                    Encoder
# ===========================================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#################################### transformer #############################
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, 1
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_inputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs


#################################### CNN #####################################
class CNNConvLayersPre(nn.Module):
    def __init__(self, kernels):
        super(CNNConvLayersPre, self).__init__()

        self.cnn_conv = nn.ModuleList()
        for l in range(len(kernels)):
            self.cnn_conv.append(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernels[l], 1), padding=(kernels[l] // 2, 0)))
            self.cnn_conv.append(nn.PReLU())

    def forward(self, x):
        for l, m in enumerate(self.cnn_conv):
            x = m(x)
        return x


class CNNConvLayersMid(nn.Module):
    def __init__(self, channels, kernels):
        super(CNNConvLayersMid, self).__init__()

        self.cnn_conv = nn.ModuleList()
        self.cnn_conv.append(nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=(kernels[0], 1),
                                       padding=(kernels[0] // 2, 0)))
        self.cnn_conv.append(nn.PReLU())
        self.cnn_conv.append(nn.AvgPool2d(kernel_size=(kernels[0], 1), stride=(2, 1), padding=(kernels[0] // 2, 0)))

        for l in range(1, len(kernels) - 1):
            self.cnn_conv.append(
                nn.Conv2d(in_channels=channels[l - 1], out_channels=channels[l], kernel_size=(kernels[l], 1),
                          padding=(kernels[l] // 2, 0)))
            self.cnn_conv.append(nn.PReLU())
            self.cnn_conv.append(nn.AvgPool2d(kernel_size=(kernels[l], 1), stride=(2, 1), padding=(kernels[l] // 2, 0)))

    def forward(self, x):
        for l, m in enumerate(self.cnn_conv):
            x = m(x)
        return x


class CNNConvLayersLast(nn.Module):
    def __init__(self, channels, kernels, feature_dim):
        super(CNNConvLayersLast, self).__init__()

        self.cnn_conv = nn.ModuleList()
        self.cnn_conv.append(
            nn.Conv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size=(kernels[-1], feature_dim),
                      padding=(kernels[-1] // 2, 0)))
        self.cnn_conv.append(nn.PReLU())
        self.cnn_conv.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        for l, m in enumerate(self.cnn_conv):
            x = m(x)
        shapes = x.data.shape
        x = x.view(shapes[0], shapes[1] * shapes[2] * shapes[3])
        return x


class CNNConv(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super(CNNConv, self).__init__()
        kernels = [3, 5, 7]
        channels = [128, 256, out_dim]
        self.conv1 = CNNConvLayersPre(kernels)
        self.conv2 = CNNConvLayersMid(channels, kernels)
        self.conv3 = CNNConvLayersLast(channels, kernels, feature_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class small_CNNConv(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super(small_CNNConv, self).__init__()
        kernels = [3, 5, 7]
        channels = [128, 256, out_dim]
        self.conv2 = CNNConvLayersMid(channels, kernels)
        self.conv3 = CNNConvLayersLast(channels, kernels, feature_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

#################################### GTN #####################################
class gate_GTN(torch.nn.Module):
    def __init__(self, num_feature, out_dim, heads, e_dim, dropout):
        super(gate_GTN, self).__init__()
        self.fc = nn.Linear(num_feature, out_dim)
        self.conv1 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.gate = nn.Linear(out_dim*2, 1)


    def forward(self, x, edge_index, edge_attr):
        x_embed0 = self.fc(x)
        x_embed1 = self.conv1(x_embed0, edge_index, edge_attr)
        x_embed2 = F.leaky_relu(x_embed1)
        coeff = torch.sigmoid(self.gate(torch.cat([x_embed0, x_embed2], -1))).repeat(1, x_embed2.size(-1))
        output = coeff * x_embed0 + (1 - coeff) * x_embed2
        return output



class gate_GTN1(torch.nn.Module):
    def __init__(self, num_feature, out_dim, heads, e_dim, dropout):
        super(gate_GTN1, self).__init__()
        self.fc = nn.Linear(num_feature, out_dim)
        self.conv1 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.gate = nn.Linear(out_dim*2, 1)
        self.gn = GroupNorm(16, out_dim)


    def forward(self, x, edge_index, edge_attr):
        x_embed0 = self.fc(x)
        x_embed1 = self.conv1(x_embed0, edge_index, edge_attr)
        x_embed1 = self.gn(x_embed1)
        x_embed2 = F.gelu(x_embed1)

        coeff = torch.sigmoid(self.gate(torch.cat([x_embed0, x_embed2], -1))).repeat(1, x_embed2.size(-1))
        output = coeff * x_embed0 + (1 - coeff) * x_embed2
        return output

class Res_GTN1(torch.nn.Module):
    def __init__(self, num_feature, e_dim, out_dim, heads, dropout):
        super(Res_GTN1, self).__init__()
        self.conv1 = gate_GTN(num_feature, out_dim, heads, e_dim, dropout)
        self.conv2 = gate_GTN(out_dim, out_dim, heads, e_dim, dropout)
        self.conv3 = gate_GTN(out_dim, out_dim, heads, e_dim, dropout)


    def forward(self, x, edge_index, edge_attr):
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)

        return x


class Atom_GTN1(torch.nn.Module):
    def __init__(self, num_feature, e_dim, out_dim, heads, dropout):
        super(Atom_GTN1, self).__init__()

        self.conv1 = gate_GTN(num_feature, out_dim, heads, e_dim, dropout)
        self.conv2 = gate_GTN(out_dim, out_dim, heads, e_dim, dropout)
        self.conv3 = gate_GTN(out_dim, out_dim, heads, e_dim, dropout)


    def forward(self, x, edge_index, edge_attr):
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)

        return x


class Res_GTN2(torch.nn.Module):
    def __init__(self, num_feature, e_dim, out_dim, heads, dropout):
        super(Res_GTN2, self).__init__()
        self.conv1 = gate_GTN1(num_feature, out_dim, heads, e_dim, dropout)
        self.conv2 = gate_GTN1(out_dim, out_dim, heads, e_dim, dropout)
        self.conv3 = gate_GTN1(out_dim, out_dim, heads, e_dim, dropout)


    def forward(self, x, edge_index, edge_attr):
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)

        return x


class Atom_GTN2(torch.nn.Module):
    def __init__(self, num_feature, e_dim, out_dim, heads, dropout):
        super(Atom_GTN2, self).__init__()

        self.conv1 = gate_GTN1(num_feature, out_dim, heads, e_dim, dropout)
        self.conv2 = gate_GTN1(out_dim, out_dim, heads, e_dim, dropout)
        self.conv3 = gate_GTN1(out_dim, out_dim, heads, e_dim, dropout)


    def forward(self, x, edge_index, edge_attr):
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)

        return x



class Res_GTN(torch.nn.Module):
    def __init__(self, num_feature, e_dim, out_dim, heads, dropout):
        super(Res_GTN, self).__init__()
        self.conv1 = TransformerConv(num_feature, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.conv2 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.conv3 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)

        self.gn = GroupNorm(16, out_dim)
        self.lin4 = torch.nn.Linear(out_dim, out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv1(x, edge_index, edge_attr)
        # print(x.shape)
        x = self.gn(x)
        x = F.gelu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.gn(x)
        x = F.gelu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.gn(x)
        x = F.gelu(x)
        return x


class Atom_GTN(torch.nn.Module):
    def __init__(self, num_feature, e_dim, out_dim, heads, dropout):
        super(Atom_GTN, self).__init__()
        ARMAlayer = 2
        self.conv1 = TransformerConv(num_feature, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.conv2 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.conv3 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)

        self.gn = GroupNorm(4, out_dim)
        self.lin4 = torch.nn.Linear(out_dim, out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.gn(x)
        x = F.gelu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.gn(x)
        x = F.gelu(x)

        return x




class AtomPooling(nn.Module):
    def __init__(self, input_dim):
        super(AtomPooling, self).__init__()
        self.pool = SelfAttentionPooling(input_dim, input_dim)

    def forward(self, atom_features, index_list):
        x = []
        index = index_list.tolist()

        for st2end in index:
            x.append(self.pool(atom_features[int(st2end[0]):int(st2end[1]) + 1, :]))
        # for i in range(len(index)-1):
        #     x.append(self.pool(atom_features[index[i]:index[i+1], :]))
        y = torch.cat((x[0], x[1]), dim=0)
        for j in range(2, len(x)):
            y = torch.cat((y, x[j]), dim=0)
        return y






######################################## 0830 ###########################################
# 加入esm2
#########################################################################################
res_dim = 256
res_dim1 = 256
extra_dim = 256
atom_dim = 16

global_out = 256
local_out = 256

d_model = res_dim + atom_dim + extra_dim  # Embedding Size
d_ff = 1024  # FeedForward dimension
d_k = d_v = 128  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

kernels = [3, 5, 7, 9]



class ANTIGEN_18(nn.Module):
    def __init__(self):
        super(ANTIGEN_18, self).__init__()


        self.esm2_transform = torch.nn.Sequential(
            torch.nn.LayerNorm(1280),
            torch.nn.Linear(1280, 640),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(640, 320),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(320, extra_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(extra_dim, extra_dim)
        )

        self.seq_encoder = Encoder(extra_dim+256, d_ff, d_k, d_v, n_heads, n_layers)
        self.seq_fc = nn.Linear(512, 256)


        self.fc = nn.Linear(105, 256)
        self.res_Encoder = Res_GTN(256, 2, res_dim, 16, 0.8)
        self.atom_Encoder = Atom_GTN(5, 3, atom_dim, 16, 0.8)

        self.pool = AtomPooling(atom_dim)


        self.attention = Encoder(d_model, d_ff, d_k, d_v, n_heads, n_layers)
        self.CNN = CNNConv(res_dim + atom_dim + extra_dim, global_out)

        self.fc1 = nn.Linear(res_dim + atom_dim + extra_dim, local_out)

        self.fc2 = nn.Linear(res_dim, res_dim * 2)
        self.fc3 = nn.Linear(global_out * 2 + local_out * 2, global_out + local_out)
        self.fc4 = nn.Linear(global_out + local_out, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_feat_wt, index_wt,
                res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_feat_mt, index_mt, y):
        # print(len(index2[0]))
        mutpos1, mutpos2 = '', ''
        for i in range(res_x_wt.shape[0]):
            if res_x_wt[i][-1] == 1:
                mutpos1 = i

        for j in range(res_x_mt.shape[0]):
            if res_x_mt[j][-1] == 1:
                mutpos2 = j

        res_x_wt = self.fc(res_x_wt)
        res_x_mt = self.fc(res_x_mt)

        # sequence level
        extra_feat_wt = self.esm2_transform(extra_feat_wt)
        pre_seq_feat_wt = torch.cat((res_x_wt, extra_feat_wt), dim=1)
        mid_seq_feat_wt = self.seq_encoder(pre_seq_feat_wt)
        last_seq_feat_wt = self.seq_fc(mid_seq_feat_wt)


        extra_feat_mt = self.esm2_transform(extra_feat_mt)
        pre_seq_feat_mt = torch.cat((res_x_mt, extra_feat_mt), dim=1)
        mid_seq_feat_mt = self.seq_encoder(pre_seq_feat_mt)
        last_seq_feat_mt = self.seq_fc(mid_seq_feat_mt)



        res_wt = self.res_Encoder(res_x_wt, res_ei_wt, res_e_wt)
        atom_wt = self.atom_Encoder(atom_x_wt, atom_ei_wt, atom_e_wt)
        atom_wt = self.pool(atom_wt, index_wt)
        feature_wt = torch.cat((res_wt, atom_wt, last_seq_feat_wt.squeeze(0)), dim=1)

        global_features_wt = self.attention(feature_wt)
        global_features_wt = self.CNN(global_features_wt.squeeze(0))
        local_features_wt = self.fc1(feature_wt)[mutpos1][:].unsqueeze(0)



        res_mt = self.res_Encoder(res_x_mt, res_ei_mt, res_e_mt)
        atom_mt = self.atom_Encoder(atom_x_mt, atom_ei_mt, atom_e_mt)
        atom_mt = self.pool(atom_mt, index_mt)
        feature_mt = torch.cat((res_mt, atom_mt, last_seq_feat_mt.squeeze(0)), dim=1)

        global_features_mt = self.attention(feature_mt)
        global_features_mt = self.CNN(global_features_mt.squeeze(0))
        local_features_mt = self.fc1(feature_mt)[mutpos2][:].unsqueeze(0)

        feature = torch.cat((global_features_wt, global_features_mt, local_features_wt, local_features_mt), dim=1)
        feature = self.fc6(self.fc5(self.fc4(self.fc3(feature))))
        return feature
