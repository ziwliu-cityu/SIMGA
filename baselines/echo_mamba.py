import torch.nn.functional as F
import torch
from torch import nn
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

class Mamba4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(Mamba4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        
        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
            
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores



# 定义FFT层
class FFTLayer(nn.Module):
    def __init__(self):
        super(FFTLayer, self).__init__()

    def forward(self, x):
        return torch.fft.fft(x)

# 定义逆FFT层
class InverseFFTLayer(nn.Module):
    def __init__(self):
        super(InverseFFTLayer, self).__init__()

    def forward(self, x):
        return torch.fft.ifft(x)


class LearnableHighpassFilter(nn.Module):
    def __init__(self, input_size, cutoff_freq=0.001):
        super(LearnableHighpassFilter, self).__init__()
        self.input_size = input_size
        self.cutoff_freq = nn.Parameter(torch.tensor(cutoff_freq, dtype=torch.float32))
        self.filterh = nn.Parameter(torch.ones(input_size, dtype=torch.cfloat))
    
    def forward(self, input_f):
        device = input_f.device  # 获取输入张量的设备信息
        
        # 创建频率向量
        freqs = torch.fft.fftfreq(self.input_size, device=device)
        
        # 创建高通滤波器
        highpass_filter = ((freqs >= self.cutoff_freq) | (freqs <= -self.cutoff_freq)).float()
        
        # 应用滤波器
        output_f = input_f * highpass_filter + input_f * self.filterh
        
        return output_f


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.combining_weights = nn.Parameter(torch.rand(3))
        self.dense1 = nn.Linear(d_model , d_model)
        self.dense2 = nn.Linear(d_model  , d_model)

        self.l1 = nn.Linear(d_model, d_model)
        self.l2 = nn.Linear(d_model, d_model)

        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)
        self.ac = nn.SiLU()

        # 定义频域处理模块
        self.fft = FFTLayer()
        self.inverse_fft = InverseFFTLayer()
        self.learnable_filter = LearnableHighpassFilter(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_tensor):
        # 频域处理模块
        x_fft = self.fft(input_tensor)
        x_filtered = self.learnable_filter(x_fft)
        x_ifft = self.inverse_fft(x_filtered)
        x_norm = self.layer_norm(x_ifft.real + input_tensor)
        
        # GLU
        x_norm = self.l1(x_norm) * self.ac(self.l2(x_norm)) 

        # 正反向Mamba
        x_reverse = torch.flip(x_norm, [1])
        mamba_output = self.mamba(x_norm)
        mamba_output_f = self.mamba(x_reverse)
        m = self.mamba(input_tensor)
        #output = torch.cat((mamba_output, mamba_output_f), dim=2)  # 拼接维度调整为 dim=2
        output = mamba_output + mamba_output_f + m
        output = self.layer_norm1(output)

        # 调整后的GLU
        h_b = self.dense1(output)
        h_b = self.ac(h_b)
        h_a = self.dense2(output)
        output = h_a * h_b + h_a + output
        output = self.dropout(output)
        hidden_states = self.layer_norm2(output)
        hidden_states = self.ffn(hidden_states)
        
        return hidden_states



class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

