import torch.nn.functional as F
import torch
from torch import nn
from functools import partial
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, VanillaAttention
import math
import numpy as np
class SIGMA(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SIGMA, self).__init__(config, dataset)

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
            SIGMALayers(
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



class GMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super(GMambaBlock, self).__init__()
        self.combining_weights = nn.Parameter(torch.tensor([0.1, 0.1, 0.8], dtype=torch.float32))
        #self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        #self.weights_f = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.dense1 = nn.Linear(d_model, d_model)
        self.dense2 = nn.Linear(d_model, d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.gru = nn.GRU(d_model, d_model, num_layers=1, bias=False,batch_first=True)
        self.selective_gate_sig = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(d_model, d_model)
          
        )
        self.selective_gate_si = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model)
            
        )
        self.selective_gate = nn.Sequential(
            nn.Dropout(0.2),
        )
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        #self.ac = nn.SiLU()
        #self.dropout = nn.Dropout(0.2)
        #self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        #self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def initialize_weights(self):
            with torch.no_grad():
                self.combining_weights.data = F.softmax(self.combining_weights.data, dim=0)
                self.weights.data = F.softmax(self.weights.data, dim=0)
    def forward(self, input_tensor):
        self.gru.flatten_parameters()
        combining_weights = F.softmax(self.combining_weights.data, dim=0)
        #weights = F.softmax(self.weights.data, dim=0)
        h1 = self.dense1(input_tensor)
        g1 = self.conv1d(input_tensor.transpose(1, 2))
        g1 = g1.transpose(1, 2)
        gru_input = self.conv1d(g1) 
        #h1 = self.ac(h1)
        #h2 = self.conv1d(input_tensor.transpose(1, 2))
        #h2 = self.ac(h2.transpose(1, 2))
        #h2 = self.conv1d(input_tensor.transpose(1,2))
        #h2 = h2.transpose(1,2)
        # [2048, n, 64] 再进行反转
        flipped_input = input_tensor.clone() #torch.Size([2048, 50, 64])
        
        flipped_input[:, :45, :] = input_tensor[:, :45, :].flip(dims=[1])
        h2 = flipped_input
        h2 = self.dense2(h2)
        h2 = self.dense2(flipped_input) + flipped_input
        # [n,50,64] 再进行反转
        #flipped_input[:2048, :, :] = input_tensor[:2048, :, :].flip(dims=[0])
        mamba_output_f = self.mamba(flipped_input)
        mamba_output = self.mamba(input_tensor)
        #mamba_output = self.LayerNorm(self.dropout(self.mamba(mamba_output)))
        h1 = input_tensor + h1
        #h2 = flipped_input
            
            
        h1 = self.selective_gate_si(h1) + self.selective_gate_sig(h1)
        h1 = self.selective_gate(h1)

        h2 = self.selective_gate_si(h2) + self.selective_gate_sig(h1)
        h2 = self.selective_gate(h2)
            
        mamba_output = mamba_output * h1 + mamba_output
        mamba_output_f = mamba_output_f * h2 +  mamba_output_f 
        
        gru_output, _ = self.gru(g1)
      
        
            
        #combined_states = (
        #                      self.combining_weights[0] * mamba_output_f + 
        #                      self.combining_weights[1] * mamba_output+ 
        #                      self.combining_weights[2] * gru_output
        #                  )
        combined_states = (
          self.combining_weights[2] * mamba_output +
          self.combining_weights[1] * mamba_output_f+
          self.combining_weights[0] * gru_output
        )
        combined_states = self.projection(combined_states)
        return combined_states

class SIGMALayers(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.gmamba = GMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    def forward(self, input_tensor):
            hidden_states = self.gmamba(input_tensor)
            if self.num_layers == 1:        # one Mamba layer without residual connection
                hidden_states = self.LayerNorm(self.dropout(hidden_states))
            else:                           # stacked Mamba layers with residual connections
                hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
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


