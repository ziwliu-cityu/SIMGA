import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
    
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)   #row-wise
        self.softmax_col = nn.Softmax(dim=-2)   #column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)       
        query_norm_inverse = 1/torch.norm(elu_query, dim=3,p=2) #(L2 norm)
        key_norm_inverse = 1/torch.norm(elu_key, dim=2,p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij',elu_query,query_norm_inverse)
        normalized_key_layer = torch.einsum('mnij,mnj->mnij',elu_key,key_norm_inverse)
        context_layer = torch.matmul(normalized_query_layer,torch.matmul(normalized_key_layer,value_layer))/ self.sqrt_attention_head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class GLINTRU(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GLINTRU, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.n_heads = config["n_heads"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.dense1 = nn.Linear(self.embedding_size, self.hidden_size)
        self.dense2 = nn.Linear(self.embedding_size, self.hidden_size)
        self.conv1d = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.selective_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.Dropout(0.3),
        )
        self.conv1dforgru = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.linearattention = MultiHeadAttention(
            self.n_heads,
            self.hidden_size,
            self.hidden_dropout_prob,
            self.attn_dropout_prob,
            self.layer_norm_eps,
        )
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))  # 使用 nn.Parameter
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        self.dense3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.denseout = nn.Linear(self.hidden_size, self.embedding_size)
        self.dropdense = nn.Dropout(0.3)
        self.dropmix = nn.Dropout(0.3)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.gelu = nn.GELU()

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, item_seq, item_seq_len):
        self.gru_layers.flatten_parameters()
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        #-------------------------First Layer-------------------------------
        attention_output = self.linearattention(item_seq_emb_dropout)
        h1 = self.dense1(item_seq_emb_dropout)
        h2 = self.dense2(item_seq_emb_dropout)
        h1 = self.conv1d(h1.transpose(1, 2))
        h1 = h1.transpose(1, 2)
        h2 = self.gelu(h2)
        
        #-------------------------Mixed Temporal Block----------------------
        gru_output, _ = self.gru_layers(h1)
        selective_gate = self.selective_gate(h1)
        gru_output = self.projection(gru_output)
        gru_output = selective_gate * gru_output
        gru_output = self.conv1dforgru(gru_output.transpose(1, 2))
        gru_output = gru_output.transpose(1, 2)

        softmax_weights = F.softmax(self.weights, dim=0)  # 使用一个新的变量
        expert_output = softmax_weights[0] * gru_output + softmax_weights[1] * attention_output
        h = expert_output * h2
        h = self.dense(h)
        h = self.dropmix(h)
        h = self.LayerNorm(h + item_seq_emb_dropout)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)

        x1 = self.dense3(h)
        x2 = self.dense4(h)
        x2 = self.gelu(x2)
        x = x1 * x2
        x = self.denseout(x)
        x = self.dropdense(x)
        x = self.LayerNorm(x + h)

        seq_output = self.gather_indexes(x, item_seq_len - 1)
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