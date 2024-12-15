import torch
from torch import nn
from functools import partial
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss

# Partially based on the implementation of MLPMixer from Lucidrain
# https://github.com/lucidrains/mlp-mixer-pytorch

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x.clone())) + x
    
def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class MLP4Rec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(MLP4Rec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        expansion_factor = 4
        chan_first = partial(nn.Conv1d, kernel_size = 1)
        chan_last = nn.Linear
        self.num_feature_field = len(config['selected_features'])
        self.layerSize = self.num_feature_field + 1

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )

        self.sequenceMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        self.featureMixer = PreNormResidual(self.hidden_size, FeedForward(self.layerSize, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.layers = nn.ModuleList([])
        for i in range(self.num_feature_field+1):
            self.layers.append(self.sequenceMixer)
            self.layers.append(self.channelMixer)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        if sparse_embedding is not None:
            feature_embeddings = sparse_embedding
        if dense_embedding is not None:
            if sparse_embedding is not None:
                feature_embeddings = torch.cat((sparse_embedding,dense_embedding),2)
            else:
                feature_embeddings = dense_embedding
        item_emb = torch.unsqueeze(item_emb,2)
        item_emb = torch.cat((item_emb,feature_embeddings),2)
        mixer_outputs = torch.split(item_emb,[1]*(self.num_feature_field+1),2)
        mixer_outputs = torch.stack(mixer_outputs,0)
        mixer_outputs = torch.squeeze(mixer_outputs)
        for _ in range(self.n_layers):
            for x in range(self.num_feature_field+1):
                mixer_outputs[x] = self.layers[x*2](mixer_outputs[x])
                mixer_outputs[x] = self.layers[(x*2)+1](mixer_outputs[x])
            mixer_outputs = torch.movedim(mixer_outputs,0,2)
            batch_size = mixer_outputs.size()[0]
            mixer_outputs = torch.flatten(mixer_outputs,0,1)
            mixer_outputs = self.featureMixer(mixer_outputs)
            mixer_outputs = torch.reshape(mixer_outputs,(batch_size,self.max_seq_length,self.layerSize,self.hidden_size))
            mixer_outputs = torch.movedim(mixer_outputs,2,0)

        output = self.gather_indexes(mixer_outputs[0], item_seq_len - 1)
        output = self.LayerNorm(output)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
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
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores