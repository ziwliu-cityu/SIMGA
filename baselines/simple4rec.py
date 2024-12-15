import torch
from torch import nn
from functools import partial
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, VanillaAttention
from recbole.model.loss import BPRLoss


class sMLPBlock(nn.Module):
    def __init__(self,h,w,c):
        super().__init__()
        self.proj_h=nn.Linear(h,h)
        self.proj_w=nn.Linear(w,w)
        self.fuse=nn.Linear(3*c,c)
    
    def forward(self,x):
        x_h=self.proj_h(x.permute(0,1,3,2)).permute(0,1,3,2)
        x_w=self.proj_w(x)
        x_id=x
        x_fuse=torch.cat([x_h,x_w,x_id],dim=1)
        out=self.fuse(x_fuse.permute(0,2,3,1)).permute(0,3,1,2)
        return out


class TokenMixing(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels=in_shape[0],
                                    out_channels=in_shape[0],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_shape[0])
        self.smlp = sMLPBlock(h=in_shape[1], w=in_shape[2], c=in_shape[0])
        self.bn0 = nn.BatchNorm2d(in_shape[0])
        self.bn1 = nn.BatchNorm2d(in_shape[0])
    def forward(self, x):
        x = x+self.dwconv(self.bn0(x))
        x = x+self.smlp(self.bn1(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_shape_0):
        super().__init__()
        self.PM = nn.Conv2d(in_shape_0, 2 * in_shape_0, 2, stride=2, padding=0, groups=in_shape_0)
    def forward(self, x):
        return self.PM(x)

    
def FeedForward(dim, expansion_factor = 2, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class SMLP(nn.Module):
    def __init__(self, in_shape, expansion_factor = 2, dropout=0.):
        super().__init__()
        self.c = nn.Sequential(nn.Linear(in_shape[0], expansion_factor * in_shape[0], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[0], in_shape[0], bias=False),
                               nn.Dropout(dropout),)
        self.h = nn.Sequential(nn.Linear(in_shape[1], expansion_factor * in_shape[1], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[1], in_shape[1], bias=False),
                               nn.Dropout(dropout),)
        self.w = nn.Sequential(nn.Linear(in_shape[2], expansion_factor * in_shape[2], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[2], in_shape[2], bias=False),
                               nn.Dropout(dropout),)
        self.norm2 = nn.LayerNorm(in_shape[2])
    def forward(self, x):
        xn = self.norm2(x)
        x0 = (self.c(xn.transpose(1,3).contiguous())).transpose(3,1).contiguous()
        x1 = (self.h(xn.transpose(2,3).contiguous())).transpose(3,2).contiguous()
        x2 = self.w(xn)
        y = x0 + x1 + x2
        return y

class SMLPREC(SequentialRecommender):
    def __init__(self, config, dataset, seq_len = 50):
        super(SMLPREC, self).__init__(config, dataset)


        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size'] 
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        expansion_factor = 3
        chan_first = partial(nn.Conv1d, kernel_size = 1)
        chan_last = nn.Linear
        self.num_feature_field = len(config['selected_features'])
        self.layerSize = self.num_feature_field + 1

        #self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )

        self.layers = nn.ModuleList([])
        dim_new = self.num_feature_field+1
        self.layers.append(SMLP([dim_new, seq_len, self.hidden_size], expansion_factor, self.hidden_dropout_prob))
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
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


        mixer_outputs = item_emb.transpose(1,2).contiguous()

        for x in range(self.n_layers):
            mixer_outputs = self.layers[0](mixer_outputs)

        mixer_outputs = mixer_outputs.transpose(1, 0).contiguous()
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
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  
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
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) 
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) 
        return scores