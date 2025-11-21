# _*_ coding:utf-8 _*_
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

print(torch.cuda.is_available())
print(torch.__version__)
curPath = os.path.abspath(os.path.dirname('__file__'))
# rootPath = os.path.split(curPath)[0]
rootPath = curPath
sys.path.append(rootPath)
modelPath = rootPath + '/model'
from tools.dataset_class_params_bundle import *
from tools.utils import *
from tools.metric import *
import scipy.sparse as sp
import torch.nn.functional as F
import tools.loss as loss

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse

parser = argparse.ArgumentParser(description='MultiCBR')
parser.add_argument('--lr', type=float, help='input learning rate', default=1e-6, required=False)
parser.add_argument('--multi', type=float, help='input Effective API proportion', default=1, required=False)
parser.add_argument('--latent', type=int, help='input Mashup/API latent feature', default=512, required=False)
parser.add_argument('--L2', type=float, help='input L2 regularization', default=1e-8, required=False)
parser.add_argument('--lt_threshold', type=int, help='input long tail threshold', default=2, required=False)
parser.add_argument('--tag', type=str, help='input train tag', default="train_ub", required=False)
parser.add_argument('--device', type=int, help='input gpu device', default=1, required=False)
parser.add_argument('--batch', type=int, help='input batch size', default=128, required=False)
parser.add_argument('--user_weight', type=float, help='input convolution shape', default=0.3, required=False)
parser.add_argument('--bundle_weight', type=float, help='input convolution shape', default=0.9, required=False)
args = parser.parse_args()
class MultiCBRConfig(object):
    def __init__(self, ds_config, mushup_emb, bundle_emb, service_emb):
        # no
        self.model_name = 'LACL-full-lr{0}-x{1}-dim{2}-l2{3}'.format(
            args.lr,
            args.multi,
            args.latent,
            args.L2)
        if args.tag != "":
            self.model_name += "-{0}".format(args.tag)
        self.feature_dim = args.latent
        self.embed_dim = args.latent
        self.dropout = 0.2
        self.epison = 1e-8
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.num_bundle = ds_config.num_bundle
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed
        self.num_layers = 1
        self.mashup_emb = torch.from_numpy(mushup_emb).float()
        self.bundle_emb = torch.from_numpy(bundle_emb).float()
        self.service_emb = torch.from_numpy(service_emb).float()
        self.item_level_ratios = 0.2
        self.bundle_level_ratios = 0.2
        self.bundle_agg_ratios = 0.2
        self.aug_type = "ED"
        self.c_lambda = 0.01
        self.ed_interval = 1
        self.c_temp = 0.2
        print("{0}==>{1}".format(self.num_mashup, self.num_api))
        self.ds = ds_config
        self.lantent_dim = args.latent
        self.K = 3
        self.lr = args.lr
        self.L2 = args.L2
        self.batch_size = args.batch
        device = 'cuda:{0}'.format(args.device)
        self.device = (device if torch.cuda.is_available() else 'cpu')
        self.model_name += "-bs{0}".format(self.batch_size)
        self.num_layers = 2
        self.fusion_weights = {'modal_weight': [0.1, 0.2, 0.7], 'UB_layer': [0.33, 0.33, 0.33], 'UI_layer': [0.33, 0.33, 0.33], 'BI_layer': [0.33, 0.33, 0.33]}
        self.UB_ratio = 0.05
        self.UI_ratio = 0.0
        self.BI_ratio = 0.15
        print(self.model_name)
        print(self.device)
def cal_bpr_loss(pred):
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)
    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)
    return loss
def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph
def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph
def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values
def zero_out_ones(sparse_tensor):
    values = sparse_tensor.values()
    indices = sparse_tensor.indices()
    mask = values == 1
    indices_to_zero = indices[:, mask]
    values.scatter_(0, indices_to_zero[0], 0)
    return torch.sparse_coo_tensor(indices, values, sparse_tensor.size())
def filter_sparse_tensor(sparse_tensor):
    values = sparse_tensor.values()
    indices = sparse_tensor.indices()
    filtered_values = values[values > 1]
    filtered_indices = indices[:, values > 1]
    filtered_tensor = torch.sparse_coo_tensor(filtered_indices, filtered_values, sparse_tensor.size())
    return filtered_tensor
class Autoencoder(nn.Module):
    def __init__(self, embed_dim, num_bundles, num_users, num_items):
        super(Autoencoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_bundles = num_bundles
        self.num_users = num_users
        self.num_items = num_items
        self.bundle_encoder = nn.Linear(self.embed_dim, self.num_bundles)
        self.user_encoder = nn.Linear(self.embed_dim, self.num_users)
        self.item_decoder = nn.Linear(self.num_items, self.embed_dim)
        self.relu = nn.ReLU()        
    def forward(self, item_text):
        bundle_emb = self.item_decoder(self.relu(self.bundle_encoder(item_text)).T)
        user_emb = self.item_decoder(self.relu(self.user_encoder(item_text)).T)
        return bundle_emb, user_emb
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
    def forward(self, x, attn_mask=None, length_mask=None):
        N = x.shape[0]
        L = x.shape[1]
        attn_output, _ = self.attention(
            x, x, x,
            key_padding_mask=attn_mask
        )
        x = x + self.dropout(attn_output)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x+y)

class LACL(nn.Module):
    def __init__(self, config, create_embeddings = True, pretrain=None):
        super(LACL, self).__init__()
        self.embedding_size = config.embed_dim
        self.embed_L2_norm = config.L2
        self.num_users = config.num_mashup
        self.num_bundles = config.num_bundle
        self.num_items = config.num_api
        self.config = config
        self.device = config.device
        self.num_layers = config.num_layers
        self.c_temp = config.c_temp
        self.fusion_weights = config.fusion_weights
        self.init_emb()
        self.init_fusion_weights()
        self.ub_graph = sp.coo_matrix((config.ds.mashup_ds.train_ub_matrix), shape=(self.num_users, self.num_bundles)).tocsr()
        self.ui_graph = sp.coo_matrix((config.ds.mashup_ds.mashup_api_matrix), shape=(self.num_users, self.num_items)).tocsr()
        self.bi_graph = sp.coo_matrix((config.ds.mashup_ds.bundle_item_matrix), shape=(self.num_bundles, self.num_items)).tocsr()
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)

        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)

        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, config.UB_ratio)

        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, config.UI_ratio)
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, config.UI_ratio)

        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, config.BI_ratio)
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, config.BI_ratio)

        self.items_flc = nn.Linear(self.num_items, config.embed_dim)
        self.bundle_flc = nn.Linear(config.embed_dim, self.num_bundles)

        self.query = nn.Sequential(nn.Linear(config.embed_dim, config.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.embed_dim, 1))
        self.user_transformer_encoder = TransformerEncoderLayer(d_model=config.embed_dim, num_heads=2, dropout=0.2)
        self.bundle_transformer_encoder = TransformerEncoderLayer(d_model=config.embed_dim, num_heads=2, dropout=0.2)
        self.item_transformer_encoder = TransformerEncoderLayer(d_model=config.embed_dim, num_heads=2, dropout=0.2)
        self.ae = Autoencoder(config.embed_dim, self.num_bundles, self.num_users, self.num_items)
        self.encoder = nn.Linear(self.num_items, self.num_bundles)
        self.pool_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

        if self.config.aug_type == 'MD':
            self.init_md_dropouts()
        elif self.config.aug_type == "Noise":
            self.init_noise_eps()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mashup_emb = nn.Parameter(torch.tensor(config.mashup_emb)).float().to(self.device)
        self.bundle_emb = nn.Parameter(torch.tensor(config.bundle_emb)).float().to(self.device)
        self.service_emb = nn.Parameter(torch.tensor(config.service_emb)).float().to(self.device)
        emb_dim = self.mashup_emb.shape[1]
        self.mlp = nn.Sequential(nn.Linear(emb_dim, (emb_dim+config.embed_dim)//2),
                                 nn.ReLU(),
                                 nn.Linear((emb_dim+config.embed_dim)//2, config.embed_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def init_md_dropouts(self):
        self.UB_dropout = nn.Dropout(self.config.UB_ratio, True)
        self.UI_dropout = nn.Dropout(self.config.UI_ratio, True)
        self.BI_dropout = nn.Dropout(self.config.BI_ratio, True)
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout
        }

    def init_noise_eps(self):
        self.UB_eps = self.config.UB_ratio
        self.UI_eps = self.config.UI_ratio
        self.BI_eps = self.config.BI_ratio
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }

    def to_csr(self, sparse_tensor):
        if not torch.is_tensor(sparse_tensor) or not sparse_tensor.is_sparse:
            raise ValueError("Input must be a PyTorch sparse tensor.")
        indices = sparse_tensor._indices().numpy()
        values = sparse_tensor._values().numpy()
        size = sparse_tensor.size()
        coo_matrix = sp.coo_matrix((values, (indices[0], indices[1])), shape=size)
        csr_matrix = coo_matrix.tocsr()
        return csr_matrix
    def to_tensor(self, graph):
        graph = graph.tocoo()
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

        return graph
    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def init_fusion_weights(self):
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"
        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1),\
            "The number of layer fusion weights does not correspond to number of layers"
        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)

    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.config.aug_type == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(device)


    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device

        if modification_ratio != 0:
            if self.config.aug_type == "ED":
                graph = bipartite_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)

    def transformer_layer(self, embeds, graph_type, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            if graph_type == "I":
                embeds = self.item_transformer_encoder(embeds, mask)
            elif graph_type == "U":
                embeds = self.user_transformer_encoder(embeds, mask)
            else:
                embeds = self.bundle_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:

            embeds = self.item_transformer_encoder(embeds, mask)
        
        return embeds
    def get_layer_coef(self, all_features):
        for i in range(len(all_features)):
            if i == 0:
                layer_common = self.query(all_features[i])
            else:
                layer_common = torch.cat([layer_common,self.query(all_features[i])],dim = -1)
            weight_common = self.softmax(layer_common)

        return weight_common.unsqueeze(2)

    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.config.aug_type == "MD" and not test:
                mess_dropout = self.mess_dropout_dict[graph_type]
                features = mess_dropout(features)
            elif self.config.aug_type == "Noise" and not test:
                random_noise = torch.rand_like(features).to(self.device)
                eps = self.eps_dict[graph_type]
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps
            a_feature, b_feature = torch.split(features, (A_feature.shape[0], B_feature.shape[0]), 0)    
            if graph_type == "UB":
                # print(a_feature.shape, b_feature.shape)
                a_feature = a_feature + self.transformer_layer(a_feature, "U")
                b_feature = b_feature + self.transformer_layer(b_feature, "B")
            # if graph_type == "UI":
            #     a_feature = a_feature + self.transformer_layer(a_feature, "U")
            #     b_feature = b_feature + self.transformer_layer(b_feature, "I")
            # if graph_type == "BI":
            #     a_feature = a_feature + self.transformer_layer(a_feature, "B")
            #     b_feature = b_feature + self.transformer_layer(b_feature, "I")

            features = torch.cat((a_feature, b_feature), 0)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        # simple embedding dropout on bundle embeddings
        if self.config.aug_type == "MD" and not test:
            mess_dropout = self.mess_dropout_dict[graph_type]
            aggregated_feature = mess_dropout(aggregated_feature)
        elif self.config.aug_type == "Noise" and not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            eps = self.eps_dict[graph_type]
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature


    def fuse_users_bundles_feature(self, users_feature, bundles_feature):
        users_feature = torch.stack(users_feature, dim=0)
        bundles_feature = torch.stack(bundles_feature, dim=0)

        # Modal aggregation
        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        return users_rep, bundles_rep

    def pooling(self, all_feature):
        pool_feature_att = self.spatialattention(all_feature)
        # pool_feature = pool_feature_att.T * all_adaptive
        pool_feature = pool_feature_att.T * all_feature[0] + pool_feature_att.T * all_feature[1]
        # pool_feature = self.tanh(pool_feature)
        pool_feature = F.normalize(pool_feature, p=2, dim=1)
        
        return pool_feature

    def spatialattention(self, all_feature):
        all_mean = torch.mean(torch.stack(all_feature).permute(2, 0, 1), dim=1, keepdim=True)
        all_max, _ = torch.max(torch.stack(all_feature).permute(2, 0, 1), dim=1, keepdim=True)

        pool_feature_att = torch.cat([all_max, all_mean], dim=1)
        pool_feature_att = self.pool_conv(pool_feature_att.transpose(0, 1)).squeeze()

        pool_feature_att = self.sigmoid(pool_feature_att)
        return pool_feature_att  
    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        if test:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        if test:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
        else:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)
            # print("aaaaa",UI_users_feature.shape,UI_items_feature.shape,UI_bundles_feature.shape)

        #  =============================  BI graph propagation  =============================
        if test:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
        else:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)

        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]

        users_rep, bundles_rep = self.fuse_users_bundles_feature(users_feature, bundles_feature)

        return users_rep, bundles_rep


    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def cal_infonce_loss(self, embeds1, embeds2, all_embeds2, temp=0.2):
        normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
        normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
        normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
        nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
        deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
        # print(nume_term.shape,deno_term.shape)
        cl_loss = (nume_term + deno_term).sum()
        return cl_loss
    def cal_loss(self, users_feature, bundles_feature):
        # users_feature / bundles_feature: [bs, 1+neg_num, emb_size]
        pred = torch.sum(users_feature * bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        # cl is abbr. of "contrastive loss"
        u_view_cl = self.cal_c_loss(users_feature, users_feature)
        b_view_cl = self.cal_c_loss(bundles_feature, bundles_feature)

        c_losses = [u_view_cl, b_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss

    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.config.UB_ratio)

            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.config.UI_ratio)
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.config.UI_ratio)

            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.config.BI_ratio)
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.config.BI_ratio)

        users, bundles = batch
        users_rep, bundles_rep = self.get_multi_modal_representations()
        users_text, bundles_text, item_text = self.mlp(self.mashup_emb), self.mlp(self.bundle_emb), self.mlp(self.service_emb)

        bundles_text_item, users_text_item = self.ae(item_text)

        users_text = F.normalize(users_text, p=2, dim=1) + F.normalize(users_text_item, p=2, dim=1)
        bundles_text = F.normalize(bundles_text, p=2, dim=1) + F.normalize(bundles_text_item, p=2, dim=1)


        users_des = users_text[users]
        bundles_des_pos = bundles_text[bundles][:,0,:]
        bundles_des_neg = bundles_text[bundles][:,1,:]
      
        users_embedding = users_rep[users].unsqueeze(1).expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]
        kd_loss = self.cal_infonce_loss(users_rep[users], users_des, users_text) + \
                self.cal_infonce_loss(bundles_embedding[:,0,:], bundles_des_pos, bundles_des_pos) + \
                self.cal_infonce_loss(bundles_embedding[:,1,:], bundles_des_neg, bundles_des_neg)## +intra_c_loss

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss, kd_loss


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result

        scores = torch.mm(users_feature[users], bundles_feature.t())
        loss = cal_bpr_loss(scores)
        return scores, loss

def adjust_learning_rate(optimizer, epoch):

    modellrnew = config.lr * (0.1 ** (epoch // 30))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


class Train(object):
    def __init__(self, input_model, input_config, train_iter, test_iter, val_iter, sample_lt_iter, log, input_ds,
                 model_path=None):
        self.model = input_model
        self.config = input_config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.sample_lt_iter = sample_lt_iter
        self.optim = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=config.L2)
        self.epoch = 200
        self.top_k_list = [5, 10, 15, 20, 25, 30, 40]
        self.log = log
        self.ds = input_ds

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = modelPath + '/checkpoint/%s.pth' % self.config.model_name
        self.early_stopping = EarlyStopping(patience=5, path=self.model_path)
        print(self.early_stopping.patience)
        self.api_des = torch.LongTensor(self.ds.api_ds.description).to(self.config.device)

    def train(self):

        data_iter = self.train_iter
        self.model.train()
        print('Start training ...')

        batch_cnt = len(data_iter)
        ed_interval_bs = int(batch_cnt * self.config.ed_interval)
        for epoch in range(self.epoch):
            self.model.train()
            api_loss = []
            bpr_loss = []
            c_loss = []
            kd_loss = []
            sim_loss = []
            user_iga_loss = []
            bundle_iga_loss = []
            epoch_anchor = epoch * batch_cnt
            for batch_idx, batch_data in enumerate(data_iter):
                users_b = batch_data[0].to(self.config.device)
                bundles = batch_data[1].to(self.config.device)

                batch_anchor = epoch_anchor + batch_idx
                ED_drop = False
                if self.config.aug_type== "ED" and (batch_anchor+1) % ed_interval_bs == 0:
                    ED_drop = True
                
                bpr_loss_, c_loss_, kd_loss_ = model([users_b, bundles], ED_drop)
                api_loss_ = bpr_loss_ + self.config.c_lambda * c_loss_  + kd_loss_* 0.0001 
                self.optim.zero_grad()

                api_loss_.backward()
                self.optim.step()
                api_loss.append(api_loss_.item())
                bpr_loss.append(bpr_loss_.item())
                kd_loss.append(kd_loss_.item())
                c_loss.append(c_loss_.item())


            api_loss = np.average(api_loss)
            c_loss = np.average(c_loss)
            bpr_loss = np.average(bpr_loss)
            kd_loss = np.average(kd_loss)
            user_iga_loss = np.average(user_iga_loss)
            bundle_iga_loss = np.average(bundle_iga_loss)
            sim_loss = np.average(sim_loss)
            info = '[Epoch:%s] ApiLoss:%s C_Loss: %s BPR_Loss: %s KD_Loss: %s ' % (epoch + 1, api_loss.round(6), (c_loss*self.config.c_lambda).round(6),bpr_loss.round(6),(kd_loss*0.00001).round(6))
            print(info)
            self.log.write(info + '\n')
            self.log.flush()
            val_loss = self.evaluate()
            self.early_stopping(float(val_loss), self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, test=False, sample=False):
        if test:
            if sample:
                data_iter = self.sample_lt_iter
                label = 'Sample Test Long-tailed'
                print("Start Sample Test lt")
            else:
                data_iter = self.test_iter
                label = 'Test'
                print('Start testing ...')

        else:
            data_iter = self.val_iter
            label = 'Evaluate'
        self.model.eval()

        # API    
        ndcg_a = np.zeros(len(self.top_k_list))
        recall_a = np.zeros(len(self.top_k_list))
        ap_a = np.zeros(len(self.top_k_list))
        pre_a = np.zeros(len(self.top_k_list))

        pb_a = np.zeros(len(self.top_k_list))

        api_loss = []
        num_batch = len(data_iter)
        with torch.no_grad():
            rs = model.get_multi_modal_representations(test=True)  # users_feature, bundles_feature
            for batch_idx, batch_data in enumerate(data_iter):
                users_b = batch_data[0].to(self.config.device)
                train_mask_u_b = batch_data[2].to(self.config.device)
                ground_truth_u_b = batch_data[3]

                pred_b, api_loss_ = model.evaluate(rs, users_b) 
                pred_b -= 1e8*train_mask_u_b.to(config.device)
                api_loss.append(api_loss_.item())
                pred_b = pred_b.cpu().detach()

                ndcg_, recall_, pre_ = metric_bundle_normal_dp(ground_truth_u_b, pred_b.cpu(),
                                                               top_k_list=self.top_k_list)
                ndcg_a += ndcg_
                recall_a += recall_
                pre_a += pre_

        api_loss = np.average(api_loss)
        ndcg_a /= num_batch
        recall_a /= num_batch
        pre_a /= num_batch

        info = '[%s] ApiLoss:%s \n' \
               'NDCG_A:%s\n' \
               'Pre_A:%s\n' \
               'Recall_A:%s' % (
                   label, api_loss.round(6), ndcg_a.round(6), pre_a.round(6), recall_a.round(6))

        print(info)
        self.log.write(info + '\n')
        self.log.flush()
        return api_loss

    def case_analysis(self):
        case_path = modelPath + '/case/{0}.json'.format(config.model_name)
        a_case = open(case_path, mode='w')
        api_case = []
        self.model.eval()
        with torch.no_grad():

            for batch_idx, batch_data in enumerate(self.test_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                api_target = batch_data[3].argsort(descending=True)[:, :3].tolist()
                api_pred_ = self.model(des, index, self.api_des)

                api_pred_ = api_pred_.cpu().argsort(descending=True)[:, :3].tolist()

                for i, api_tuple in enumerate(zip(api_target, api_pred_)):
                    target = []
                    pred = []
                    name = self.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    api_case.append((name, target, pred))

        json.dump(api_case, a_case)
        a_case.close()
    def get_metrics(self, metrics, grd, pred, topks):
        tmp = {"recall": {}, "ndcg": {}}
        for topk in topks:
            _, col_indice = torch.topk(pred, topk)
            row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)

            row_indice = row_indice.cpu()
            col_indice = col_indice.cpu()
            is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
            row_indice = row_indice.to("cuda")
            col_indice = col_indice.to("cuda")
            tmp["recall"][topk] = self.get_recall(pred, grd, is_hit, topk)
            tmp["ndcg"][topk] = self.get_ndcg(pred, grd, is_hit, topk)

        for m, topk_res in tmp.items():
            for topk, res in topk_res.items():
                for i, x in enumerate(res):
                    metrics[m][topk][i] += x

        return metrics
    def get_recall(self, pred, grd, is_hit, topk):
        epsilon = 1e-8
        hit_cnt = is_hit.sum(dim=1)
        num_pos = grd.sum(dim=1)


        denorm = pred.shape[0] - (num_pos == 0).sum().item()
        nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

        return [nomina, denorm]


    def get_ndcg(self, pred, grd, is_hit, topk):
        def DCG(hit, topk, device):
            hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
            return hit.sum(-1)

        def IDCG(num_pos, topk, device):
            hit = torch.zeros(topk, dtype=torch.float)
            hit[:num_pos] = 1
            return DCG(hit, topk, device)

        device = grd.device
        IDCGs = torch.empty(1 + topk, dtype=torch.float)
        IDCGs[0] = 1  # avoid 0/0
        for i in range(1, topk + 1):
            IDCGs[i] = IDCG(i, topk, device)

        num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
        dcg = DCG(is_hit, topk, device)

        idcg = IDCGs[num_pos]
        ndcg = dcg / idcg.to(device)

        denorm = pred.shape[0] - (num_pos == 0).sum().item()
        nomina = ndcg.sum().item()

        return [nomina, denorm]
if __name__ == '__main__':
    # load ds
    print('Start ...')
    start_time = time.time()
    now = time.time()
    ds = TextDataset(args.multi, is_random=False)
    print('Time for loading dataset: ', get_time(now))
    setup_seed(2020)
    # initial
    # train_idx, val_idx, test_idx = get_indices(ds.mashup_ds)

    with open('./data/generation/emb/msb_emb.pkl', 'rb') as file:
        mashup_emb, bundle_emb, service_emb = pickle.load(file)

    print(mashup_emb.shape,bundle_emb.shape,service_emb.shape)

    config = MultiCBRConfig(ds,mashup_emb,bundle_emb,service_emb)
    train_ds = BPRDataset(ds, sample_indices=ds.train_idx, neg_num=1)
    train_iter = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_iter = DataLoader(ds.mashup_ds, batch_size=len(ds.val_idx),
                        sampler=SubsetRandomSampler(ds.val_idx), drop_last=True)
    test_iter = DataLoader(ds.mashup_ds, batch_size=len(ds.test_idx),
                        sampler=SubsetRandomSampler(ds.test_idx), drop_last=True)
    sample_lt_iter = DataLoader(ds.mashup_ds, batch_size=len(ds.sample_lt),
                                sampler=SubsetRandomSampler(ds.sample_lt), drop_last=True)

    model = LACL(config)
    model.to(config.device)

    # training
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d", timeStruct)
    log_path = modelPath + '/log/{0}.log'.format(config.model_name)
    log = open(log_path, mode='a')
    log.write(strTime + '\n')
    log.flush()

    train_func = Train(input_model=model,
                       input_config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       sample_lt_iter=sample_lt_iter,
                       log=log,
                       input_ds=ds,
                       )
    # training
    train_func.train()

    # testing
    train_func.evaluate(test=True)
    train_func.evaluate(test=True, sample=True)
    # train_func.case_analysis()
    log.close()

    print(config.model_name)
    # print("{0}==>{1}".format(config.num_mashup, config.num_api))