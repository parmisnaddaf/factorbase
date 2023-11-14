#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:39:58 2023

@author: pnaddaf
"""

from pymysql import connect
from pandas import DataFrame
from numpy import zeros, int64, int32, float64, float32, multiply, dot, sum, array, identity
from itertools import permutations
import numpy as np



db = db_name = "binary_imdb"
connection = connect(host="rcg-cs-ml-dev.dcr.sfu.ca", user="admin", password="joinbayes", db=db)
cursor = connection.cursor()

db_setup = db_name + "_setup"
connection_setup = connect(host="rcg-cs-ml-dev.dcr.sfu.ca", user="admin", password="joinbayes", db=db_setup)
cursor_setup = connection_setup.cursor()

db_bn = db_name + "_bn"
connection_bn = connect(host="rcg-cs-ml-dev.dcr.sfu.ca", user="admin", password="joinbayes", db=db_bn)
cursor_bn = connection_bn.cursor()

keys = {}
cursor_setup.execute("SELECT TABLE_NAME FROM entitytables");
entity_tables = cursor_setup.fetchall()
entities = {}
for i in entity_tables:
    cursor.execute("SELECT * FROM " + i[0])
    rows = cursor.fetchall()
    cursor.execute("SHOW COLUMNS FROM " + db + "." + i[0])
    columns = cursor.fetchall()
    entities[i[0]] = DataFrame(rows, columns=[columns[j][0] for j in range(len(columns))])
    cursor_setup.execute("SELECT COLUMN_NAME FROM entitytables WHERE TABLE_NAME = " + "'" + i[0] + "'")
    key = cursor_setup.fetchall()
    keys[i[0]] = key[0][0]
    


cursor_setup.execute("SELECT TABLE_NAME FROM relationtables")
relation_tables = cursor_setup.fetchall()
relations = {}
for i in relation_tables:
    cursor.execute("SELECT * FROM " + i[0])
    rows = cursor.fetchall()
    cursor.execute("SHOW COLUMNS FROM " + db + "." + i[0])
    columns = cursor.fetchall()
    relations[i[0]] = DataFrame(rows, columns=[columns[j][0] for j in range(len(columns))])
    cursor_setup.execute("SELECT COLUMN_NAME FROM foreignkeycolumns WHERE TABLE_NAME = " + "'" + i[0] + "'")
    key = cursor_setup.fetchall()
    keys[i[0]] = key[0][0], key[1][0]
    
    
    
    
relation_names = tuple(i[0] for i in relation_tables)

indices = {}
for i in entity_tables:
    cursor_setup.execute("SELECT COLUMN_NAME FROM entitytables WHERE TABLE_NAME = '" + i[0] + "'")
    key = cursor_setup.fetchall()[0][0]
    indices[key] = {}
    for index, row in entities[i[0]].iterrows():
        indices[key][row[key]] = index
        
        
matrices = {}
for i in relation_tables:
    cursor_setup.execute("SELECT REFERENCED_TABLE_NAME FROM foreignkeycolumns WHERE TABLE_NAME = " + "'" + i[0] + "'")
    reference = cursor_setup.fetchall()
    matrices[i[0]] = zeros((len(entities[reference[0][0]].index), len(entities[reference[1][0]].index)))


for i in relation_tables:
    cursor_setup.execute("SELECT COLUMN_NAME FROM foreignkeycolumns WHERE TABLE_NAME = '" + i[0] + "'")
    key = cursor_setup.fetchall()
    cursor_setup.execute("SELECT COLUMN_NAME, REFERENCED_COLUMN_NAME FROM foreignkeycolumns WHERE TABLE_NAME = '" + i[0] + "'")
    reference = cursor_setup.fetchall()
    for index, row in relations[i[0]].iterrows():
        matrices[i[0]][indices[reference[0][1]][row[key[0][0]]]][indices[reference[1][1]][row[key[1][0]]]] = 1
        
        
cursor_setup.execute("SELECT COLUMN_NAME, TABLE_NAME FROM attributecolumns")
attribute_columns = cursor_setup.fetchall()
attributes = {}
for i in attribute_columns:
    attributes[i[0]] = i[1]        
        
        


cursor_bn.execute("SELECT DISTINCT child FROM final_path_bayesnets_view")
childs = cursor_bn.fetchall()
rules = []
multiples = []
states = []
functors = {}
variables = {}
nodes = {}
masks = {}
base_indices = []
mask_indices = []
sort_indices = []
stack_indices = []
values = []
for i in range(len(childs)):
    rule = [childs[i][0]]
    cursor_bn.execute("SELECT parent FROM final_path_bayesnets_view WHERE child = " + "'" + childs[i][0] + "'")
    parents = cursor_bn.fetchall()
    for j in parents:
        if j[0] != '':
            rule += [j[0]]
    rules.append(rule)
    if len(rule) == 1:
        multiples.append(0)
    else:
        multiples.append(1)
    relation_check = 0
    for j in rule:
        if j.find(',') != -1:
            relation_check = 1
    functor = {}
    variable = {}
    node = {}
    state = []
    mask = {}
    unmasked_variables = []
    for j in range(len(rule)):
        fun = rule[j].split('(')[0]
        functor[j] = fun
        if rule[j].find(',') == -1:
            var = rule[j].split('(')[1][:-1]
            variable[j] = var
            node[j] = var[:-1]
            if relation_check == 0:
                unmasked_variables.append(var)
                state.append(0)
            else:
                mas = []
                for k in rule:
                    func = k.split('(')[0]
                    if func not in relation_names:
                            func = attributes[func]
                    if k.find(',') != -1 and k.find(var) != -1:
                        unmasked_variables.append(k.split('(')[1][:-1])
                        mas.append([func, k.split('(')[1].split(',')[0], k.split('(')[1].split(',')[1][:-1]]) 
                mask[j] = mas
                state.append(1)
        else:
            unmasked_variables.append(rule[j].split('(')[1][:-1])
            if fun in relation_names:
                state.append(2)
            else:
                state.append(3)     
    functors[i] = functor
    variables[i] = variable
    nodes[i] = node
    states.append(state)
    masks[i] = mask
    masked_variables = [unmasked_variables[0]]
    base_indice = [0]
    mask_indice = []
    for j in range(1, len(unmasked_variables)):
        mask_check = 0
        for k in range(len(masked_variables)):
            if unmasked_variables[j] == unmasked_variables[k]:
                mask_indice.append([k, j])
                mask_check = 1
        if mask_check == 0:
            base_indice.append(j)
            masked_variables.append(unmasked_variables[j])
    sort_indice = []
    sorted_variables = []
    if relation_check == 0:
        sort_indice.append([False, 0])
        sorted_variables.append(masked_variables[0])
    else:
        indices_permutations = list(permutations(range(len(masked_variables))))
        variables_permutations = list(permutations(masked_variables))
        for j in range(len(variables_permutations)):
            indices_chain = []
            variables_chain = []
            first = variables_permutations[j][0].split(',')[0]
            second = variables_permutations[j][0].split(',')[1]
            indices_chain.append([False, indices_permutations[j][0]])
            variables_chain.append(variables_permutations[j][0])
            untransposed_check = 1
            transposed_check = 1
            if len(variables_permutations[j]) > 1:
                for k in range(1, len(variables_permutations[j])):
                    next_first = variables_permutations[j][k].split(',')[0]
                    next_second = variables_permutations[j][k].split(',')[1]
                    if second == next_first:
                        second = next_second
                        indices_chain.append([False, indices_permutations[j][k]])
                        variables_chain.append(next_first + ',' + next_second)
                    elif second == next_second:
                        second = next_first
                        indices_chain.append([True, indices_permutations[j][k]])
                        variables_chain.append(next_second + ',' + next_first)    
                    else:
                        untransposed_check = 0
                        break
                if untransposed_check != 1:
                    indices_chain[0] = [True, indices_permutations[j][0]]
                    variables_chain[0] = second + ',' + first
                    temp = first
                    first = second
                    second = temp
                    for k in range(1, len(variables_permutations[j])):
                        next_first = variables_permutations[j][k].split(',')[0]
                        next_second = variables_permutations[j][k].split(',')[1]
                        if second == next_first:
                            second = next_second
                            indices_chain.append([False, indices_permutations[j][k]])
                            variables_chain.append(next_first + ',' + next_second)
                        elif second == next_second:
                            second = next_first
                            indices_chain.append([True, indices_permutations[j][k]])
                            variables_chain.append(next_second + ',' + next_first)    
                        else:
                            transposed_check = 0
                            break
            if untransposed_check == 1 or transposed_check == 1 or len(variables_permutations[j]) == 1:
                sort_indice = indices_chain
                sorted_variables = variables_chain
                break    
    stack_indice = []
    for j in range(1, len(sorted_variables)):
        second = sorted_variables[j].split(',')[1]
        for k in range(j - 1, -1, -1):
            previous_first = sorted_variables[k].split(',')[0]
            if previous_first == second:
                stack_indice.append([k, j])   
    base_indices.append(base_indice)
    mask_indices.append(mask_indice)
    sort_indices.append(sort_indice)
    stack_indices.append(stack_indice)
    cursor_bn.execute("SELECT * FROM `" + childs[i][0] + "_cp`")
    value = cursor_bn.fetchall()
    values.append(value)
    
    
ground_truth = []

for i in range(len(rules)):
    # print(rules[i])
    for j in values[i]:
        unmasked_matrices = []
        for k in range(len(rules[i])):
            if states[i][k]== 0:
                matrix = zeros((len(entities[nodes[i][k]].index), 1))
                for l in range(len(entities[nodes[i][k]][functors[i][k]])):
                    value = entities[nodes[i][k]][functors[i][k]][l]
                    if type(j[k+multiples[i]]) == str:
                        if type(value) == int64 or type(value) == int32:
                            value = str(value)
                        elif type(value) == float64 or type(value) == float32:
                            value = str(int(value))
                    if value == j[k+multiples[i]]:
                        matrix[indices[keys[nodes[i][k]]][entities[nodes[i][k]][keys[nodes[i][k]]][l]]][0] = 1
                unmasked_matrices.append(matrix)
            elif states[i][k]== 1:
                for l in masks[i][k]:
                    matrix = zeros(matrices[l[0]].shape)
                    for m in range(len(entities[nodes[i][k]][functors[i][k]])):
                        if entities[nodes[i][k]][functors[i][k]][m] == j[k+multiples[i]]:
                            if variables[i][k] == l[1]:
                                matrix[indices[keys[nodes[i][k]]][entities[nodes[i][k]][keys[nodes[i][k]]][m]],:] = 1
                            elif variables[i][k] == l[2]:
                                matrix[:,indices[keys[nodes[i][k]]][entities[nodes[i][k]][keys[nodes[i][k]]][m]]] = 1
                    unmasked_matrices.append(matrix)
            elif states[i][k]== 2:
                if j[k+multiples[i]] == 'F':
                    unmasked_matrices.append(1 - matrices[functors[i][k]])
                else:
                    unmasked_matrices.append(matrices[functors[i][k]])
            elif states[i][k]== 3:
                if j[k+multiples[i]] == 'N/A':
                    unmasked_matrices.append(1 - matrices[attributes[functors[i][k]]])
                else:
                    matrix = zeros(matrices[attributes[functors[i][k]]].shape)
                    for l in range(len(relations[attributes[functors[i][k]]][functors[i][k]])):
                        if relations[attributes[functors[i][k]]][functors[i][k]][l] == j[k+multiples[i]]:
                            matrix[indices[keys[attributes[functors[i][k]]][0]][relations[attributes[functors[i][k]]][keys[attributes[functors[i][k]]][0]][l]]][indices[keys[attributes[functors[i][k]]][1]][relations[attributes[functors[i][k]]][keys[attributes[functors[i][k]]][1]][l]]] = 1 
                    unmasked_matrices.append(matrix)
        masked_matrices = []
        for k in base_indices[i]:
            masked_matrices.append(unmasked_matrices[k])
        for k in mask_indices[i]:
            masked_matrices[k[0]] = multiply(masked_matrices[k[0]], unmasked_matrices[k[1]])
        sorted_matrices = []
        for k in sort_indices[i]:
            if k[0]:
                sorted_matrices.append(masked_matrices[k[1]].T)
            else:
                sorted_matrices.append(masked_matrices[k[1]])
        stacked_matrices = sorted_matrices.copy()   
        pop_counter = 0
        for k in stack_indices[i]:
            for l in range(k[1] - k[0] - pop_counter):
                stacked_matrices[k[0]] = dot(stacked_matrices[k[0]], stacked_matrices[k[0] + 1])
                stacked_matrices.pop(k[0] + 1)
                pop_counter += 1
            stacked_matrices[k[0]] = multiply(stacked_matrices[k[0]], identity(len(stacked_matrices[k[0]])))
        result = stacked_matrices[0]
        for k in range(1, len(stacked_matrices)):
            result = dot(result, stacked_matrices[k])
        # print(sum(result))
        ground_truth.append(sum(result))
    # print("------------------------------------")
    

    
std_dev = np.std(ground_truth)

for i in range(len(ground_truth)):
    ground_truth[i] = ground_truth[i] / std_dev
    
    
    
from typing import Optional, Tuple, List
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.init as init

import random


random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)



EPS = 1e-15
MAX_LOGSTD = 10


import torch.nn as nn
import torch.nn.functional as F


class node_mlp(torch.nn.Module):

    def __init__(self, input, layers= [16, 16], normalize = True, dropout_rate = 0.1):

        super(node_mlp, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(input, layers[0])])

        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i],layers[i+1]))

        self.norm_layers = None
        if normalize:
            self.norm_layers =  torch.nn.ModuleList([torch.nn.BatchNorm1d(c) for c in [input]+layers])
        self.dropout = torch.nn.Dropout(dropout_rate)
        # self.reset_parameters()

    def forward(self, in_tensor, activation = torch.tanh, applyActOnTheLastLyr=True):
        h = in_tensor
        for i in range(len(self.layers)):
            if self.norm_layers!=None:
                if len(h.shape)==2:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h= h.reshape(-1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h=h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)
            if i != (len(self.layers)-1) or applyActOnTheLastLyr:
                h = activation(h)
        return h

class MultiLatetnt_SBM_decoder(torch.nn.Module):

    def __init__(self, number_of_rel, Lambda_dim, in_dim, normalize, DropOut_rate, node_trns_layers=[64]):
        super(MultiLatetnt_SBM_decoder, self).__init__()
 
        self.nodeTransformer = torch.nn.ModuleList(
            node_mlp(in_dim, node_trns_layers + [Lambda_dim], normalize, DropOut_rate) for i in range(number_of_rel))
 
        self.lambdas = torch.nn.ParameterList(
            torch.nn.Parameter(torch.Tensor(Lambda_dim, Lambda_dim)) for i in range(number_of_rel))
        self.numb_of_rel = number_of_rel
        self.reset_parameters()
 
    def reset_parameters(self):
        for i, weight in enumerate(self.lambdas):
            self.lambdas[i] = init.xavier_uniform_(weight)
 
    def forward(self, in_tensor, sigmoid: bool = True):
        gen_adj = []
        for i in range(self.numb_of_rel):
            z = self.nodeTransformer[i](in_tensor)
            h = torch.mm(z, (torch.mm(self.lambdas[i], z.t())))
            gen_adj.append(h)
        return torch.sigmoid(torch.sum(torch.stack(gen_adj), 0)) if sigmoid else torch.sum(torch.stack(gen_adj), 0)

    def forward_pairwise(self, z, edge_index, sigmoid: bool = True):
        gen_adj = []
        for i in range(self.numb_of_rel):
            z_transformed = self.nodeTransformer[i](z)
            h = torch.mm(z_transformed, torch.mm(self.lambdas[i], z_transformed.t()))
            gen_adj.append(h)
        adj_matrix = torch.sum(torch.stack(gen_adj), 0)
        return torch.sigmoid(adj_matrix[edge_index[0], edge_index[1]]) if sigmoid else 0







class MLPDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(MLPDecoder, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, z):
        return self.layers(z)


# class MLPDecoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels=64):
#         super(MLPDecoderRegression, self).__init__()
#         self.layers = torch.nn.Sequential(
#             torch.nn.Linear(in_channels, hidden_channels),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_channels, out_channels)
#         )

#     def forward(self, z):
#         return self.layers(z)


class GAE(torch.nn.Module):

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = MultiLatetnt_SBM_decoder(...) if decoder is None else decoder
        GAE.reset_parameters(self)
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor: 
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        return self.decoder.forward(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        pos_loss = -torch.log(
            self.decoder.forward_pairwise(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0), num_neg_samples = 2000)
        neg_loss = -torch.log(1 -
                              self.decoder.forward_pairwise(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder.forward_pairwise(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder.forward_pairwise(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

class VGAE1(GAE):

    def __init__(self, encoder: Module, decoder: Optional[Module] = None, 
                 node_feat_decoder: Optional[Module] = None,
                 number_of_rel: int = 1, Lambda_dim: int = 32, 
                 in_dim: int = 48, normalize: bool = True, 
                 DropOut_rate: float = 0.1, node_trns_layers: List[int] = [64, 64]):
        sbm_decoder = MultiLatetnt_SBM_decoder(number_of_rel, Lambda_dim, in_dim, normalize, DropOut_rate, node_trns_layers)
        super().__init__(encoder, decoder=sbm_decoder)  # pass your decoder to the GAE class
        self.node_feat_decoder = MLPDecoder(out_channels, num_features) if node_feat_decoder is None else node_feat_decoder


    def node_feat_recon_loss(self, x: Tensor, z: Tensor) -> Tensor:
        x_recon = self.node_feat_decoder(z)
        return torch.nn.functional.mse_loss(x_recon, x)    

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        x_recon = self.node_feat_decoder(z)
        return z, x_recon

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


import os
import torch
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import VGAE
import copy
import numpy as np
# os.environ['TORCH'] = torch.__version__
# print(torch.__version__)
# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git


    
    

# dataset = Planetoid("\..", "Cora")
# dataa = dataset[0]
dataa = torch.load("/home/pnaddaf/factorbase/db/imdb.pt")
dataa.train_mask = dataa.val_mask = dataa.test_mask = dataa.y = None

# def binarize_node_features(data):
#     data.x[data.x > 0] = 1
#     return data


from sklearn.decomposition import PCA

def reduce_node_features(data, n_components=10):
    pca = PCA(n_components=n_components)
    
    reduced_features = pca.fit_transform(data.x.numpy())
    
    data.x = torch.tensor(reduced_features, dtype=torch.float)
    return data


def binarize_features(features):
    mean_val = features.mean()
    binarized_features = (features >= mean_val).float()
    return binarized_features

data_bi = dataa
data_re = reduce_node_features(data_bi)
data_re.x = binarize_features(data_re.x)
data1 = copy.deepcopy(data_re)
data = train_test_split_edges(data_re)




class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) 
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True) 
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        
out_channels = 48  
num_features = 10  
epochs = 100      
Lambda_dim = 32   

node_trns_layers = [out_channels, 64, Lambda_dim]

model = VGAE1(VariationalGCNEncoder(num_features, out_channels), 
              Lambda_dim=Lambda_dim,
              node_trns_layers=node_trns_layers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



def train_motif():
    model.train()
    optimizer.zero_grad()
    z, x_recon = model.encode(x, train_pos_edge_index)  # Get both z and x_recon

    A_pred = model.decoder(z)
    # A_pred = torch.sigmoid(torch.mm(z, z.t()))


    for i in range(1, 11):
        feature_name = f'feature_{i}'
        entities['nodes_table'][feature_name] = ((x_recon[:, i-1].detach().numpy()) > 0.5).astype(int)



    matrices['edges_table'] = A_pred.detach().numpy() 
    
    
    predicted = []
    for i in range(len(rules)):
        # print(rules[i])
        for j in values[i]:
            unmasked_matrices = []
            for k in range(len(rules[i])):
                match states[i][k]:
                    case 0:
                        matrix = zeros((len(entities[nodes[i][k]].index), 1))
                        for l in range(len(entities[nodes[i][k]][functors[i][k]])):
                            value = entities[nodes[i][k]][functors[i][k]][l]
                            if type(j[k+multiples[i]]) == str:
                                if type(value) == int64 or type(value) == int32:
                                    value = str(value)
                                elif type(value) == float64 or type(value) == float32:
                                    value = str(int(value))
                            if value == j[k+multiples[i]]:
                                matrix[indices[keys[nodes[i][k]]][entities[nodes[i][k]][keys[nodes[i][k]]][l]]][0] = 1
                        unmasked_matrices.append(matrix)
                    case 1:
                        for l in masks[i][k]:
                            matrix = zeros(matrices[l[0]].shape)
                            for m in range(len(entities[nodes[i][k]][functors[i][k]])):
                                if entities[nodes[i][k]][functors[i][k]][m] == j[k+multiples[i]]:
                                    if variables[i][k] == l[1]:
                                        matrix[indices[keys[nodes[i][k]]][entities[nodes[i][k]][keys[nodes[i][k]]][m]],:] = 1
                                    elif variables[i][k] == l[2]:
                                        matrix[:,indices[keys[nodes[i][k]]][entities[nodes[i][k]][keys[nodes[i][k]]][m]]] = 1
                            unmasked_matrices.append(matrix)
                    case 2:
                        if j[k+multiples[i]] == 'F':
                            unmasked_matrices.append(1 - matrices[functors[i][k]])
                        else:
                            unmasked_matrices.append(matrices[functors[i][k]])
                    case 3:
                        if j[k+multiples[i]] == 'N/A':
                            unmasked_matrices.append(1 - matrices[attributes[functors[i][k]]])
                        else:
                            matrix = zeros(matrices[attributes[functors[i][k]]].shape)
                            for l in range(len(relations[attributes[functors[i][k]]][functors[i][k]])):
                                if relations[attributes[functors[i][k]]][functors[i][k]][l] == j[k+multiples[i]]:
                                    matrix[indices[keys[attributes[functors[i][k]]][0]][relations[attributes[functors[i][k]]][keys[attributes[functors[i][k]]][0]][l]]][indices[keys[attributes[functors[i][k]]][1]][relations[attributes[functors[i][k]]][keys[attributes[functors[i][k]]][1]][l]]] = 1 
                            unmasked_matrices.append(matrix)
            masked_matrices = []
            for k in base_indices[i]:
                masked_matrices.append(unmasked_matrices[k])
            for k in mask_indices[i]:
                masked_matrices[k[0]] = multiply(masked_matrices[k[0]], unmasked_matrices[k[1]])
            sorted_matrices = []
            for k in sort_indices[i]:
                if k[0]:
                    sorted_matrices.append(masked_matrices[k[1]].T)
                else:
                    sorted_matrices.append(masked_matrices[k[1]])
            stacked_matrices = sorted_matrices.copy()   
            pop_counter = 0
            for k in stack_indices[i]:
                for l in range(k[1] - k[0] - pop_counter):
                    stacked_matrices[k[0]] = dot(stacked_matrices[k[0]], stacked_matrices[k[0] + 1])
                    stacked_matrices.pop(k[0] + 1)
                    pop_counter += 1
                stacked_matrices[k[0]] = multiply(stacked_matrices[k[0]], identity(len(stacked_matrices[k[0]])))
            result = stacked_matrices[0]
            for k in range(1, len(stacked_matrices)):
                result = dot(result, stacked_matrices[k])
            # print(sum(result))
            predicted.append(sum(result))
        # print("---------------------------------------------------------------------------------------------------------------")

    for i in range(len(ground_truth)):
        predicted[i] = predicted[i] / std_dev
                 

    motif = ((a-b)**2 for a, b in zip(ground_truth, predicted))
    motif_loss = np.sum(np.fromiter(motif, dtype=float))





                           
    recon_loss = model.recon_loss(z, train_pos_edge_index)
    node_feat_loss = model.node_feat_recon_loss(x, z)

    alpha = 0.7
    loss = (1-alfa)*(recon_loss + (1 / data.num_nodes) * model.kl_loss() + node_feat_loss) + (alfa)*((1/len(ground_truth))*motif_loss)
    loss.backward()
    optimizer.step()
    return float(loss) 


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z, _ = model.encode(x, data.test_pos_edge_index)

    return model.test(z, pos_edge_index, neg_edge_index)

for epoch in range(1, epochs + 1):
    loss = train_motif()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

