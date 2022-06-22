import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN


import argparse
import time

import networkx as nx

import tensorflow as tf

import numpy as np
import scipy.sparse as sp
import json
import os
import shutil

from dataloaders import DataLoaderPolyvore

# Set random seed
seed = int(time.time()) # 12342
np.random.seed(seed)
tf.random.set_seed(
    seed
)

def accuracy(pred, target):
    r"""Computes the accuracy of correct predictions.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()

# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="polyvore",
                choices=['fashiongen', 'polyvore', 'amazon'],
                help="Dataset string.")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                help="Learning rate")

ap.add_argument("-wd", "--weight_decay", type=float, default=0.,
                help="Learning rate")

ap.add_argument('--num_channels', type=int, default=2,
                    help='number of channels')

ap.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')

ap.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')

ap.add_argument('--norm', type=str, default='true',
                        help='normalization')

ap.add_argument("-e", "--epochs", type=int, default=4000,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs='+', default=[350, 350, 350],
                help="Number hidden units in the GCN layers.")

ap.add_argument("-do", "--dropout", type=float, default=0.5,
                help="Dropout fraction")

ap.add_argument("-deg", "--degree", type=int, default=1,
                help="Degree of the convolution (Number of supports)")

ap.add_argument("-sdir", "--summaries_dir", type=str, default="logs/",
                help="Directory for saving tensorflow summaries.")

ap.add_argument("-sup_do", "--support_dropout", type=float, default=0.15,
                help="Use dropout on the support matrices, dropping all the connections from some nodes")

ap.add_argument('-ws', '--write_summary', dest='write_summary', default=False,
                help="Option to turn on summary writing", action='store_true')

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-bn', '--batch_norm', dest='batch_norm',
                help="Option to turn on batchnorm in GCN layers", action='store_true')
fp.add_argument('-no_bn', '--no_batch_norm', dest='batch_norm',
                help="Option to turn off batchnorm", action='store_false')
ap.set_defaults(batch_norm=True)

ap.add_argument("-amzd", "--amz_data", type=str, default="Men_bought_together",
            choices=['Men_also_bought', 'Women_also_bought', 'Women_bought_together', 'Men_bought_together'],
            help="Dataset string.")

args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')

# Define parameters
DATASET = args['dataset']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = "img"
NUMCLASSES = 2
DEGREE = args['degree']
BATCH_NORM = args['batch_norm']
BN_AS_TRAIN = False
SUP_DO = args['support_dropout']
ADJ_SELF_CONNECTIONS = True
VERBOSE = True
num_channels = args['num_channels']
node_dim = args['node_dim']
num_layers = args['num_layers']
norm = args['norm']

# prepare data_loader
dl = DataLoaderPolyvore()
train_features, adj_train, train_labels, train_r_indices, train_c_indices = dl.get_phase('train')
val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase('valid')
test_features, adj_test, test_labels, test_r_indices, test_c_indices = dl.get_phase('test')
G = nx.from_scipy_sparse_matrix(adj_train, create_using=nx.Graph)
edges = np.array(G.edges)
G_val = nx.from_scipy_sparse_matrix(adj_val, create_using=nx.Graph)
edges_val = np.array(G_val.edges)
G_test = nx.from_scipy_sparse_matrix(adj_test, create_using=nx.Graph)
edges_test = np.array(G_test.edges)
# train_features = train_features.toarray()
# val_features = val_features.toarray()

train_r_indices = train_r_indices.astype('int')
train_c_indices = train_c_indices.astype('int')
val_r_indices = val_r_indices.astype('int')
val_c_indices = val_c_indices.astype('int')

train_features, mean, std = dl.normalize_features(train_features, get_moments=True)
val_features = dl.normalize_features(val_features, mean=mean, std=std)
test_features = dl.normalize_features(test_features, mean=mean, std=std)

num_nodes = edges[0].shape[0]

final_f1 = 0

num_classes = np.max(train_labels)+1

for i, edge in enumerate(edges):
    # print(torch.from_numpy(edge).type(torch.FloatTensor).unsqueeze(-1).size())
    if i == 0:
        A = torch.from_numpy(edge).type(torch.FloatTensor).unsqueeze(-1)
    else:
        A = torch.cat([A, torch.from_numpy(edge).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
# print(num_nodes)
# print(A.size())
# print(torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1).size())
# A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

print("Num edge" + str(edges.shape[-1]))
print("Num Channels "+ str(num_channels))
print("Win " + str(train_features.shape[1]))
print("Wout "+ str(node_dim))
print("Num of classes "+ str(num_classes))
print("Num of Layers "+ str(num_layers))

model = GTN.GTN(num_edge=edges.shape[-1],
            num_channels=num_channels,
            w_in=train_features.shape[1],
            w_out=node_dim,
            num_class=num_classes,
            num_layers=num_layers,
            norm=norm)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

loss = nn.CrossEntropyLoss()
# Train & Valid & Test
best_val_loss = 10000
best_test_loss = 10000
best_train_loss = 10000
best_train_f1 = 0
best_val_f1 = 0
best_test_f1 = 0

for i in range(NB_EPOCH):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.005:
            param_group['lr'] = param_group['lr'] * 0.9
    print('Epoch:  ', i + 1)
    model.zero_grad()
    model.train()
    loss, y_train, Ws = model(A, torch.from_numpy(train_features).to(torch.float32), train_labels, train_r_indices, train_c_indices)
    train_f1 = torch.mean(
        accuracy(torch.argmax(y_train.detach(), dim=1), train_labels)).cpu().numpy()
    print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
    loss.backward()
    optimizer.step()
    model.eval()
    # Valid
    with torch.no_grad():
        val_loss, y_valid, _ = model.forward(edges_val, val_features, val_labels, val_r_indices, val_c_indices)
        val_f1 = torch.mean(accuracy(torch.argmax(y_valid, dim=1), val_labels)).cpu().numpy()
        print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
        test_loss, y_test, W = model.forward(edges_test, test_features, test_labels, test_r_indices, test_c_indices)
        test_f1 = torch.mean(accuracy(torch.argmax(y_test, dim=1), test_labels)).cpu().numpy()
        print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
    if val_f1 > best_val_f1:
        best_val_loss = val_loss.detach().cpu().numpy()
        best_test_loss = test_loss.detach().cpu().numpy()
        best_train_loss = loss.detach().cpu().numpy()
        best_train_f1 = train_f1
        best_val_f1 = val_f1
        best_test_f1 = test_f1
print('---------------Best Results--------------------')
print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
final_f1 += best_test_f1