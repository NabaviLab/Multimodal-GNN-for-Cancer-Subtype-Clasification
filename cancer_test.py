    ## import all necessary libaraies
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
import sklearn.metrics

import torch
torch.manual_seed(2022)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset

# from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GATConv, global_mean_pool
# from torch_geometric.datasets import Planetoid
# from torch_geometric.datasets import MNISTSuperpixels

# import torch_geometric.transforms as T

# import matplotlib.pyplot as plt
from utils import *
from layer_model import *

import gc

gc.collect()

torch.cuda.empty_cache()

## set all the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default = 0.01, help='learning rate.')
parser.add_argument('--big_lr', type=str2bool, nargs='?', default = True, help='use the larger learning rate.')
parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')
parser.add_argument('--omic_mode', type=int, default = 0, help='which modes of omic to use')
parser.add_argument('--num_omic', type=int, default = 1, help='# of the omic(s) used')
parser.add_argument('--cancer_subtype', type=str2bool, nargs='?', default = False, help='if use the cancer subtype for classification')
parser.add_argument('--specific_type',type=str, default='brca', choices=['brca','luad'], help='which cancer type to use for subtype classification')
parser.add_argument('--shuffle_index',type=int, default=0, help='which shuffle index to use')
parser.add_argument('--batch_size', type=int, default = 16, help='# of genes')
parser.add_argument('--epochs', type=int, default = 100, help='# of epoch')
parser.add_argument('--dropout', type=float, default = 0.6, help='dropout rate')
parser.add_argument('--model', type=str, default = 'gat', choices=['gat','gatv2','gcn','multi-gcn','baseline'], help='which model to use')
parser.add_argument('--decay', type=float, default = 0.9, help='decay rate of the learing rate')
parser.add_argument('--poolsize', type=int, default = 8, help='the max pooling size')
parser.add_argument('--poolrate', type=float, default = 0.8, help='the pooling rate used in the self-attention pooling layer')
parser.add_argument('--gene_gene', type=str2bool, nargs='?', default = True, help='if use the Gene-gene inner connections')
parser.add_argument('--mirna_gene', type=str2bool, nargs='?', default = True, help='if use mirna-mrna connections')
parser.add_argument('--mirna_mirna', type=str2bool, nargs='?', default = True, help='include the meta-path within the mirna')
parser.add_argument('--parallel', type=str2bool, nargs='?', default = True, help='if use the parallel structure')
parser.add_argument('--l2', type=str2bool, nargs='?', default = True, help='if use the l2 regularization')
parser.add_argument('--decoder', type=str2bool, nargs='?', default = True, help='if use the decoder for the graph')
parser.add_argument('--edge_attribute', type=str2bool, nargs='?', default = False, help='if use multi-demension attributes for edges')
parser.add_argument('--edge_weight', type=str2bool, nargs='?', default = False, help='if use score as the edge weight instead of binary edges')
parser.add_argument('--train_ratio', type=float, default = 0.8, help='the ratio of the training data')
parser.add_argument('--test_ratio', type=float, default = 0.1, help='the ratio of the test data')

args = parser.parse_args()

## double check corresponding num_omic and network options according to the selcted omic_mode
## mode 0: mRNA
## mode 1: miRNA
## mode 2: mRNA + miRNA
## mode 3: mRNA + CNV
## mode 4: mRNA + CNV + miRNA
args.num_omic = omic_mode_translation(args.omic_mode)
args.gene_gene, args.mirna_gene, args.mirna_mirna, num_mirna = validate_network_choice(args.omic_mode, args.gene_gene, args.mirna_gene, args.mirna_mirna)

if args.omic_mode == 1:
    args.num_gene = 0

if args.model == 'baseline':
    args.decoder = False
    args.parallel = False
## print the validated input arguments
print('Current arguments:')
print(args)

path = 'data/cancer/'

expression_variance_path = path + 'expression_variance.tsv'
non_null_index_path = path + 'biogrid_non_null.csv'
if args.cancer_subtype:
    if args.specific_type.lower() == 'brca':
        shuffle_index_path = path + 'brca_shuffle_index.tsv'
        cancer_subtype_label_path = path + 'brca_subtype.csv'
        expression_data_path = path + 'expression_data_brca.tsv'
        cnv_data_path = path + 'cnv_data_brca.tsv'
        mirna_data_path = path +'mirna_data_brca.tsv'
else:
    expression_data_path = path + 'standardized_expression_data_with_labels.tsv'
    cnv_data_path = path + 'standardized_cnv_data_with_labels.tsv'
    mirna_data_path = path +'top_100_mirna_data.tsv'
    shuffle_index_path = path + 'common_trimmed_shuffle_index_'+ str(args.shuffle_index) + '.tsv'
adjacency_matrix_path = path + 'adj_matrix_biogrid.npz'
mirna_to_gene_matrix_path = path + 'standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz'

## use the loading function to load the data
if args.omic_mode < 3:
    expr_all_data, mirna_all_data = load_exp_and_real_mirna_data(expression_data_path, mirna_data_path)

    adj, train_data_all, labels, shuffle_index = down_sampling_exp_and_real_mirna_data(expression_variance_path=expression_variance_path,
                                                                        expression_data=expr_all_data,
                                                                        mirna_data=mirna_all_data,
                                                                        omic_mode=args.omic_mode,
                                                                        non_null_index_path=non_null_index_path,
                                                                        shuffle_index_path=shuffle_index_path,
                                                                        adjacency_matrix_path=adjacency_matrix_path,
                                                                        mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
                                                                        gene_gene=args.gene_gene,
                                                                        mirna_gene=args.mirna_gene,
                                                                        mirna_mirna=args.mirna_mirna,
                                                                        number_gene=args.num_gene,
                                                                        singleton=False)
else:
    expr_all_data, cnv_all_data, mirna_all_data = load_exp_cnv_and_real_mirna_data(expression_data_path, cnv_data_path, mirna_data_path)

    adj, train_data_all, labels, shuffle_index = down_sampling_exp_cnv_and_real_mirna_data(expression_variance_path=expression_variance_path,
                                                                        expression_data=expr_all_data,
                                                                        cnv_data=cnv_all_data,
                                                                        mirna_data=mirna_all_data,
                                                                        omic_mode=args.omic_mode,
                                                                        non_null_index_path=non_null_index_path,
                                                                        shuffle_index_path=shuffle_index_path,
                                                                        adjacency_matrix_path=adjacency_matrix_path,
                                                                        mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
                                                                        gene_gene=args.gene_gene,
                                                                        mirna_gene=args.mirna_gene,
                                                                        mirna_mirna=args.mirna_mirna,
                                                                        number_gene=args.num_gene,
                                                                        singleton=False)


## filter data if a specific cancer type is selected
if args.cancer_subtype:
    train_data_all, labels = filter_data_by_cancer_type(cancer_subtype_label_path,
                                                        train_data_all,
                                                        expr_all_data)

## process the labels to make sure it starts from 0
from sklearn import preprocessing
from torch_sparse import SparseTensor

le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)  
# if not args.singleton:      
#     adj, train_data_all = removeZeroAdj(adj, train_data_all)

## adds the self connecting edges and convert the adj to edge_index format
adj_for_loss = adj.todense()
adj = adj/np.max(adj)
adj = adj.astype('float32')
adj.setdiag(0)
adj = adj + sp.eye(adj.shape[0])

adj = sp.coo_matrix(adj)
edge_index = torch.stack([torch.tensor(adj.row), torch.tensor(adj.col)], dim=0)
edge_weight = torch.Tensor(adj.data)

## convert uniformly edge weight into multi-dimension edge attributes if needed.
if args.edge_attribute:
    edge_attribute = disassemble_edge_weights(edge_weight, edge_index, args.num_gene, args.num_omic)

# print(edge_weight.shape)

## split the training and test data
shuffle_index = shuffle_index.astype(np.int32).reshape(-1)

train_size, val_size = int(len(shuffle_index)* args.train_ratio), int(len(shuffle_index)* (1- args.test_ratio))
train_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[0:train_size]]
val_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[train_size:val_size]]
test_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[val_size:]]
train_labels = labels[np.array(shuffle_index[0:train_size])]
val_labels = labels[shuffle_index[train_size:val_size]]
test_labels = labels[shuffle_index[val_size:]]

# ## dropout some training samples
# train_data, train_labels = dropout_data(train_data, train_labels, 0.75)

# ll, cnt = np.unique(train_labels,return_counts=True)

nclass = len(np.unique(labels))
# print(nclass)

train_labels = train_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)
val_labels = val_labels.astype(np.int64)
train_data = torch.FloatTensor(train_data)
test_data = torch.FloatTensor(test_data)
val_data = torch.FloatTensor(val_data)
train_labels = torch.LongTensor(train_labels)
test_labels = torch.LongTensor(test_labels)
val_labels = torch.LongTensor(val_labels)

dset_train = TensorDataset(train_data, train_labels)
train_loader = DataLoader(dset_train, batch_size = args.batch_size, shuffle = True)
dset_test = TensorDataset(test_data, test_labels)
test_loader = DataLoader(dset_test, shuffle = False)
dset_val = TensorDataset(val_data, val_labels)
val_loader = DataLoader(dset_val, batch_size = args.batch_size, shuffle = True)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

print('Device', device)

if args.model == 'gcn':
    model = GCN(args.model, 
                args.parallel, 
                args.l2, args.decoder, 
                args.poolsize, 
                args.poolrate,
                args.edge_weight, 
                args.edge_attribute, 
                args.num_gene,
                num_mirna, 
                args.omic_mode, 
                nclass, 
                args.dropout).to(device)
else:
    model = GAT(args.model, 
                args.parallel, 
                args.l2, args.decoder, 
                args.poolsize, 
                args.poolrate,
                args.edge_weight, 
                args.edge_attribute, 
                args.num_gene,
                num_mirna, 
                args.omic_mode, 
                nclass, 
                args.dropout).to(device)
if args.model == 'baseline':
    model = Baseline(args.model, 
                args.parallel, 
                args.l2, args.decoder, 
                args.poolsize, 
                args.poolrate,
                args.edge_weight, 
                args.edge_attribute, 
                args.num_gene,
                num_mirna, 
                args.omic_mode, 
                nclass, 
                args.dropout).to(device)

## optimizer with adjusted lr
global_lr = args.lr
global_step = 0
decay = args.decay
decay_steps = train_size


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 20% every 12 epochs"""
    if args.big_lr:
        lr = args.lr * pow(decay, float(global_step// decay_steps))
    else:
        lr = args.lr * (0.8 ** (epoch // 12))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr= args.lr)

l2_regularization = 5e-4

t_total_train = time.time()

for epoch in range(args.epochs):
    ## update learning rate
    cur_lr = adjust_learning_rate(optimizer,epoch)

    ## start the timer and the criterion
    t_start = time.time()
    model.train()
    loss_all = 0.0
    accuracy_all = 0.0
    count = 0

    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # print(batch_x.shape)
        # print(batch_y.shape)


        ## as gat or gatv2 not support static graph
        ## create the repeated edge_index for the batch
        count += 1
        batch_edge_index = edge_index.type(torch.int64)

        ## create the edge weight for the batch
        if args.edge_attribute:
            # batch_edge_weight = edge_attribute.type(torch.double)
            batch_edge_weight = edge_attribute
        else:
            # batch_edge_weight = edge_weight.type(torch.double)
            batch_edge_weight = edge_weight

        # print(batch_edge_weight.shape)
        ## concate batch numbers set of graph together
        for i in range(batch_y.shape[0] - 1):
            # print(i)
            if args.edge_weight and args.edge_attribute == False:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
            elif args.edge_attribute:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
            batch_edge_index = torch.cat([batch_edge_index, edge_index+i*(args.num_gene+num_mirna)], axis=1)
            # print(batch_edge_index.size)

        # print(batch_edge_index.shape)
        # print(batch_edge_weight.shape)
        # print(batch_x.shape)
        # print(torch.max(batch_edge_index), torch.min(batch_edge_index))
        # print(torch.max(batch_edge_weight), torch.min(batch_edge_weight))

        # if args.edge_attribute == False:
        #     batch_edge_weight = batch_edge_weight.view(-1,1)
        # print(batch_edge_index.shape)
        # print(batch_edge_weight.shape)
        batch_edge_index = batch_edge_index.to(device)
        batch_edge_weight = batch_edge_weight.to(device)


        
        optimizer.zero_grad()
        if args.decoder:
            # print(args.edge_weight)
            x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
        else:
            out = model(batch_x, batch_edge_index, batch_edge_weight)
        # loss_batch = nn.CrossEntropyLoss()(out, batch_y)

        ## use different loss function with or without encoder
        if args.decoder:
            loss_batch = model.loss(x_reconstruct, batch_x, out, batch_y, l2_regularization)
        else:
            loss_batch = model.loss(batch_x.view(batch_x.size()[0], -1), batch_x, out, batch_y, l2_regularization)
        accuracy_batch = accuracy(out, batch_y)
        loss_batch.backward()
        optimizer.step()
        # loss_all += batch_y.shape[0] * loss.item()
        loss_all += loss_batch.item()
        accuracy_all += accuracy_batch
        global_step += args.batch_size

        
    ## test on the validation data
    # model.eval()

    # running_vloss = 0.0
    # for i, (batch_x, batch_y) in enumerate(val_loader):
    #     batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    #     batch_edge_index = edge_index
    #     for i in range(batch_y.shape[0] - 1):
    #         tmp = torch.cat([batch_edge_index, edge_index+i*(args.num_gene+100)], axis=1)
        
    #     batch_edge_index = batch_edge_index.to(device)

    #     voutputs = model(batch_x, batch_edge_index)
    #     vloss = nn.CrossEntropyLoss()(voutputs, batch_y)
    #     running_vloss += vloss
    t_stop = time.time() - t_start
    accuracy_all = accuracy_all / count
    print(f'epoch: {epoch}, loss: {loss_all}, accuracy:{accuracy_all}')
    # print(f'epoch: {epoch}, accuracy:{accuracy_all}')
    print('training_time:',t_stop)

def test(loader, num_classes):
    model.eval()

    correct = 0
    test_accuracy = 0
    predictions = pd.DataFrame()
    print(len(loader.dataset))
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        ## as gat or gatv2 not support static graph
        ## create the repeated edge_index for the batch
        batch_edge_index = edge_index.type(torch.int64)
        
        if args.edge_attribute:
            batch_edge_weight = edge_attribute
        else:
            batch_edge_weight = edge_weight

        for i in range(batch_y.shape[0] - 1):
            if args.edge_weight and args.edge_attribute == False:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
            elif args.edge_attribute:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
            batch_edge_index = torch.cat([batch_edge_index, edge_index+i*(args.num_gene+num_mirna)], axis=1)
        # print(tmp.size)

    # print(batch_edge_index.shape)

        batch_edge_index = batch_edge_index.to(device)
        batch_edge_weight = batch_edge_weight.to(device)

        if args.decoder:
            x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
        else:
            out = model(batch_x, batch_edge_index, batch_edge_weight)
        
        px = pd.DataFrame(out.detach().cpu().numpy())            
        predictions = pd.concat((predictions, px), axis=0)

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == batch_y).sum())  # Check against ground-truth labels.
        test_accuracy += accuracy(out, batch_y)
    
    ## print the final classification report 
    classification_report = sklearn.metrics.classification_report(test_labels, np.argmax(np.asarray(predictions), 1), labels=range(num_classes))
    print(classification_report)
    return correct / len(loader.dataset), test_accuracy/len(loader.dataset)  # Derive ratio of correct predictions.

print(test(test_loader,nclass))