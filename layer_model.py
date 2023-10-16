import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
# from torch.utils.loader import DataLoader

# from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GATv2Conv, ChebConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import MNISTSuperpixels

import torch_geometric.transforms as T
from self_attention import SAGPool

# num_genes = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## define the GAT class
class GAT(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    self_attention,
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(GAT, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.self_attention = self_attention
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.head = 8
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)

        # self.conv1 = GATConv(self.num_features, self.hid, heads=self.head)
        # self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, dropout=dropout_rate)
        if method == 'gatv2':
            if self.edge_attributes:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head, edge_dim=2)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, edge_dim=2)
            elif self.edge_weights:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head, edge_dim=1)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, edge_dim=1)
            else:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head)
                # self.conv3 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, dropout=dropout_rate)

            ## if include the self-attention pooling layer in the model
            # if self.self_attention:
            #     self.pool1 = SAGPool(self.hid * self.head, ratio=self.poolrate)
            #     self.pool2 = SAGPool(self.hid * self.head, ratio=self.poolrate)

        elif method == 'gat':
            if self.edge_attributes:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head, edge_dim=2)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, edge_dim=2)
            elif self.edge_weights:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head, edge_dim=1)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, edge_dim=1)
            else:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head)

            ## if include the self-attention pooling layer in the model
            # if self.self_attention:
            #     self.pool1 = SAGPool(self.hid * self.head, ratio=self.poolrate)
            #     self.pool2 = SAGPool(self.hid * self.head, ratio=self.poolrate)

        
        # if self.poolsize <= 1:
        #     if method == 'gcn':
        #         if self.self_attention:
        #             self.linear_input = self.num_nodes * self.hid * 2
        #         else:
        #             self.linear_input = self.num_nodes * self.hid
        #     else:
        #         print('Only GCN model can use self-attention now.')
        #         quit()
        #         # if self.self_attention:
        #         #     self.linear_input = self.num_nodes * self.hid * self.head * 2
        #         # else:
        #         #     self.linear_input = self.num_nodes * self.hid * self.head
        # else:
        #     if method == 'gcn':
        #         self.linear_input = math.floor(self.num_nodes / self.poolsize) * self.hid
        #     else:
        self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid * self.head
        print(self.linear_input)

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        # self.linear2 = nn.Linear(linear_input//4, linear_input//8)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)
        # self.linear3 = nn.Linear(linear_input//8, num_classes)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:

            parallel_input = self.raised_dimension*(self.num_gene+self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene+self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        # x = torch.cat([x, pos], dim=-1)
        
        # print("Input data shape")
        # print(x.shape)
        # print(edge_index.shape)

        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]
            # print(x_cnv.shape)
            # print(x_exp.shape)
            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            # print(x_mirna.shape)
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            # print(x_gene.shape)
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)
            # print(x_gene.shape)

            # print(x_mirna.shape)
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)
            # print(x_mirna.shape)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        # print(x.shape)
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)
        # x = x.view(-1,1)
        # print('Reformated data shape')
        # print(x.shape)

        # x = F.dropout(x, p=0.8, training=self.training)
        # print('after first dropout:')
        # print(x.shape, edge_index.shape)
        # print(torch.max(edge_index))
        if self.edge_weights:
            # print(edge_index.type())
            # print(edge_weight.type())
            # print(x.type())
            # print('Passing through Conv1 layer with edge_weight.')
            x = self.conv1(x, edge_index, edge_weight)

            ## use different activation function based on the models
            x = F.leaky_relu(x)
        else:
            # print('Passing through Conv1 layer without edge_weight.')
            x = self.conv1(x, edge_index)

            ## use different activation function based on the models
            x = F.leaky_relu(x)
        
        # x = F.dropout(x, p=0.6, training=self.training)

        # print(x.shape)
        if self.edge_weights:
            # print('Passing through Conv2 layer with edge_weight.')
            x = self.conv2(x, edge_index, edge_weight)

            x = F.leaky_relu(x)
        else:
            # print('Passing through Conv2 layer without edge_weight.')
            x = self.conv2(x, edge_index) ## output shape: [batches * num_node, hid * head]

            x = F.leaky_relu(x)
        # print(x.shape)

        ## pooling on the graph to reduce nodes
        # print(x.shape)
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                        ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]
        # print(x.shape)

        x = x.view(-1, self.hid * self.head) ## output shape:[batches * floor(num_node / poolsize), hid * head]
        # print(x.shape)

        # x = self.conv3(x, edge_index)
        # x = F.elu(x)
        # print(x.shape)

        # batch_index = np.array(range(batches))
        # batch_index = np.repeat(batch_index, num_genes+100)
        # batch_index = torch.from_numpy(batch_index).to(device)
        # x = global_mean_pool(x, batch_index)
        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        # print(x.shape)
        x = self.linear1(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.linear2(x)
        x = F.relu(x)
        # print(x.shape)
        # x = self.linear3(x)
        # print(x.shape)

        if self.decoder:
            # print('Passing decoder')
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)

            x_reconstruct  = nn.Dropout(self.dropout_rate)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)

        if self.parallel:
            ## the two layer shallow FC network
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)
            # x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)

            x = torch.cat((x,x_parallel),1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0
        
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss = 1*loss1 + 1*loss2
        
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss += 0.2* l2_regularization* l2_loss
        return loss


class GCN(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    self_attention,
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(GCN, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.self_attention = self_attention
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)
    
        if method == 'gcn':
            self.conv1 = ChebConv(self.raised_dimension, self.hid, K=5)
            self.conv2 = ChebConv(self.hid, self.hid, K=5)

            ## if include the self-attention pooling layer in the model
            if self.self_attention:
                self.pool1 = SAGPool(self.hid, ratio=self.poolrate)
                self.pool2 = SAGPool(self.hid, ratio=self.poolrate)

        if self.poolsize <= 1:
            if method == 'gcn':
                if self.self_attention:
                    self.linear_input = (self.num_gene + self.num_mirna) * self.hid * 2
                else:
                    self.linear_input = (self.num_gene + self.num_mirna) * self.hid
        else:
            if method == 'gcn':
                self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:

            parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        # x = torch.cat([x, pos], dim=-1)
        
        # print("Input data shape")
        # print(x.shape)
        # print(edge_index.shape)

        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]
            # print(x_cnv.shape)
            # print(x_exp.shape)
            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            # print(x_mirna.shape)
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            # print(x_gene.shape)
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)
            # print(x_gene.shape)

            # print(x_mirna.shape)
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)
            # print(x_mirna.shape)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        # print(x.shape)
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)
        # x = x.view(-1,1)
        # print('Reformated data shape')
        # print(x.shape)

        # x = F.dropout(x, p=0.8, training=self.training)
        # print('after first dropout:')
        # print(x.shape, edge_index.shape)
        # print(torch.max(edge_index))
        if self.edge_weights:
            # print(edge_index.type())
            # print(edge_weight.type())
            # print(x.type())
            # print('Passing through Conv1 layer with edge_weight.')
            x = self.conv1(x, edge_index, edge_weight)

            x = F.relu(x)

            ## if want to use self-attention pooling layer after conv layer 1
            if self.self_attention:
                ## pass through the self-attention layer
                # print('Passing through the self-attention layer 1')
                # print('with edge weight')
                # print(x.shape)
                # print(edge_index.shape)
                # print(edge_weight.shape)
                # print(self.create_batch_index(batches).shape)
                x, edge_index, edge_weight, batch, _ = self.pool1(x, edge_index, edge_weight, self.create_batch_index(batches))
                x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        else:
            # print('Passing through Conv1 layer without edge_weight.')
            x = self.conv1(x, edge_index)

            x = F.relu(x)

            if self.self_attention:
                ## pass through the self-attention layer 1
                # print('Passing through the self-attention layer 1')
                # print('without edge weight')
                # print(x.shape) ## [batch * num_node, num_hid]
                # print(edge_index.shape) ## [2, edges in one graph * batch]
                # print(edge_weight.shape) ## []
                # print(self.create_batch_index(batches).shape) ## [batch * num_node]
                x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, self.create_batch_index(batches))
                x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
                # print(x1.shape) ## [?, 2 * num_hid]
                # print(edge_index.shape) ## [2, poolrate*]
                # print(batch.shape)
        
        # x = F.dropout(x, p=0.6, training=self.training)

        # print(x.shape)
        if self.edge_weights:
            # print('Passing through Conv2 layer with edge_weight.')
            x = self.conv2(x, edge_index, edge_weight)

            x = F.relu(x)
            
            if self.self_attention:
                ## pass through the self-attention layer
                # print('Passing through the self-attention layer 2')
                # print('with edge weight')
                x, edge_index, edge_weight, batch, _ = self.pool2(x, edge_index, edge_weight, batch)
                x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        else:
            # print('Passing through Conv2 layer without edge_weight.')
            x = self.conv2(x, edge_index) ## output shape: [batches * num_node, hid * head]

            x = F.relu(x)

            if self.self_attention:
                ## pass through the self-attention layer
                # print('Passing through the self-attention layer 2')
                # print('without edge weight')
                x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
                x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
                # print(x2.shape)
                # print(edge_index.shape)
                # print(batch.shape)
        # print(x.shape)

        ## pooling on the graph to reduce nodes
        # print(x.shape)
        if self.self_attention:
            ## if use self-attention pooling then the output has the dimension: [unmasked_nodes, hid * head * 2]
            x = x1 + x2
        else:
            x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
            x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                        ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]
        # print(x.shape)

        if self.method == 'gcn':
            if self.self_attention:
                ## x has the feature dimension of 2 * self.hid as the GMP and GAP concatnation
                x = x.view(-1, 2 * self.hid)
            x = x.view(-1, self.hid) ## output shape:[batches * floor(num_node / poolsize), hid]
        # print(x.shape)

        # x = self.conv3(x, edge_index)
        # x = F.elu(x)
        # print(x.shape)

        # batch_index = np.array(range(batches))
        # batch_index = np.repeat(batch_index, num_genes+100)
        # batch_index = torch.from_numpy(batch_index).to(device)
        # x = global_mean_pool(x, batch_index)
        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        # print(x.shape)
        x = self.linear1(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.linear2(x)
        x = F.relu(x)
        # print(x.shape)
        # x = self.linear3(x)
        # print(x.shape)

        if self.decoder:
            # print('Passing decoder')
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)

            x_reconstruct  = nn.Dropout(0.2)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)

        if self.parallel:
            ## the two layer shallow FC network
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)
            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)

            x = torch.cat((x,x_parallel),1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                # print(x_reconstruct.shape)
                # print(x_target_flatten.shape)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0
        
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss = 1*loss1 + 1*loss2
        
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss += 0.2* l2_regularization* l2_loss
        return loss


class Baseline(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    self_attention,
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(Baseline, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.self_attention = self_attention
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)

        parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

        self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//2)
        self.parallel_linear2 = nn.Linear(parallel_input//2, parallel_input//4)
        self.parallel_linear3 = nn.Linear(parallel_input//4, self.concate_layer)
        # self.parallel_linear3 = nn.Linear(parallel_input//4, parallel_input//8)
        # self.parallel_linear4 = nn.Linear(parallel_input//8, self.concate_layer)
        self.classifier = nn.Linear(self.concate_layer, num_classes)
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):

        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]
            # print(x_cnv.shape)
            # print(x_exp.shape)
            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            # print(x_mirna.shape)
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            # print(x_gene.shape)
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)
            # print(x_gene.shape)

            # print(x_mirna.shape)
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)
            # print(x_mirna.shape)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        x_parallel = x_parallel.view(batches,-1)
        
        x_parallel = self.parallel_linear1(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear2(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear3(x_parallel)
        x_parallel = F.relu(x_parallel)

        x_parallel = F.dropout(x_parallel, p=self.dropout_rate, training=self.training)
        x_parallel = self.classifier(x_parallel)
        return F.log_softmax(x_parallel, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        # if self.num_mirna == 0 or self.num_features == 1:
        #     x_target = x_target.view(x_target.size()[0], -1)
        #     loss1 = nn.MSELoss()(x_reconstruct, x_target)
        # else:
        #     x_target_exp_mirna = x_target[:,:,0]
        #     x_target_cnv = x_target[:,:,1]

        #     ## separate mirna from the rest
        #     x_target_cnv = x_target_cnv[:,:-100]
        #     x_target_exp = x_target_exp_mirna[:,:-100]
        #     x_target_mirna = x_target_exp_mirna[:,-100:]
        #     x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
        #     loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        # loss = 1*loss1 + 1*loss2
        loss = 1*loss2
        
        # if self.l2:
        #     l2_loss = 0.0
        #     for param in self.parameters():
        #         data = param* param
        #         l2_loss += data.sum()

        #     loss += 0.2* l2_regularization* l2_loss
        return loss