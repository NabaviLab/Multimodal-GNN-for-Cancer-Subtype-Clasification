from tokenize import Double
import numpy as np
import pandas as pd
import scipy.sparse as sp
from random import sample
import torch
import argparse

## old data loading functions from the gnn model
def high_variance_expression_gene(expression_variance_path, non_null_path, num_gene, singleton=False):
    ## filter gene list by expression variance
    gene_variance = pd.read_csv(expression_variance_path, sep='\t', index_col=0, header=0)

    if singleton:
        non_null_row = pd.read_csv(non_null_path, sep=',', header=0)
        gene_variance['id'] = range(gene_variance.shape[0])
#         print(non_null_row['gene'])
        gene_variance_non_null = gene_variance.loc[gene_variance.index.isin(non_null_row['gene']),:]
        gene_list = gene_variance_non_null.nlargest(num_gene, 'variance').index
#         print(gene_list)
        gene_variance_non_null.index = gene_variance_non_null['id']
        gene_list_index = gene_variance_non_null.nlargest(num_gene, 'variance').index
    else:
        ## load expression data
        # print(gene_variance['variance'])
        gene_list = gene_variance.nlargest(num_gene, 'variance').index
        gene_variance.index = range(gene_variance.shape[0])
        gene_list_index = gene_variance.nlargest(num_gene, 'variance').index
    return gene_list, gene_list_index

def get_mirna_inner_connection(mirna_connection):
    mirna_connection_row = []
    mirna_connection_col = []
    for i in range(mirna_connection.shape[0]):
        current_indices = np.nonzero(mirna_connection[i,])
        # print(current_indices)
        column_indexes = current_indices[1]
        if len(column_indexes) > 1:
            for x in range(len(column_indexes)):
                for y in range(x, len(column_indexes)):
                    mirna_connection_row += [column_indexes[x]]
                    mirna_connection_col += [column_indexes[y]]
    
    mirna_connection_data = [1] * len(mirna_connection_row)
    mirna_connection_adj = sp.csr_matrix((mirna_connection_data, (mirna_connection_row, mirna_connection_col)),shape=(mirna_connection.shape[1], mirna_connection.shape[1]))
    mirna_index = mirna_connection_adj.nonzero()
    mirna_row = mirna_index[0]
    mirna_col = mirna_index[1]
    mirna_data = [1] * len(mirna_row)
    mirna_adj = sp.csr_matrix((mirna_data, (mirna_row, mirna_col)),shape=(mirna_connection.shape[1], mirna_connection.shape[1]))
    return(mirna_adj.toarray())


def load_exp_and_real_mirna_data(expression_data_path, mirna_data_path):
    ## load multi-omics data
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    mirna_data = pd.read_csv(mirna_data_path, sep='\t', index_col=0, header=0)
    return expression_data, mirna_data

def load_exp_cnv_and_real_mirna_data(expression_data_path, cnv_data_path, mirna_data_path):
    ## load multi-omics data
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    cnv_data = pd.read_csv(cnv_data_path, sep='\t', index_col=0, header=0)
    cnv_data = cnv_data.drop(['icluster_cluster_assignment','sample'], axis=1)
    cnv_data = (cnv_data - cnv_data.min().min()) / (cnv_data.max().max() - cnv_data.min().min())
    mirna_data = pd.read_csv(mirna_data_path, sep='\t', index_col=0, header=0)
    return expression_data, cnv_data, mirna_data

def down_sampling_exp_and_real_mirna_data(expression_variance_path, 
                                            expression_data, 
                                            mirna_data, 
                                            omic_mode, 
                                            non_null_index_path, 
                                            shuffle_index_path, 
                                            adjacency_matrix_path, 
                                            mirna_to_gene_matrix_path, 
                                            gene_gene, 
                                            mirna_gene, 
                                            mirna_mirna, 
                                            number_gene,  
                                            singleton=False):
    ## obtain high varaince gene list
    high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(expression_variance_path, non_null_index_path, number_gene, singleton)

    ## get labels before filtering columns
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1
    
    if expression_data.shape[0] == mirna_data.shape[0]:
        print('Exp and miRNA sample numbers match.')
    # print(mirna_influence_data.shape)
    ## filter multi-omics data by gene list
    expression_data = expression_data.loc[:,high_variance_gene_list]
    expression_data.index = range(expression_data.shape[0])
    
    ## involves mirna data in the following
    if omic_mode > 0:
        mirna_data.index = range(mirna_data.shape[0])
        data = pd.concat([expression_data, mirna_data], axis=1)
    # print(data.shape)
    # print(mirna_influence_data.shape)
    # print(expression_data.shape)
        num_samples = mirna_data.shape[0]
    ## concatenate expr and mirna
        data = np.asarray(data).reshape(num_samples, -1 ,1)
        print(data.shape)
    # data = np.concatenate([data, np.asarray(expression_data).reshape(num_samples, -1, 1) ], axis = 2)
    else:
        data = np.asarray(expression_data).reshape(expression_data.shape[0], -1 ,1)

    ## load adjacency matrix
    if gene_gene:
        gene_gene_adj = sp.load_npz(adjacency_matrix_path)
        gene_gene_adj_mat = gene_gene_adj.todense()
        gene_gene_adj_mat = gene_gene_adj_mat / (gene_gene_adj_mat.max() - gene_gene_adj_mat.min())
        gene_gene_adj_selected = gene_gene_adj_mat[high_variance_gene_index,:]
        gene_gene_adj_selected = gene_gene_adj_selected[:,high_variance_gene_index]
    else:
        gene_gene_adj_selected = np.identity(number_gene)
    
    ## load mirna_to_gene matrix
    if mirna_gene or mirna_mirna:
        mirna_gene_adj = sp.load_npz(mirna_to_gene_matrix_path)
        mirna_gene_adj = mirna_gene_adj.todense()
        # gene_gene_adj_selected = edge_filter(gene_gene_adj_selected)
        # print(gene_gene_adj_selected.shape)
        mirna_gene_adj_selected = mirna_gene_adj[high_variance_gene_index,:]
        # print(mirna_gene_adj_selected.shape)
        # mirna_gene_adj_selected = edge_filter(np.transpose(mirna_gene_adj_selected))
        # mirna_gene_adj_selected = np.transpose(mirna_gene_adj_selected)
    else:
        mirna_gene_adj_selected = np.zeros((number_gene,100))

    if omic_mode == 2:
        top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
        if mirna_mirna:
            print(mirna_gene_adj_selected.shape)
            print(get_mirna_inner_connection(mirna_gene_adj_selected).shape)
            bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
        else:
            bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), np.identity(100)), axis=1)
    
        supra_adj = np.concatenate((top_supra_adj, bottom_supra_adj), axis=0)

        ## convert the dense matrix back to sparse matrix
        supra_adj = sp.csr_matrix(supra_adj)

        if singleton:
            print('including singleton')
            supra_adj = supra_adj + sp.eye(supra_adj.shape[0])
    elif omic_mode == 0:
        supra_adj = sp.csr_matrix(gene_gene_adj_selected)
    elif omic_mode == 1:
        supra_adj = sp.csr_matrix(get_mirna_inner_connection(mirna_gene_adj_selected))


    # del features['iCluster']
    shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    # print(shuffle_index.shape)
    
    return supra_adj, np.asarray(data), labels.to_numpy(), shuffle_index.to_numpy()

def down_sampling_exp_cnv_and_real_mirna_data(expression_variance_path, 
                                            expression_data, 
                                            cnv_data,
                                            mirna_data, 
                                            omic_mode, 
                                            non_null_index_path, 
                                            shuffle_index_path, 
                                            adjacency_matrix_path, 
                                            mirna_to_gene_matrix_path, 
                                            gene_gene, 
                                            mirna_gene, 
                                            mirna_mirna, 
                                            number_gene,  
                                            singleton=False):
    ## obtain high varaince gene list
    high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(expression_variance_path, non_null_index_path, number_gene, singleton)

    ## get labels before filtering columns
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1
    
    if expression_data.shape[0] == mirna_data.shape[0]:
        print('Exp and miRNA sample numbers match.')
    # print(mirna_influence_data.shape)
    ## filter multi-omics data by gene list
    expression_data = expression_data.loc[:,high_variance_gene_list]
    cnv_data = cnv_data.loc[:,high_variance_gene_list]
    expression_data.index = range(expression_data.shape[0])
    mirna_data.index = range(mirna_data.shape[0])
    cnv_data.index = range(cnv_data.shape[0])
    # print(expression_data.shape)
    # print(mirna_data.shape)

    ##  only pad CNV data when using Exp, CNV and miRNA
    if omic_mode == 4:
        cnv_padding = pd.DataFrame(np.zeros((mirna_data.shape[0],mirna_data.shape[1])))
        cnv_data_padded = pd.concat([cnv_data, cnv_padding], axis=1)

        data = pd.concat([expression_data, mirna_data], axis=1)
        
        # print(data.shape)
        # print(mirna_influence_data.shape)
        # print(expression_data.shape)
        num_samples = mirna_data.shape[0]
        ## concatenate expr and mirna
        data= np.asarray(data).reshape(num_samples, -1 ,1)
        cnv_data_padded = np.asarray(cnv_data_padded).reshape(num_samples, -1 ,1)
        data = np.concatenate([data,cnv_data_padded], axis=2)
        print(data.shape)
    else:
        num_samples = expression_data.shape[0]
        data = np.array(expression_data).reshape(num_samples, -1 ,1)
        print(data.shape)
        cnv_data = np.asarray(cnv_data).reshape(num_samples, -1 ,1)
        print(data.shape)
        data = np.concatenate([data,cnv_data], axis=2)
        print(data.shape)
    # data = np.concatenate([data, np.asarray(expression_data).reshape(num_samples, -1, 1) ], axis = 2)

    ## load adjacency matrix
    if gene_gene:
        gene_gene_adj = sp.load_npz(adjacency_matrix_path)
        gene_gene_adj_mat = gene_gene_adj.todense()
        gene_gene_adj_mat = gene_gene_adj_mat / (gene_gene_adj_mat.max() - gene_gene_adj_mat.min())
        gene_gene_adj_selected = gene_gene_adj_mat[high_variance_gene_index,:]
        gene_gene_adj_selected = gene_gene_adj_selected[:,high_variance_gene_index]
    else:
        gene_gene_adj_selected = np.identity(number_gene)
    
    ## load mirna_to_gene matrix
    if mirna_gene or mirna_mirna:
        mirna_gene_adj = sp.load_npz(mirna_to_gene_matrix_path)
        mirna_gene_adj = mirna_gene_adj.todense()
        # gene_gene_adj_selected = edge_filter(gene_gene_adj_selected)
        # print(gene_gene_adj_selected.shape)
        mirna_gene_adj_selected = mirna_gene_adj[high_variance_gene_index,:]
        # print(mirna_gene_adj_selected.shape)
        # mirna_gene_adj_selected = edge_filter(np.transpose(mirna_gene_adj_selected))
        # mirna_gene_adj_selected = np.transpose(mirna_gene_adj_selected)
    else:
        mirna_gene_adj_selected = np.zeros((number_gene,100))

    if omic_mode > 3:
        if mirna_mirna:
            print(mirna_gene_adj_selected.shape)
            print(get_mirna_inner_connection(mirna_gene_adj_selected).shape)
            if mirna_gene:
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
            else:
                top_supra_adj = np.concatenate((gene_gene_adj_selected, np.zeros((number_gene,100))), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(np.zeros((number_gene,100))), get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
        else:
            if mirna_gene:
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), np.identity(100)), axis=1)
            else:
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), np.identity(100)), axis=1)
        
        supra_adj = np.concatenate((top_supra_adj, bottom_supra_adj), axis=0)

        ## convert the dense matrix back to sparse matrix
        supra_adj = sp.csr_matrix(supra_adj)
    elif omic_mode == 3:
        supra_adj = sp.csr_matrix(gene_gene_adj_selected)

    if singleton:
        print('including singleton')
        supra_adj = supra_adj + sp.eye(supra_adj.shape[0])

    # del features['iCluster']
    shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    # print(shuffle_index.shape)
    
    return supra_adj, np.asarray(data), labels.to_numpy(), shuffle_index.to_numpy()

def dropout_data(data, labels, drop_out=0.6):
    dropout_index = sample(range(len(labels)), round(len(labels)*drop_out))
    dropped_data = data[dropout_index,:]
    dropped_labels = labels[dropout_index]
    return dropped_data, dropped_labels

def accuracy(output, labels): # average of each batch 
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def edge_filter(adj, top=10, which_axis=1):
    adj = np.array(adj)
    print(adj.shape)
    new_adj = adj * (np.argsort(np.argsort(adj,axis=which_axis)) >= adj.shape[1] - top)
    return(new_adj)

def disassemble_edge_weights(edge_weights, edge_index, num_gene, num_attributes):
    edge_index_transposed = edge_index.T
    edge_attributes = np.zeros((edge_index.shape[1], num_attributes))
    for idx, x in enumerate(edge_index_transposed):
        # print(x)
        if x[0] < num_gene and x[1] < num_gene: ## gene-gene connection
            edge_attributes[idx,0] = edge_weights[idx]
        elif x[0] < num_gene and x[1] >= num_gene: ## mirna-gene connection
            edge_attributes[idx,1] = edge_weights[idx]
        elif x[0] >= num_gene and x[1] < num_gene: ## mirna-gene connection
            edge_attributes[idx,1] = edge_weights[idx]
        else: ## mirna-mirna connections
            edge_attributes[idx,0] = edge_weights[idx]
    return(torch.Tensor(edge_attributes))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def omic_mode_translation(omic_mode):
    if omic_mode == 0:
        print('Using Expression Data Only.')
        return 1
    elif omic_mode == 1:
        print('Using miRNA Data Only.')
        return 1
    elif omic_mode == 2:
        print('Using Expression and miRNA Data.')
        return 2
    elif omic_mode == 3:
        print('Using Expression and CNV Data.')
        return 2
    elif omic_mode == 4:
        print('Using Expression, CNV and miRNA Data.')
        return 3
    
def validate_network_choice(omic_mode, gene_gene, mirna_gene, mirna_mirna):
    if omic_mode == 0:
        if mirna_gene or mirna_mirna:
            print('miRNA-Gene or miRNA-miRNA network not available when only using expression data.')
            return True, False, False, 0
        else:
            return gene_gene, mirna_gene, mirna_mirna, 0
    elif omic_mode == 1:
        if gene_gene:
            print('Gene-Gene not available when only using miRNA data.')
            return False, True, True, 100
        else:
            return gene_gene, mirna_gene, mirna_mirna, 100
    elif omic_mode == 3:
        if mirna_gene or mirna_mirna:
            print('miRNA-Gene or miRNA-miRNA network not available when using expression and CNV data.')
            return True, False, False, 0
        else:
            return gene_gene, mirna_gene, mirna_mirna, 0
    else:
        return gene_gene, mirna_gene, mirna_mirna, 100

def filter_data_by_cancer_type(cancer_subtype_label_path, data, expression_data):
    cancer_subtype_label = pd.read_csv(cancer_subtype_label_path, sep=',', header=0)
    cancer_subtype_label = cancer_subtype_label[cancer_subtype_label['patient'].isin(expression_data['sample'].tolist())]
    expression_index = expression_data[expression_data['sample'].isin(cancer_subtype_label['patient'].tolist())]['sample']

    ## check if the order of the patient is the same
    for i in range(cancer_subtype_label.shape[0]):
        if expression_index.iloc[i] != cancer_subtype_label.iloc[i,0]:
            print('Mismatch in patients!')
            quit()

    subtype_sample_index = expression_data['sample'].isin(cancer_subtype_label['patient'].tolist())

    data = data[subtype_sample_index]
    labels = cancer_subtype_label['subtype']

    return np.asarray(data), labels.to_numpy()
