import torch
from torch import nn
import torch.nn.functional as f
from dgl.nn.pytorch import GATConv, GraphConv, GatedGraphConv

from data_loader.dataset import DataEntry
from data_loader.batch_graph import GGNNBatchGraph
from .layers import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class GGNAT(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_heads=3, num_steps=8, pool_layer=2):
        super(GGNAT, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.pool_layer = pool_layer
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
                                   
        self.attention_layer = GATConv(input_dim, output_dim, num_heads, 0.5, 0.2)
        
        self.concat_dim = output_dim + output_dim

        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.avg_conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.avgpool1 = torch.nn.AvgPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.avg_conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)
        self.avgpool2 = torch.nn.AvgPool1d(2, stride=2)


        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.avg_conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.avgpool1_for_concat = torch.nn.AvgPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.avg_conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)
        self.avgpool2_for_concat = torch.nn.AvgPool1d(2, stride=2)
    

        self.mlp_h = nn.Linear(in_features=output_dim, out_features=1) 
        self.mlp = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()


    def pooling(self, h_i, c_i):

        Y_1 = self.maxpool1(f.relu(self.conv_l1(h_i.transpose(1, 2))))
        Y_2 = self.maxpool2(f.relu(self.conv_l2(Y_1))).transpose(1, 2)
        
        P_1 = self.avgpool1(f.relu(self.avg_conv_l1(h_i.transpose(1, 2))))
        P_2 = self.avgpool2(f.relu(self.avg_conv_l2(P_1))).transpose(1, 2)

        Z_1 = self.maxpool1_for_concat(f.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(f.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)

        U_1 = self.avgpool1_for_concat(f.relu(self.avg_conv_l1_for_concat(c_i.transpose(1, 2))))
        U_2 = self.avgpool2_for_concat(f.relu(self.avg_conv_l2_for_concat(U_1))).transpose(1, 2)

        before_avg1 = torch.mul(self.mlp_h(Y_2), self.mlp(Z_2)).mean(dim=1)
        before_avg2 = torch.mul(self.mlp_h(P_2), self.mlp(U_2)).mean(dim=1)
        avg = torch.cat([before_avg1, before_avg2], dim=1) 
        return avg


    def forward(self, batch, cuda=True):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(device)
        
        out1 = self.ggnn(graph, features, edge_types)     
        out1 = self.attention_layer(graph, out1)   
       
        h_i, _ = batch.de_batchify_graphs(out1)
        h_i = h_i.mean(dim=2)

        x_i, _ = batch.de_batchify_graphs(features)
        c_i = torch.cat((h_i, x_i), dim=-1)            
        res= self.pooling(h_i, c_i)
        return self.sigmoid(res.mean(dim=-1))
       
    
    def get_embedding(self, data_entry: DataEntry):
        graph = GGNNBatchGraph()
        graph.add_subgraph(data_entry.graph)
        
    
    def predict_prob(self, data_entry: DataEntry):
        graph = GGNNBatchGraph()
        graph.add_subgraph(data_entry.graph)
        prediction = self(graph, cuda=True)
        return prediction

    
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.gcn1 = GraphConv(in_feats=input_dim, out_feats=input_dim)               
        self.concat_dim = output_dim + output_dim

        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.avg_conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.avgpool1 = torch.nn.AvgPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.avg_conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)
        self.avgpool2 = torch.nn.AvgPool1d(2, stride=2)

        self.conv_l3 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool3 = torch.nn.MaxPool1d(2, stride=2)
        self.avgpool3 = torch.nn.AvgPool1d(2, stride=2)

        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.avg_conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.avgpool1_for_concat = torch.nn.AvgPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.avg_conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)
        self.avgpool2_for_concat = torch.nn.AvgPool1d(2, stride=2)

        self.mlp_h = nn.Linear(in_features=output_dim, out_features=1) 
        self.mlp = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.ln = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def pool_2layer(self, h_i, c_i):

        Y_1 = self.maxpool1(f.relu(self.conv_l1(h_i.transpose(1, 2))))
        Y_2 = self.maxpool2(f.relu(self.conv_l2(Y_1))).transpose(1, 2)
        
        P_1 = self.avgpool1(f.relu(self.avg_conv_l1(h_i.transpose(1, 2))))
        P_2 = self.avgpool2(f.relu(self.avg_conv_l2(P_1))).transpose(1, 2)

        Z_1 = self.maxpool1_for_concat(f.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(f.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)

        U_1 = self.avgpool1_for_concat(f.relu(self.avg_conv_l1_for_concat(c_i.transpose(1, 2))))
        U_2 = self.avgpool2_for_concat(f.relu(self.avg_conv_l2_for_concat(U_1))).transpose(1, 2)

        before_avg1 = torch.mul(self.mlp_h(Y_2), self.mlp(Z_2)).mean(dim=1)
        before_avg2 = torch.mul(self.mlp_h(P_2), self.mlp(U_2)).mean(dim=1)
        before_avg = torch.cat([before_avg1, before_avg2], dim=1)
        return before_avg

    def forward(self, batch, cuda=True):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(device)
        outputs = self.gcn1(graph, features)
        h_i, _ = batch.de_batchify_graphs(outputs)
        x_i, _ = batch.de_batchify_graphs(features)
        c_i = torch.cat((h_i, x_i), dim=-1)

        res = self.pool_2layer(h_i, c_i)
        return self.sigmoid(res.mean(dim=-1))
        


    

        


