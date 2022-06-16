import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers import GraphConvolution
from layers import GraphAttention
from layers import denseMLP


class Model(nn.Module):
    def __init__(
            self,
            num_g_layers,
            num_d_layers,
            hidden_dim_g,
            hidden_dim_d,
            dropout_prob,
            readout='sum',
            act_f=F.relu,
            initial_node_dim=58,
            initial_edge_dim=6,
        ):
        super().__init__()

        self.num_g_layers = num_g_layers
        self.embedding_node = torch.nn.ModuleList([nn.Linear(initial_node_dim, hidden_dim_g, bias=False),  # for protein
                                                   nn.Linear(initial_node_dim, hidden_dim_g, bias=False)]) # for ligand
        self.embedding_edge = torch.nn.ModuleList([nn.Linear(initial_edge_dim, hidden_dim_g, bias=False),  # for protein
                                                   nn.Linear(initial_edge_dim, hidden_dim_g, bias=False)]) # for ligand
        self.g_p_layers = torch.nn.ModuleList()
        self.g_l_layers = torch.nn.ModuleList()
        self.readout = readout
        self.act = act_f

        # pocket graph learning layer
        for _ in range(self.num_g_layers):
            g_p_layer = None
            g_p_layer = GraphConvolution(
                hidden_dim=hidden_dim_g,
                dropout_prob=dropout_prob,
                act=act_f, )
            self.g_p_layers.append(g_p_layer)

        # ligand graph learning layer
        for _ in range(self.num_g_layers):
            g_l_layer = None
            g_l_layer = GraphAttention(
                hidden_dim=hidden_dim_g,
                dropout_prob=dropout_prob,
                act=act_f,
            )
            self.g_l_layers.append(g_l_layer)

        # affinity calculation layer
        self.dense_layers = denseMLP(input_dim=hidden_dim_g,
                                     hidden_dim=hidden_dim_d,
                                     num_layer=num_d_layers,
                                     output_dim=1, bias=False)

    def forward(
            self,
            graph_p, graph_l,
            training=False,
        ):
        # protein graph embedding
        h_p = self.embedding_node[0](graph_p.ndata['h'].float())
        e_ij_p = self.embedding_edge[0](graph_p.edata['e_ij'].float())
        # ligand graph embedding
        h_l = self.embedding_node[1](graph_l.ndata['h'].float())
        e_ij_l = self.embedding_edge[1](graph_l.edata['e_ij'].float())

        # embedding 값 할당
        graph_p.ndata['h'] = h_p
        graph_p.edata['e_ij'] = e_ij_p
        graph_l.ndata['h'] = h_l
        graph_l.edata['e_ij'] = e_ij_l

        # protein graph learning layer forward
        for i in range(self.num_g_layers):
            graph_p = self.g_p_layers[i](
                graph=graph_p,
                training=training
            )
        readout_p = dgl.readout_nodes(graph_p, 'h', op=self.readout)

        # ligand graph learning layer forward
        for j in range(self.num_g_layers):
            graph_l = self.g_l_layers[j](
                graph=graph_l,
                training=training
            )
        readout_l = dgl.readout_nodes(graph_l,'h', op=self.readout)

        out_comb = self.dense_layers(readout_p + readout_l)

        return out_comb


