import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from libs.layers import GraphConvolution
from libs.layers import GraphAttention
from libs.layers import GraphAttention_dist
from libs.layers import denseMLP

from libs.model_utils import join_graph


class Model_vectorization(nn.Module):
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
        self.embedding_node = torch.nn.ModuleList([nn.Linear(initial_node_dim, hidden_dim_g, bias=False),
                                                   nn.Linear(initial_node_dim, hidden_dim_g, bias=False)])
        self.embedding_edge = torch.nn.ModuleList([nn.Linear(initial_edge_dim, hidden_dim_g, bias=False),
                                                   nn.Linear(initial_edge_dim, hidden_dim_g, bias=False)])
        self.g_p_layers = torch.nn.ModuleList()
        self.g_l_layers = torch.nn.ModuleList()
        self.readout = readout
        self.act = act_f

        # pocket layer
        for _ in range(self.num_g_layers):
            g_p_layer = None
            g_p_layer = GraphConvolution(
                hidden_dim=hidden_dim_g,
                dropout_prob=dropout_prob,
                act=act_f, )
            self.g_p_layers.append(g_p_layer)

        # ligand layer
        for _ in range(self.num_g_layers):
            g_l_layer = None
            g_l_layer = GraphAttention(
                hidden_dim=hidden_dim_g,
                dropout_prob=dropout_prob,
                act=act_f,
            )
            self.g_l_layers.append(g_l_layer)

        self.dense_layers = denseMLP(input_dim=hidden_dim_g,
                                     hidden_dim=hidden_dim_d,
                                     num_layer=num_d_layers,
                                     output_dim=1, bias=False)

    def forward(
            self,
            graph_p, graph_l,
            training=False,
        ):
        h_p = self.embedding_node[0](graph_p.ndata['h'].float())
        e_ij_p = self.embedding_edge[0](graph_p.edata['e_ij'].float())
        h_l = self.embedding_node[1](graph_l.ndata['h'].float())
        e_ij_l = self.embedding_edge[1](graph_l.edata['e_ij'].float())

        graph_p.ndata['h'] = h_p
        graph_p.edata['e_ij'] = e_ij_p
        graph_l.ndata['h'] = h_l
        graph_l.edata['e_ij'] = e_ij_l

        # pocket layer forward
        for i in range(self.num_g_layers):
            graph_p = self.g_p_layers[i](
                graph=graph_p,
                training=training
            )
        readout_p = dgl.readout_nodes(graph_p, 'h', op=self.readout)

        # ligand layer forward
        for j in range(self.num_g_layers):
            graph_l = self.g_l_layers[j](
                graph=graph_l,
                training=training
            )
        readout_l = dgl.readout_nodes(graph_l,'h', op=self.readout)

        tmp_graph_list = dgl.unbatch(graph_l)

        out_comb = self.dense_layers(readout_p + readout_l)

        return out_comb


class Model_distance(nn.Module):
    def __init__(
            self,
            num_g_layers,
            hidden_dim_g,
            hidden_dim_d,
            dropout_prob,
            readout='sum',
            act_f=F.relu,
            initial_node_dim=58,
            initial_edge_dim=6,  # + bond length
            interact_node_dim=64,
            interact_edge_dim=1, # only bond length feature
    ):
        super().__init__()

        self.num_g_layers = num_g_layers
        self.embedding_node = torch.nn.ModuleList([nn.Linear(initial_node_dim, hidden_dim_g, bias=False),
                                                   nn.Linear(initial_node_dim, hidden_dim_g, bias=False)])
        self.embedding_edge = torch.nn.ModuleList([nn.Linear(initial_edge_dim, hidden_dim_g, bias=False),
                                                   nn.Linear(initial_edge_dim, hidden_dim_g, bias=False)])
        self.embedding_interact_node = nn.Linear(interact_node_dim, hidden_dim_g, bias=False)
        self.embedding_interact_edge = nn.Linear(interact_edge_dim, hidden_dim_g, bias=False)

        self.g_p_layers = torch.nn.ModuleList()
        self.g_l_layers = torch.nn.ModuleList()
        self.g_i_layers = torch.nn.ModuleList()
        self.readout = readout
        self.act = act_f
        self.device = 'cpu'

        # pocket layer
        for _ in range(self.num_g_layers):
            g_p_layer = None
            g_p_layer = GraphConvolution(
                hidden_dim=hidden_dim_g,
                dropout_prob=dropout_prob,
                act=act_f, )
            self.g_p_layers.append(g_p_layer)

        # ligand layer
        for _ in range(self.num_g_layers):
            g_l_layer = None
            g_l_layer = GraphAttention(
                hidden_dim=hidden_dim_g,
                dropout_prob=dropout_prob,
                act=act_f,
            )
            self.g_l_layers.append(g_l_layer)

        # distance calculation
        for _ in range(self.num_g_layers):
            g_i_layer = None
            g_i_layer = GraphAttention_dist(
                hidden_dim=hidden_dim_g,
                dropout_prob=dropout_prob,
                act=act_f,
            )
            self.g_i_layers.append(g_i_layer)

        # interaction learning layer
        self.dense_layers = denseMLP(input_dim=hidden_dim_d,
                                     hidden_dim=hidden_dim_d,
                                     output_dim=1, bias=False)

    def forward(
            self,
            graph_p, graph_l, coordinate,
            training=False,
    ):

        self.device = graph_p.device

        # save early stage of graphs for calculating distance
        early_stage = {'graph_p': graph_p, 'graph_l': graph_l}
        h_p = self.embedding_node[0](graph_p.ndata['h'].float())
        e_ij_p = self.embedding_edge[0](graph_p.edata['e_ij'].float())
        h_l = self.embedding_node[1](graph_l.ndata['h'].float())
        e_ij_l = self.embedding_edge[1](graph_l.edata['e_ij'].float())

        graph_p.ndata['h'] = h_p
        graph_p.edata['e_ij'] = e_ij_p
        graph_l.ndata['h'] = h_l
        graph_l.edata['e_ij'] = e_ij_l

        # pocket layer forward
        for i in range(self.num_g_layers):
            graph_p = self.g_p_layers[i](
                graph=graph_p,
                training=training
            )

        # ligand layer forward
        for j in range(self.num_g_layers):
            graph_l = self.g_l_layers[j](
                graph=graph_l,
                training=training
            )

        # generate joined graph
        joined_graph = join_graph(early_stage, coordinate).to(self.device)

        h_joined = self.embedding_interact_node(joined_graph.ndata['h'].float())
        vdw_joined = self.embedding_interact_edge(joined_graph.edata['vdw'].float())

        joined_graph.ndata['h'] = h_joined
        joined_graph.edata['vdw'] = vdw_joined

        # interaction attention layer forward
        for k in range(self.num_g_layers):
            joined_graph = self.g_i_layers[k](
                graph=joined_graph,
                training=training
            )
        readout = dgl.readout_nodes(joined_graph,'h', op=self.readout)

        # interaction dense layer
        out = self.dense_layers(readout)

        return out