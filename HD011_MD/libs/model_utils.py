import dgl
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance_matrix


def join_graph(early_stage_graph:dict, coordinate_list:map):

    graph_p_list = dgl.unbatch(early_stage_graph['graph_p'].to('cpu'))
    graph_l_list = dgl.unbatch(early_stage_graph['graph_l'].to('cpu'))

    if len(graph_p_list) == len(graph_l_list):
        batch_size = len(graph_p_list)
    else:
         raise ValueError('Have to match the number of graphs about protein(pocket) and ligand')

    joined_graph_list = []
    new_edge_num_list = []

    for idx in range(batch_size):
        joined_graph = dgl.batch([graph_p_list[idx], graph_l_list[idx]], ndata=['h'], edata=['e_ij'])
        start_idx_l = joined_graph._batch_num_nodes['_N'].detach().cpu().numpy()[0]
        coord_p = np.array(next(coordinate_list[idx]))
        coord_l = np.array(next(coordinate_list[idx]))

        distance = distance_matrix(coord_p, coord_l)
        distance_idx = np.where(distance<5)
        l_atom_idx = distance_idx[0] + start_idx_l    # src
        p_atom_idx = distance_idx[1]      # dst

        joined_graph = dgl.to_homogeneous(joined_graph, ndata=['h'], edata=['e_ij'])
        joined_graph_tmp = dgl.to_homogeneous(joined_graph, ndata=['h'], edata=['e_ij'])

        new_edge_num = distance_idx[0].shape[0]
        new_edge_num_list.append(new_edge_num)
        bond_feature_list = joined_graph.edata['e_ij'].detach().cpu().numpy()
        bond_feature_vdw_list = np.array([[2.5] for _ in range(bond_feature_list.shape[0])])

        for i in range(new_edge_num):
            src = p_atom_idx[i]
            dst = l_atom_idx[i]

            layer = nn.Linear(6, 64, bias=False)
            bond_feature = layer(torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float64).float())
            bond_feature = bond_feature.detach().cpu().numpy()
            bond_feature_vdw = np.array([distance[dst - start_idx_l, src]]).reshape(1,1)

            joined_graph.add_edges(src, dst)
            bond_feature_list = np.concatenate((bond_feature_list,bond_feature), axis=0)
            bond_feature_vdw_list = np.concatenate((bond_feature_vdw_list, bond_feature_vdw), axis=0)

            joined_graph.add_edges(dst, src)
            bond_feature_list = np.concatenate((bond_feature_list,bond_feature), axis=0)
            bond_feature_vdw_list = np.concatenate((bond_feature_vdw_list, bond_feature_vdw), axis=0)

        bond_feature_list = torch.tensor(bond_feature_list, dtype=torch.float64)
        bond_feature_vdw_list = torch.tensor(bond_feature_vdw_list, dtype=torch.float64)

        joined_graph.edata['e_ij'] = bond_feature_list
        joined_graph.edata['vdw'] = bond_feature_vdw_list

        joined_graph_list.append(joined_graph)

    joined_graph_list = dgl.batch(joined_graph_list)

    return joined_graph_list




