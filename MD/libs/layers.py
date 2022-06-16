# torch.nn은 클래스, torch.nn.functional은 함수
# torch.nn으로 구현한 클래스의 경우는 attribute를 활용해 state를 저장하고 활용할 수 있다
# torch.nn.functional으로 구현한 함수의 경우는 instance화 시킬 필요 없이 사용이 가능하다
import math

import torch.nn as nn  # basic building bloc for graphs
import torch.nn.functional as F
import dgl.function as fn       # dgl에서 제공하는 모든 내장함수 기능을 호스팅

from dgl.nn.functional import edge_softmax      # pytorch관련 NN모듈용 패키지


class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            bias=True,
            act=F.relu,
        ):
        # super()로 기반클래스(nn.Module)초기화
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.act = act

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, h):
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        return h


class denseMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_layer,
            output_dim,
            bias=True,
            act=F.relu,
        ):
        # super()로 기반클래스(nn.Module)초기화
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.act = act
        self.linear_list = nn.ModuleList()
        num_layer_mid = int((num_layer - 2) / 2)
        self.linear_list.append(nn.Linear(input_dim, hidden_dim, bias=bias))  # first layer added
        num_layer = num_layer - 3

        hidden_dim_in = int(hidden_dim)
        hidden_dim_out = int(hidden_dim)
        flag_dim_down = 0

        for i in range(num_layer):
            if i == num_layer_mid and flag_dim_down == 0:
                hidden_dim_in = hidden_dim
                hidden_dim_out = int(hidden_dim / 2)
                self.linear_list.append(nn.Linear(hidden_dim_in, hidden_dim_out, bias=bias))
                hidden_dim_in = hidden_dim_out
                flag_dim_down = 1
            if i == num_layer - 1 and flag_dim_down == 1:
                hidden_dim_out = int(hidden_dim_out / 2)
                self.linear_list.append(nn.Linear(hidden_dim_in, hidden_dim_out, bias=bias))
                flag_dim_down = 2
            else:
                self.linear_list.append(nn.Linear(hidden_dim_in, hidden_dim_out, bias=bias))

        self.linear_out = nn.Linear(hidden_dim_out, output_dim, bias=bias)

    def forward(self, h):
        for i, linear_layer in enumerate(self.linear_list):
            h = self.linear_list[i](h)
            h = self.act(h)
        h = self.linear_out(h)

        return h


class GraphConvolution(nn.Module):
    def __init__(
            self,
            hidden_dim,
            act=F.relu,
            dropout_prob=0.2,
        ):
        super().__init__()

        self.act = act
        self.norm = nn.LayerNorm(hidden_dim)
        self.prob = dropout_prob
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
            self,
            graph,
            training=False
        ):
        h0 = graph.ndata['h']

        # fn.copy_u(u, out): source node를 사용하여 메세지를 계산하는 메세지 함수
        # 'u(str)': source feature field
        # 'out(str)': output message field
        # fn.sum(): 합계로 메시지를 집계하는 built-in reduce function
        # update_all: 지정된 유형의 모든 edge에 메세지를 보내고 해당 대상 유형의 모든 노드를 업데이트
        graph.update_all(fn.copy_u('h','m'), fn.sum('m', 'u_'))

        h = self.act(self.linear(graph.ndata['u_'])) + h0
        h = self.norm(h)

        h = F.dropout(h, p=self.prob, training=training)

        graph.ndata['h'] = h
        return graph


class GraphAttention(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads=4,
            bias_mlp=True,
            dropout_prob=0.2,
            act=F.relu,
    ):
        super().__init__()

        self.mlp = MLP(
            input_dim=hidden_dim,
            hidden_dim=2 * hidden_dim,
            output_dim=hidden_dim,
            bias=bias_mlp,
            act=act,
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.splitted_dim = hidden_dim // num_heads

        self.prob = dropout_prob
        self.act = act

        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w6 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            graph,
            training=False
    ):
        h0 = graph.ndata['h']
        e_ij = graph.edata['e_ij']

        graph.ndata['u'] = F.leaky_relu(self.w1(h0).view(-1, self.num_heads, self.splitted_dim))
        graph.ndata['v'] = F.leaky_relu(self.w2(h0).view(-1, self.num_heads, self.splitted_dim))
        graph.edata['x_ij'] = self.w3(e_ij).view(-1, self.num_heads, self.splitted_dim)

        graph.apply_edges(fn.v_add_e('v', 'x_ij', 'm'))  # message passing
        graph.apply_edges(fn.u_mul_e('u', 'm', 'attn'))  # attention
        graph.edata['attn'] = edge_softmax(graph, graph.edata['attn'] / math.sqrt(self.splitted_dim))

        graph.ndata['k'] = F.leaky_relu(self.w4(h0).view(-1, self.num_heads, self.splitted_dim))
        graph.edata['x_ij'] = F.leaky_relu(self.w5(e_ij).view(-1, self.num_heads, self.splitted_dim))
        graph.apply_edges(fn.v_add_e('k', 'x_ij', 'm'))

        graph.edata['m'] = graph.edata['attn'] * graph.edata['m']
        graph.update_all(fn.copy_edge('m', 'm'), fn.sum('m', 'h'))

        h = self.w6(h0) + graph.ndata['h'].view(-1, self.hidden_dim)
        h = self.norm(h)

        # Add and Norm module
        h = h + self.mlp(h)
        h = self.norm(h)

        # Apply dropout on node features
        h = F.dropout(h, p=self.prob, training=training)

        graph.ndata['h'] = h
        return graph
