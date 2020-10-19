import torch_geometric
import torch
import torch.nn as nn

class GCNConv(torch_geometric.nn.MessagePassing):
    """
    Copied (with minor changes) from the example
        in https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = torch_geometric.utils.degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 6: Return new node embeddings.
        return aggr_out

class MultiGCNConv(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(dim, dim) for _ in range(layers)])
        self.conv_activation = nn.Sigmoid()

    def forward(self, vertices, edges):
        edges = [(u, v) for u, v, _ in edges]
        edges = torch.tensor(edges).T
        if vertices.is_cuda:
            edges = edges.cuda()
        for conv in self.convs[:-1]:
            vertices = conv(vertices, edges)
            vertices = self.conv_activation(vertices)
        if len(self.convs) > 0:
            vertices = self.convs[-1](vertices, edges)
        return vertices

class GGCN(nn.Module):
    """
    Gated Graph Convnet as described in https://arxiv.org/pdf/1711.00740.pdf
    """
    def __init__(self, dim, valid_edge_types, n_steps):
        super().__init__()
        self.dim = dim
        self.n_edge_types = len(valid_edge_types)

        if valid_edge_types is not None:
            assert len(set(valid_edge_types)) == len(valid_edge_types), "edge types should be unique"
            self.edge_type_embedding = {k: i for i, k in
                                        enumerate(sorted(valid_edge_types, key=repr))}
        else:
            self.edge_type_embedding = None

        self.n_steps = n_steps

        self.emb_to_message = nn.Linear(self.dim, self.dim * self.n_edge_types)
        self.gru_cell = nn.GRUCell(self.dim, self.dim)

    def embed_edge(self, e):
        if self.edge_type_embedding is not None:
            return self.edge_type_embedding[e]
        return 0

    def forward(self, embedding, edges):
        edges = [(u, v, self.embed_edge(e)) for u, v, e in edges]
        edges = torch.tensor(edges).T
        if embedding.is_cuda:
            edges = edges.cuda()
        for _ in range(self.n_steps):
            embedding = self.step(embedding, edges)
        return embedding

    def step(self, embedding, edges):
        """
        m[n][e] == the nth vertex's eth edge type's emdedding

        q = [m[u][e] : (u, v, e) in edges]

        r[i] == the ith vertex's result emdedding

        r[v] = g({m[u][e] : (u, v, e) in edges})
             = g({q[i] : edges[1][i]})

        r = 0
        r.scatter_add_(0, index, q)

        where index[i][k] = ev[i]
        """
        assert len(embedding.shape) == 2
        assert len(edges.shape) == 2 and edges.shape[0] == 3
        e_v = edges[1]
        e_ue = edges[[0, 2]]

        m = self.emb_to_message(embedding)
        m = m.reshape(m.shape[0], self.n_edge_types, self.dim)

        q = m[e_ue.tolist()]

        index = e_v.unsqueeze(-1).repeat(1, self.dim)

        r = torch.zeros_like(embedding)
        r.scatter_add_(0, index, q)

        new_embedding = self.gru_cell(r, embedding)

        return new_embedding