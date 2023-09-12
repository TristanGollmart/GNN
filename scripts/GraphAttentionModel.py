import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW

device = "cpu"

def make_adjacency(graph):
    n_nodes = len(graph)
    adj = np.zeros((n_nodes, n_nodes))

    ix_edge_out = [str_to_ix[c[0]] for c in list(graph.edges)]
    ix_edge_in = [str_to_ix[c[1]] for c in list(graph.edges)]

    for (i, j) in zip(ix_edge_out, ix_edge_in):
        adj[i, j] = 1
    # for i in range(n_nodes):
    #    adj[i, [key for key in x[i].keys()]] = 1
    assert np.sum(adj) == len(graph.edges)
    # add self-connection
    for i in range(n_nodes):
        adj[i, i] = 1
    return adj

def getState(graph):
    # same order as adjacency give by ix_to_str
    target_list = [graph._node[ix_to_str[ix]]['value'] for ix in range(len(graph))]
    return target_list

class GraphAttentionLayer(nn.Module):
    def __init__(self, adj, n_features, n_embd):
        super(GraphAttentionLayer, self).__init__()
        self.n_features = n_features
        self.n_embd = n_embd
        self.A = torch.tensor(adj)
        self.degree = torch.sum(self.A, dim=1)
        self.D = torch.diag(1/torch.sqrt(self.degree))
        self.A_hat = self.D @ self.A @ self.D

        self.key = nn.Linear(n_features, n_embd)
        self.query = nn.Linear(n_features, n_embd)
        self.value = nn.Linear(n_features, n_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        k = self.key(x)   # (N, d_embd)
        q = self.query(x) # (N,d_embd)
        v = self.value(x) # (N, d)
        wei = q @ k.transpose(-2, -1) # (N, N)
        wei = wei.masked_fill(self.A_hat==0, 0)
        out = wei @ v # (N, d)
        return self.relu(out)

class GAT(nn.Module):
    def __init__(self, adj, n_features, n_embd, n_classes):
        super(GAT, self).__init__()
        self.n_features = n_features
        self.n_embd = n_embd
        self.n_classes = n_classes
        self.adj = adj
        self.gcn = GraphAttentionLayer(adj, n_features, n_embd)
        self.ffd = nn.Linear(n_features, n_classes)

    def forward(self, x_in):
        # return n_classes logits
        x = self.gcn(x_in)
        logits = self.ffd(x)
        loss = F.cross_entropy(logits, x_in)
        return logits, loss


graph = nx.read_gml(r'..\data\football\football.gml')
str_to_ix = {str: ix for ix, str in enumerate(graph.nodes)}
ix_to_str = {ix: str for ix, str in enumerate(graph.nodes)}
state_dict = {}


adj = make_adjacency(graph)
x = getState(graph)
nClasses = len(np.unique(x))
x_ohe = F.one_hot(torch.tensor(x)).to(torch.float)

lr = 1e-3
max_iter = 400
model =GAT(adj, n_features=x_ohe.shape[-1], n_embd=4, n_classes=nClasses)
optimizer = AdamW(params=model.parameters(), lr=lr)

for iter in range(max_iter):
    x_pred, loss = model(x_ohe)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 100 == 0 or iter == max_iter-1:
        print(f'Iteration {iter}, loss: {loss:.4f}')

# evaluate
x_out, _ = model(x_ohe)
for i, node in enumerate(x_out.tolist()):
    print(f'predict {np.argmax(node)}, true: {x[i]}')
    single_loss = F.cross_entropy(torch.tensor([node]), torch.tensor([x[i]]))
print('finished')