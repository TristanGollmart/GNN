import numpy as np
import torch
import torch.nn as nn
import networkx as nx
device = "cpu"


def get_loss(ypred, y):
    '''
    :param y:
    :param ypred:
    :return:
    '''
    if len(np.unique(y.detach())) == 2:
        loss = nn.BCELoss()
    else:
        # expects logits as predictions!
        loss = nn.CrossEntropyLoss()
    return loss(ypred, y)

def make_adjacency(graph):
    n_nodes = len(graph)
    adj = np.zeros((n_nodes, n_nodes))
    for (i, j) in graph.edges:
        adj[i, j] = 1
    # for i in range(n_nodes):
    #    adj[i, [key for key in x[i].keys()]] = 1
    return adj

class FeedForward(nn.Module):
    def __init__(self, n_in, n_out):
        super(FeedForward, self).__init__()
        self.ffd = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.ReLU(),
            nn.Linear(n_out, n_out),
        )

    def forward(self, x):
        return self.ffd(x)

class GraphConvLayer(nn.Module):
    # implements a graph convolution layer
    def __init__(self, adj, n_features, n_embd):
        # Adjacency matrix A is expected to have self-connection
        super(GraphConvLayer, self).__init__()
        self.A = adj
        self.D_sqr_inv = torch.diag(1/torch.sqrt(torch.sum(self.A, dim=1)))
        self.A_hat = self.D_sqr_inv @ self.A @ self.D_sqr_inv
        self.n_features = n_features
        self.ffd1 = FeedForward(n_features, n_embd)
        self.ffd2 = FeedForward(n_embd, n_features)
        if n_features > 1:
            self.ln1 = nn.LayerNorm(n_features)
            self.ln2 = nn.LayerNorm(n_embd)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
    def forward(self, x):
        x = self.ffd1(self.ln1(x))
        x = self.A_hat @ x # message passing
        x = self.ffd2(self.ln2(x))
        return x

class GraphConvolutionModel(nn.Module):
    # implements a graph convolutional neural network
    def __init__(self, adj, n_features, n_embd=10, return_probs=False):
        super(GraphConvolutionModel, self).__init__()
        self.n_data = np.shape(adj)[0]
        self.A = torch.tensor(adj) + torch.diag(torch.ones(self.n_data))
        self.D_sqr_inv = torch.diag(1/torch.sqrt(torch.sum(self.A, dim=1)))
        self.A_hat = self.D_sqr_inv @ self.A @ self.D_sqr_inv
        self.n_features = n_features
        self.n_embd = n_embd
        self.gcn = GraphConvLayer(self.A_hat, self.n_features, self.n_embd)
        self.relu = nn.ReLU()
        self.return_probs = return_probs

    def forward(self, node_feats):
        x = self.gcn(node_feats)
        # skip connection: x = x + node_feats
        if self.return_probs:
            x = nn.Sigmoid()(x)
        return x

graph = nx.karate_club_graph()
labels = list(set([d['club'] for d in graph._node.values()]))
labels_to_int = {label: i for i, label in enumerate(labels)}

print(np.shape(graph.nodes))
print(graph.nodes)

graph_features = torch.tensor([labels_to_int[d['club']] for d in graph._node.values()], dtype=float) # simply label of which group member belongs to
graph_features = torch.unsqueeze(graph_features, dim=1)
features_normalized = graph_features - torch.mean(graph_features)
adj = make_adjacency(graph)

model = GraphConvolutionModel(adj, n_features=1, n_embd=3, return_probs=True)
model.to(device, dtype=float)

# print parameter
for name, param in model.named_parameters():
    print(name, param)

x_update = model(graph_features)
print(x_update)

# ------------------- TRAINING --------------------------------
max_iters = 500
lr = 5e-3
eval_interval = 100

opt = torch.optim.AdamW(model.parameters(), lr=lr)
for iter in range(max_iters):
    probs = model(features_normalized)
    loss = get_loss(probs, graph_features)
    if iter % eval_interval == 0 or iter == max_iters-1:
        print(f"step {iter}: train loss {loss:.4f}")

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

print(model(features_normalized))