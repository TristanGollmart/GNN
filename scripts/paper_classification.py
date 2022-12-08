from tensorflow import keras
import os
import pandas as pd

# get data
if not os.path.exists("data\\cora.cites"):
    zip_file = keras.utils.get_file(fname='cora.tgz',
                                    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
                                    extract=True,
                                    )
    data_dir = os.path.join(os.path.dirname(zip_file), "cora")
else:
    data_dir = os.path.join(os.getcwd(), "data")

citations_data = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"])

print(citations_data.head())
citations_data.describe()

column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers_data = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep="\t", header=None, names=column_names
)

# process data
print("papers shape:", papers_data.shape)
papers_data.head()

# get the 7 target labels
class_names = sorted(papers_data["subject"].unique())
paper_ids = sorted(papers_data["paper_id"].unique())
class_index = {name: idx for idx, name in enumerate(class_names)}
paper_index = {name: idx for idx, name in enumerate(paper_ids)}

# transform the arbitrary paper ids into successive number
papers_data["paper_id"] = papers_data["paper_id"].apply(lambda name: paper_index[name])
papers_data["subject"] = papers_data["subject"].apply(lambda name: class_index[name])
citations_data["source"] = citations_data["source"].apply(lambda name: paper_index[name])
citations_data["target"] = citations_data["target"].apply(lambda name: paper_index[name])

# visualize the data
import networkx as nx
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
colors = papers_data["subject"].tolist()
cora_graph = nx.from_pandas_edgelist(citations_data.sample(n=1500))
node_subjects = list(cora_graph.nodes)
subjects = list(papers_data[papers_data["paper_id"].isin(node_subjects)]["subject"])
nx.draw_spring(cora_graph, node_size=15,node_color=subjects)
plt.legend()
plt.show()

# transform dataframes into graph structured data:
# Node features
# Edges
# edge weights

import tensorflow as tf
from keras import layers
feature_names = set(papers_data.columns) - {"paper_id", "subject"}
print(feature_names)

edges = citations_data[["source", "target"]].to_numpy().T
print("Edges shape:", edges.shape)

# important: Set fixed order first: by paper-id from min to max
node_features = papers_data.sort_values("paper_id")[feature_names]
node_features = tf.cast(node_features.to_numpy(),dtype=tf.dtypes.float64)

# edge weights: set equal to 1 for each edge
edge_weights = tf.ones(shape=edges.shape[1])
print("edge weights shape:", edge_weights.shape)

# merge into tuple containing all relevant information about graph:
graph_info = (node_features, edges, edge_weights)

# -----------Build GNN-----------
hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

def create_ffn(hidden_units, dropout_rate, name=None):
    # returns sequential moddel, with number of hidden layers = len(hidden_units)
    # with the specified number of hidden_units
    fnn_layers = []
    for unit in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units,activation=tf.nn.gelu))
    return keras.Sequential(fnn_layers, name=name)

class GraphConvLayer(layers.Layer)
    def __init__(self,
                 hidden_units,
                 dropout_rate=0.2,
                 aggregation_type="mean",
                 combination_type="concat",
                 normalize=False,
                 *args,
                 **kwargs):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        # ffn to prepare input features from nodes
        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        # update fn is a predefined feed forward network
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_representations, weights=None):
        # node representations shape is [num_edges, embedding_dim]
        messages = self.ffn_prepare(node_representations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_representations):
        # node_indices shape = [num_edges]
        # neighbour_messages shape = [num_edges, representation_dim]
        # node_representations shape = [num_nodes, representation_dim]
        num_nodes = node_representations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_representations, aggregated_messages):
        # node_representations shape [num_nodes, representation_dim]
        # aggregated_messages shape [num_nodes, representation_dim]
        if self.combination_type == "gru":
            # create sequence of the two elements for the GRU layer
            h = tf.stack([node_representations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            h = tf.concat([node_representations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            h = node_representations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}")

        # process combined feature
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

gnn_model = GNNNodeClassifier()


print("finished")