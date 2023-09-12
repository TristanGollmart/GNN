from tensorflow import keras
import os
import pandas as pd
import numpy as np

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
nx.draw_spring(cora_graph, node_size=15, node_color=subjects)
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
node_features = tf.cast(node_features.to_numpy(), dtype=tf.dtypes.float64)

# edge weights: set equal to 1 for each edge
edge_weights = tf.ones(shape=edges.shape[1])
print("edge weights shape:", edge_weights.shape)

# merge into tuple containing all relevant information about graph:
graph_info = (node_features, edges, edge_weights)
num_classes = len(class_names)
num_features = len(feature_names)
train_data, test_data = [], []
# allocate around 50% of each class to training
for _, group_data in papers_data.groupby("subject"):
    rand_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[rand_selection])
    test_data.append(group_data[~rand_selection])

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

x_train = train_data[feature_names].to_numpy()
y_train = train_data["subject"].to_numpy()
x_test = test_data[feature_names].to_numpy()
y_test = test_data["subject"].to_numpy()

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

# Help Functions

def run_experiment(model, x_train, y_train):
    # given a model and data run the pipeline from compiling the model with relevant parameters to fitting model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    # early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fitting
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        # validation_split=0.15,
        callbacks=[early_stopping],
    )
    return history

def display_class_probabilities(probabilities):
    for instance_idx, probs in enumerate(probabilities):
        print(f"instance {instance_idx + 1}:")
        for class_idx, prob in enumerate(probs):
            print(f" - {class_values[class_idx]}: {round(prob * 100, 2)}%")

def display_learning_curve(history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history.history["loss"])
    ax[0].plot(history.history["val_loss"])
    ax[0].legend(["trainloss", "validation loss"])
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")
    ax[1].plot(history.history["acc"])
    ax[1].plot(history.history["val_acc"])
    ax[1].legend(["train acc", "validation acc"])
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("acc")
    plt.show()


def create_ffn(hidden_units, dropout_rate, name=None):
    # returns sequential moddel, with number of hidden layers = len(hidden_units)
    # with the specified number of hidden_units
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    return keras.Sequential(fnn_layers, name=name)


def create_baseline_model(hidden_units, num_classes, dropout_rate=0.2):
    inputs = layers.Input(shape=(num_features,), name='input_features')
    x = create_ffn(hidden_units, dropout_rate, name=f'ffn_block1')(inputs)
    for block_idx in range(4):
        x1 = create_ffn(hidden_units, dropout_rate, name=f'ffn_block{block_idx+2}')(x)
        x = layers.Add(name=f'skip_connection{block_idx+2}')([x, x1])
    logits = layers.Dense(num_classes, name='logits')(x)
    return keras.Model(inputs=inputs, outputs=logits, name='baseline')

baseline_model = create_baseline_model(hidden_units, num_classes, dropout_rate)
baseline_model.summary()
history = run_experiment(baseline_model, x_train, y_train)
display_learning_curve(history)


''' only for baseline model that does not account for edge information
def generate_random_instances(num_instances):
    token_probability = x_train.mean(axis=0)
    instances = []
    for _ in range(num_instances):
        probabilities = np.random.uniform(size=len(token_probability))
        instance = (probabilities < token_probability).astype(int)
        instances.append(instance)
    return instances
new_instances = generate_random_instances(100)
'''

# -----------Build GNN-----------

class GraphConvLayer(layers.Layer):
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
        # use to mask messages of all neighbours using edge_eights as binary weights
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

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        '''
        process inputs via aggregating messages
        and parsing it to a FNN to produce node embeddings
        :param inputs: tuple of three elements: node representations, edges, edge_weigts
        :return: node_embeddings of shape [num_nodes, representation_dim]
        '''
        node_representations, edges, edge_weights = inputs
        # get node indices (source) and neighbour_indices (target) from edges
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour representation shape is [num_edges, representation_dim]
        neighbour_representations = tf.gather(node_representations, neighbour_indices)

        # prepare the messages of the neighbours
        neighbour_messages = self.prepare(neighbour_representations, edge_weights)

        # aggregate the neighbours messages
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_representations)

        # update the node embedding using the neighbour messages
        return self.update(node_representations, aggregated_messages)


class GNNNodeClassifier(tf.keras.Model):
    def __init__(
            self,
            graph_info,
            num_classes,
            hidden_units,
            aggregation_type="sum",
            combination_type="concat",
            dropout_rate=0.2,
            normalize=True,
            *args,
            **kwargs,
    ):

        super(GNNNodeClassifier, self).__init__(*args, **kwargs)

        # unpack graph_info
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        if edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # scaling
        self.edge_weights = self.edge_weights / (tf.math.reduce_sum(self.edge_weights))

        # create pre-process layer
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # create first graph conv layer
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # create second gcn

        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2"
        )
        # post processing layer
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # compute logits layer
        self.compute_logits = layers.Dense(units=num_classes)

    def call(self, input_node_indices):
        x = self.preprocess(self.node_features)
        x1 = self.conv1((x, self.edges, self.edge_weights))
         # skip connection
        x = x1 + x
        x2 = self.conv2((x, self.edges, self.edge_weights))
        x = x2 + x
        #post processing
        x = self.postprocess(x)
        # Fetch node embeddings from x with queried input indices
        node_embeddings = tf.gather(x, input_node_indices)
        return self.compute_logits(node_embeddings)

# Training

gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
)

print("GNN output shape:", gnn_model([1, 10, 100]))
gnn_model.summary()

# define new train data for GNN
x_train = train_data.paper_id.to_numpy()
x_test = test_data.paper_id.to_numpy()
history = run_experiment(gnn_model, x_train, y_train)

display_learning_curve(history)

# evaluate trained model predictions on test set

_, test_accurace = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accurace, 2)}%")

print("finished")