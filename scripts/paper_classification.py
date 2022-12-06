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
feature_names = set(papers_data.columns) - {"paper_id", "subject"}
print(feature_names)

edges = citations_data[["source", "target"]].to_numpy().T
print("Edges shape:", edges.shape)

# important: Set fixed order first: by paper id from min to max
node_features = papers_data.sort_values("paper_id")[feature_names]


print("finished")