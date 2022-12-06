from tensorflow import keras
import os
import pandas as pd

zip_file = keras.utils.get_file(fname='cora.tgz',
                                origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
                                extract=True,
                                )
data_dir = os.path.join(os.path.dirname(zip_file), "cora")

citations_data = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"]
)

print(citations_data.head())

citations_data.describe()

# process data
column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers_data = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep="\t", header=None, names=column_names
)

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
