"""
MGTAB loader.

MGTAB ships as PyTorch tensors: a pre-computed 788-dim node feature matrix,
a relational graph (edge_index/edge_type/edge_weight), and bot/stance labels
for 10,199 nodes. There are no raw usernames, no raw tweets, and no
human-readable metadata fields (followers_count etc.) -- the 788 dims are
already-embedded features from the original MGTAB paper's pipeline, not
decomposable into the same named columns Cresci-2017 gives us.

We therefore do two things instead of faking a schema match:
  1. Compute genuine, comparable GRAPH features from the edge list (degree,
     in/out-degree, relation-type diversity) -- these slot into the shared
     `graph_*` columns honestly.
  2. Keep the opaque 788-dim embedding as a separate parquet file
     (mgtab_embeddings.parquet), linked by record_id, for anyone who wants
     to use it as a modeling input directly rather than pretending it maps
     onto Cresci's metadata/text columns.
"""
import torch
import pandas as pd
import numpy as np

MGTAB_DIR = "/home/claude/work/mgtab/MGTAB"


def build_mgtab():
    edge_index = torch.load(f"{MGTAB_DIR}/edge_index.pt", map_location="cpu", weights_only=False)
    edge_type = torch.load(f"{MGTAB_DIR}/edge_type.pt", map_location="cpu", weights_only=False)
    features = torch.load(f"{MGTAB_DIR}/features.pt", map_location="cpu", weights_only=False)
    labels_bot = torch.load(f"{MGTAB_DIR}/labels_bot.pt", map_location="cpu", weights_only=False)
    labels_stance = torch.load(f"{MGTAB_DIR}/labels_stance.pt", map_location="cpu", weights_only=False)

    n_nodes = features.shape[0]
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    out_degree = np.bincount(src, minlength=n_nodes)
    in_degree = np.bincount(dst, minlength=n_nodes)

    # relation-type diversity per node (how many distinct edge types touch it)
    n_types = int(edge_type.max().item()) + 1
    type_touch = np.zeros((n_nodes, n_types), dtype=bool)
    et = edge_type.numpy()
    type_touch[src, et] = True
    type_touch[dst, et] = True
    relation_type_diversity = type_touch.sum(axis=1)

    node_ids = [f"mgtab_{i:05d}" for i in range(n_nodes)]

    df = pd.DataFrame({
        "id": node_ids,
        "label": labels_bot.numpy().astype(int),
        "label_source_category": "mgtab_graph",
        "dataset_source": "mgtab",
        "graph_degree": in_degree + out_degree,
        "graph_in_degree": in_degree,
        "graph_out_degree": out_degree,
        "graph_relation_type_diversity": relation_type_diversity,
        "stance_label": labels_stance.numpy().astype(int),  # extra, not in shared schema
    })

    emb_df = pd.DataFrame(features.numpy())
    emb_df.insert(0, "record_id", ["mgtab:" + i for i in node_ids])

    return df, emb_df


if __name__ == "__main__":
    df, emb_df = build_mgtab()
    print(df.shape, emb_df.shape)
    print(df.head())
    df.to_parquet("/home/claude/work/pipeline/_mgtab_raw.parquet")
    emb_df.to_parquet("/home/claude/work/pipeline/mgtab_embeddings.parquet")
