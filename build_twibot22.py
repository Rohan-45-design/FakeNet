"""
TwiBot-22 loader.

Only label.csv was provided (id, label for ~1M accounts). The full TwiBot-22
release also includes user metadata, tweets, and a heterogeneous interaction
graph, but none of that was uploaded here -- only the label file. We include
these rows as a label-only reference split (all feature columns NaN,
has_metadata/has_text/has_graph = False) rather than dropping them silently,
so downstream users can see exactly what's available vs. missing.
"""
import pandas as pd

LABEL_CSV = "/home/claude/work/label.csv"


def build_twibot22():
    df = pd.read_csv(LABEL_CSV, dtype={"id": "string"})
    df["label"] = (df["label"].str.lower() == "bot").astype(int)
    df["label_source_category"] = "twibot22_label_only"
    df["dataset_source"] = "twibot22"
    return df


if __name__ == "__main__":
    df = build_twibot22()
    print(df.shape)
    print(df["label"].value_counts())
    df.to_parquet("/home/claude/work/pipeline/_twibot22_raw.parquet")
