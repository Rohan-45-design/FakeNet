"""
FakeNet dataset pipeline -- merge, clean, engineer, split.

Combines:
  - Cresci-2017  (full metadata + text, real bot/human labels)
  - MGTAB        (pre-computed embedding + real graph edges + real labels;
                  no raw text/metadata -- see build_mgtab.py)
  - TwiBot-22    (labels only -- no features at all; see build_twibot22.py)

TwiBot-20 was NOT included: no TwiBot-20 file was provided in the uploads,
so it is absent from this merge. Re-run this pipeline with a TwiBot-20
loader added (mirroring build_cresci.py / build_mgtab.py) once that data is
available.

Reference date for age/rate calculations: the Cresci-2017 crawl took place
around October 2016 (per the dataset's own file timestamps); we use
2016-11-01 as a fixed crawl reference so "account age" and "posting rate"
are internally consistent for that dataset, rather than computed against
today's date (which would make every Cresci account look artificially old
and would distort rate features).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import build_cresci
import build_mgtab
import build_twibot22

CRESCI_REFERENCE_DATE = pd.Timestamp("2016-11-01", tz="UTC")

SHARED_COLUMNS = [
    # identity
    "record_id", "dataset_source", "platform", "label", "label_source_category",
    # metadata features
    "statuses_count", "followers_count", "friends_count", "favourites_count",
    "listed_count", "account_age_days", "statuses_per_day", "verified",
    "default_profile", "default_profile_image", "geo_enabled", "has_url",
    "has_description", "description_length", "screen_name_length", "name_length",
    "digits_in_screen_name_ratio", "follower_friend_ratio",
    # text features
    "tweet_count", "avg_tweet_length", "avg_num_hashtags", "avg_num_urls",
    "avg_num_mentions", "reply_rate", "retweet_rate", "avg_retweet_count",
    "avg_favorite_count", "unique_source_count", "tweets_per_day",
    # graph features
    "graph_degree", "graph_in_degree", "graph_out_degree",
    "graph_relation_type_diversity",
    # availability flags
    "has_metadata", "has_text", "has_graph",
]


def _digits_ratio(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    digit_count = s.str.count(r"\d")
    return digit_count / s.str.len().replace(0, np.nan)


def prep_cresci(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["record_id"] = "cresci2017:" + df["id"].astype(str)
    df["platform"] = "twitter"

    created = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y",
                              errors="coerce")
    df["account_age_days"] = (CRESCI_REFERENCE_DATE - created).dt.total_seconds() / 86400
    df["account_age_days"] = df["account_age_days"].clip(lower=1)

    df["statuses_per_day"] = df["statuses_count"] / df["account_age_days"]
    df["tweets_per_day"] = df["tweet_count"] / df["account_age_days"]

    df["has_url"] = df["url"].notna().astype(int)
    df["has_description"] = df["description"].notna().astype(int)
    df["description_length"] = df["description"].fillna("").str.len()
    df["screen_name_length"] = df["screen_name"].fillna("").str.len()
    df["name_length"] = df["name"].fillna("").str.len()
    df["digits_in_screen_name_ratio"] = _digits_ratio(df["screen_name"])
    df["follower_friend_ratio"] = df["followers_count"] / (df["friends_count"] + 1)

    for c in ["verified", "default_profile", "default_profile_image", "geo_enabled"]:
        df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(
            df[c] if df[c].dtype != object else np.nan
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["graph_degree"] = np.nan
    df["graph_in_degree"] = np.nan
    df["graph_out_degree"] = np.nan
    df["graph_relation_type_diversity"] = np.nan

    df["has_metadata"] = 1
    df["has_text"] = (df["tweet_count"].fillna(0) > 0).astype(int)
    df["has_graph"] = 0  # Cresci-2017 provides no edge list, only follower COUNTS

    keep = [c for c in SHARED_COLUMNS if c in df.columns]
    return df[keep]


def prep_mgtab(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["record_id"] = "mgtab:" + df["id"].astype(str)
    df["platform"] = "twitter"

    for c in ["statuses_count", "followers_count", "friends_count", "favourites_count",
              "listed_count", "account_age_days", "statuses_per_day", "verified",
              "default_profile", "default_profile_image", "geo_enabled", "has_url",
              "has_description", "description_length", "screen_name_length",
              "name_length", "digits_in_screen_name_ratio", "follower_friend_ratio",
              "tweet_count", "avg_tweet_length", "avg_num_hashtags", "avg_num_urls",
              "avg_num_mentions", "reply_rate", "retweet_rate", "avg_retweet_count",
              "avg_favorite_count", "unique_source_count", "tweets_per_day"]:
        df[c] = np.nan

    df["has_metadata"] = 0
    df["has_text"] = 0
    df["has_graph"] = 1

    keep = [c for c in SHARED_COLUMNS if c in df.columns]
    return df[keep]


def prep_twibot22(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["record_id"] = "twibot22:" + df["id"].astype(str)
    df["platform"] = "twitter"

    for c in ["statuses_count", "followers_count", "friends_count", "favourites_count",
              "listed_count", "account_age_days", "statuses_per_day", "verified",
              "default_profile", "default_profile_image", "geo_enabled", "has_url",
              "has_description", "description_length", "screen_name_length",
              "name_length", "digits_in_screen_name_ratio", "follower_friend_ratio",
              "tweet_count", "avg_tweet_length", "avg_num_hashtags", "avg_num_urls",
              "avg_num_mentions", "reply_rate", "retweet_rate", "avg_retweet_count",
              "avg_favorite_count", "unique_source_count", "tweets_per_day",
              "graph_degree", "graph_in_degree", "graph_out_degree",
              "graph_relation_type_diversity"]:
        df[c] = np.nan

    df["has_metadata"] = 0
    df["has_text"] = 0
    df["has_graph"] = 0

    keep = [c for c in SHARED_COLUMNS if c in df.columns]
    return df[keep]


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["record_id"]).reset_index(drop=True)

    numeric_cols = [c for c in SHARED_COLUMNS
                    if c not in ("record_id", "dataset_source", "platform", "label",
                                 "label_source_category", "has_metadata", "has_text",
                                 "has_graph")]

    for c in numeric_cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # impute within-source median first (keeps each source's own scale)
        df[c] = df.groupby("dataset_source")[c].transform(
            lambda s: s.fillna(s.median())
        )
        # if an entire source lacks this feature, fall back to the global
        # median (documented limitation: this borrows Cresci-2017's scale
        # for sources that have no such feature at all, e.g. MGTAB/TwiBot-22
        # metadata columns -- it exists only so the column is non-null, the
        # has_metadata/has_text/has_graph flags are the honest signal to
        # actually condition on, not these imputed values)
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

    return df


def make_splits(df: pd.DataFrame, seed=42):
    strat_key = df["dataset_source"].astype(str) + "_" + df["label"].astype(str)
    train, temp = train_test_split(df, test_size=0.30, random_state=seed, stratify=strat_key)
    strat_key_temp = temp["dataset_source"].astype(str) + "_" + temp["label"].astype(str)
    val, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=strat_key_temp)
    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    return pd.concat([train, val, test], ignore_index=True)


def main():
    cresci = build_cresci.build_cresci_2017()
    mgtab, mgtab_emb = build_mgtab.build_mgtab()
    twibot22 = build_twibot22.build_twibot22()

    merged = pd.concat([
        prep_cresci(cresci),
        prep_mgtab(mgtab),
        prep_twibot22(twibot22),
    ], ignore_index=True)

    merged = clean_and_impute(merged)
    merged = make_splits(merged)

    print(merged.shape)
    print(merged["dataset_source"].value_counts())
    print(merged.groupby(["dataset_source", "split"]).size())
    print(merged.groupby(["dataset_source", "label"]).size())

    out_dir = "/home/claude/work/pipeline"
    merged.to_parquet(f"{out_dir}/fakenet_merged_dataset.parquet", index=False)
    merged.to_csv(f"{out_dir}/fakenet_merged_dataset.csv", index=False)
    mgtab_emb.to_parquet(f"{out_dir}/mgtab_embeddings.parquet", index=False)

    return merged


if __name__ == "__main__":
    main()
