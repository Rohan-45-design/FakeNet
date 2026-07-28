"""
Cresci-2017 loader + feature engineering.

Cresci-2017 is the only one of the three uploaded sources that contains
raw, human-readable metadata (users.csv) and raw text (tweets.csv) per
account. We stream tweets.csv directly out of the nested zip (via `unzip -p`)
instead of extracting it to disk, because the combined uncompressed size of
all tweets.csv files (~2.3 GB) doesn't comfortably fit in the sandbox's
available disk space.

Category -> label mapping follows the original Cresci-2017 / MIB release:
    genuine_accounts        -> human (0)
    social_spambots_1/2/3   -> bot (1)
    traditional_spambots_1-4-> bot (1)
    fake_followers          -> bot (1)
"""
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

CRESCI_ZIP = "/home/claude/work/cresci_raw/datasets_full.csv/{cat}.csv.zip"
CRESCI_INNER = "{cat}.csv/{fname}"

CATEGORIES = {
    "genuine_accounts": 0,
    "social_spambots_1": 1,
    "social_spambots_2": 1,
    "social_spambots_3": 1,
    "traditional_spambots_1": 1,
    "traditional_spambots_2": 1,
    "traditional_spambots_3": 1,
    "traditional_spambots_4": 1,
    "fake_followers": 1,
}

USER_COLS = [
    "id", "screen_name", "name", "statuses_count", "followers_count",
    "friends_count", "favourites_count", "listed_count", "url", "lang",
    "default_profile", "default_profile_image", "geo_enabled", "verified",
    "description", "created_at",
]

TWEET_USECOLS = [
    "user_id", "text", "source", "in_reply_to_status_id",
    "retweeted_status_id", "retweet_count", "favorite_count",
    "num_hashtags", "num_urls", "num_mentions", "created_at",
]


def _stream_path(cat, fname):
    zip_path = CRESCI_ZIP.format(cat=cat)
    inner = CRESCI_INNER.format(cat=cat, fname=fname)
    return zip_path, inner


def load_users(cat: str) -> pd.DataFrame:
    zip_path, inner = _stream_path(cat, "users.csv")
    proc = subprocess.run(["unzip", "-p", zip_path, inner], capture_output=True)
    from io import BytesIO
    df = pd.read_csv(BytesIO(proc.stdout), usecols=lambda c: c in USER_COLS,
                      dtype={"id": "string"}, low_memory=False,
                      encoding="utf-8", encoding_errors="replace")
    return df


def aggregate_tweet_features(cat: str) -> pd.DataFrame:
    """Stream tweets.csv in chunks, accumulate per-user aggregates without
    ever materializing the full file on disk or in memory at once."""
    zip_path = CRESCI_ZIP.format(cat=cat)
    inner = CRESCI_INNER.format(cat=cat, fname="tweets.csv")

    # Some categories (traditional_spambots_2/3/4) have no tweets.csv at all.
    listing = subprocess.run(["unzip", "-l", zip_path], capture_output=True, text=True).stdout
    if "tweets.csv" not in listing:
        return pd.DataFrame(columns=["id"])

    proc = subprocess.Popen(["unzip", "-p", zip_path, inner], stdout=subprocess.PIPE)

    acc = {}  # user_id -> running stats

    def get(uid):
        if uid not in acc:
            acc[uid] = dict(
                tweet_count=0, len_sum=0, hashtag_sum=0, url_sum=0,
                mention_sum=0, reply_sum=0, retweet_sum=0,
                rtcount_sum=0.0, favcount_sum=0.0, sources=set(),
            )
        return acc[uid]

    try:
        for chunk in pd.read_csv(proc.stdout, usecols=lambda c: c in TWEET_USECOLS,
                                  chunksize=200_000, dtype={"user_id": "string"},
                                  low_memory=False, on_bad_lines="skip",
                                  encoding="utf-8", encoding_errors="replace"):
            chunk["text"] = chunk.get("text", "").astype(str)
            for col in ["num_hashtags", "num_urls", "num_mentions",
                        "retweet_count", "favorite_count"]:
                if col not in chunk.columns:
                    chunk[col] = np.nan

            grouped = chunk.groupby("user_id", sort=False)
            for uid, g in grouped:
                if pd.isna(uid):
                    continue
                s = get(uid)
                s["tweet_count"] += len(g)
                s["len_sum"] += g["text"].str.len().sum()
                s["hashtag_sum"] += g["num_hashtags"].fillna(0).sum()
                s["url_sum"] += g["num_urls"].fillna(0).sum()
                s["mention_sum"] += g["num_mentions"].fillna(0).sum()
                s["reply_sum"] += g["in_reply_to_status_id"].notna().sum()
                s["retweet_sum"] += g["retweeted_status_id"].notna().sum()
                s["rtcount_sum"] += g["retweet_count"].fillna(0).sum()
                s["favcount_sum"] += g["favorite_count"].fillna(0).sum()
                if "source" in g.columns:
                    s["sources"].update(g["source"].dropna().unique().tolist())
    finally:
        proc.stdout.close()
        proc.wait()

    rows = []
    for uid, s in acc.items():
        n = max(s["tweet_count"], 1)
        rows.append(dict(
            id=uid,
            tweet_count=s["tweet_count"],
            avg_tweet_length=s["len_sum"] / n,
            avg_num_hashtags=s["hashtag_sum"] / n,
            avg_num_urls=s["url_sum"] / n,
            avg_num_mentions=s["mention_sum"] / n,
            reply_rate=s["reply_sum"] / n,
            retweet_rate=s["retweet_sum"] / n,
            avg_retweet_count=s["rtcount_sum"] / n,
            avg_favorite_count=s["favcount_sum"] / n,
            unique_source_count=len(s["sources"]),
        ))
    return pd.DataFrame(rows)


def build_category(cat: str, label: int) -> pd.DataFrame:
    users = load_users(cat)
    tweets = aggregate_tweet_features(cat)
    df = users.merge(tweets, on="id", how="left")
    df["label"] = label
    df["label_source_category"] = cat
    df["dataset_source"] = "cresci2017"
    return df


def build_cresci_2017() -> pd.DataFrame:
    parts = []
    for cat, label in CATEGORIES.items():
        print(f"[cresci2017] processing {cat} ...")
        parts.append(build_category(cat, label))
    df = pd.concat(parts, ignore_index=True)
    print(f"[cresci2017] total rows: {len(df)}")
    return df


if __name__ == "__main__":
    df = build_cresci_2017()
    df.to_parquet("/home/claude/work/pipeline/_cresci_raw_merged.parquet")
    print(df.shape)
    print(df.head())
