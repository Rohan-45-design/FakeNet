# FakeNet Module Documentation

## Introduction

FakeNet is a system built to automatically spot bot accounts on social media. Instead of relying on just one clue, it looks at three different types of evidence — who the account looks like on paper, what it writes, and who it talks to — and combines them to make a smarter decision about whether an account is real or fake.

The system is broken into modules, kind of like an assembly line: each module does one job, then hands its output to the next one. This makes the project easier to build, test, fix, and split across a team.

---

## Module 1: Data Collection
**In plain terms:** This is where we gather the raw examples the system will learn from — a mix of known bot accounts and known real accounts.

**Purpose:** Collect publicly available datasets containing both bot and genuine user accounts.

**Inputs:**
- Reddit Bot Dataset
- TwiBot-22
- Cresci-2017

**Responsibilities:**
- Download benchmark datasets
- Verify dataset integrity
- Organize raw datasets
- Store datasets for preprocessing

**Output:**
```
data/raw/
```

---

## Module 2: Data Preprocessing
**In plain terms:** Raw data is messy — this step cleans it up so the rest of the system can actually use it, the same way you'd wash and chop vegetables before cooking.

**Purpose:** Prepare raw datasets for feature extraction and machine learning.

**Responsibilities:**
- Remove duplicate records
- Handle missing values
- Standardize labels
- Normalize data formats
- Merge datasets into a unified schema

**Output:**
```
data/processed/
```

---

## Module 3: Metadata Feature Extraction
**In plain terms:** This looks at the "profile stats" of an account — things like how old it is and how it behaves — since bots often show telltale patterns here (e.g., posting way too often, or having no followers at all).

**Purpose:** Extract profile-based information that helps distinguish bots from genuine users.

**Features Extracted:**
- Account age
- Number of followers
- Number of following
- Number of posts
- Posting frequency
- Reddit karma
- Profile completeness

**Output:** Metadata feature vector

---

## Module 4: NLP Analysis
**In plain terms:** NLP stands for "Natural Language Processing" — basically, teaching the computer to understand text. This module reads what an account actually writes and looks for patterns bots tend to have, like repetitive phrasing or oddly generic language.

**Purpose:** Analyze user-generated text to capture linguistic and semantic characteristics.

**Responsibilities:**
- Text cleaning
- Tokenization (breaking text into small pieces the model can process)
- Text embedding generation (turning words into numbers the model can compare)
- Sentiment analysis (is the tone positive, negative, neutral?)
- Toxicity detection
- Writing style analysis
- Repetitive content detection

**Output:** NLP feature vector

---

## Module 5: Graph Neural Network (GNN)
**In plain terms:** Think of this as mapping out who talks to whom, like a friendship map. Bots often interact in unnatural patterns — e.g., only talking to other bots, or all following the same tiny cluster of accounts — and this module is built to catch that.

**Purpose:** Capture relationship patterns between users by analyzing the social interaction graph.

**Responsibilities:**
- Construct interaction graphs
- Learn node representations (a "profile" for each account based on its position in the network)
- Identify community structures (clusters of accounts that interact a lot with each other)
- Generate graph embeddings

**Graph Sources:**
- Reply network
- Mention network
- Community graph
- User interaction graph

**Output:** Graph embeddings

---

## Module 6: Multi-modal Feature Fusion
**In plain terms:** "Multi-modal" just means "multiple types of information." This step takes everything learned from the profile stats, the text, and the social network map, and blends it into one combined summary the model can make a decision from.

**Purpose:** Combine metadata, NLP, and graph embeddings into a single feature representation.

**Inputs:**
- Metadata feature vector
- NLP feature vector
- Graph embeddings

**Output:** Unified feature vector

---

## Module 7: Bot Detection Model
**In plain terms:** This is the actual decision-maker. It takes the combined summary from the previous step and gives a verdict: how likely is this account to be a bot?

**Purpose:** Classify social media accounts as bots or genuine users.

**Input:** Unified feature vector

**Output:**
- Bot probability score (e.g., 0.94 = 94% likely to be a bot)
- Human/Bot prediction

---

## Module 8: Reinforcement Learning
**In plain terms:** Bots evolve to dodge detection, so the system needs to evolve too. This module lets FakeNet learn from its past mistakes and successes, adjusting its own strategy over time — similar to how a spam filter gets smarter the more it's used.

**Purpose:** Improve the adaptability of the detection system by learning from previous predictions and evolving bot behaviors.

**Responsibilities:**
- Optimize detection strategy
- Adapt to new bot behaviors
- Improve long-term detection performance

**Output:** Updated detection policy

---

## Module 9: Evaluation
**In plain terms:** Before trusting the system, we need to check how good it actually is — how often it's right, how often it's wrong, and in what ways.

**Purpose:** Measure the effectiveness of the proposed framework.

**Evaluation Metrics:**
- Accuracy — overall, how often is it right?
- Precision — when it says "bot," how often is that correct?
- Recall — of all the actual bots, how many did it catch?
- F1-Score — a balance between precision and recall
- ROC-AUC — a measure of how well the model separates bots from humans overall

**Output:** Performance report

---

## Module 10: Deployment
**In plain terms:** Once the system works, this is how people actually get to use it.

**Purpose:** Provide an interface through which end users can access the bot detection system.

**Possible Deployment Options:**
- Web application
- REST API
- Browser extension

---

## Overall Workflow (Step-by-Step, in Order)

1. Collect social media datasets.
2. Preprocess and clean the data.
3. Extract metadata features (profile stats).
4. Perform NLP analysis (analyze the text).
5. Generate graph embeddings using Graph Neural Networks (map social connections).
6. Fuse all extracted features into one unified representation.
7. Predict whether an account is a bot or a genuine user.
8. Improve the detection strategy using Reinforcement Learning.
9. Evaluate the model's performance.
10. Deploy the final application for end users.

---
