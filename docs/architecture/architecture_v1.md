# FakeNet System Architecture (Version 1.0)

## Overview

FakeNet is a multi-modal bot detection framework built to identify automated (bot) accounts on social media platforms. Rather than relying on a single type of signal, it pulls together user metadata, textual content, and interaction graphs to get a more accurate read on which accounts are bots and which are genuine users.

The system is built as a modular pipeline — each component handles one specific task and passes its output to the next stage. This keeps the codebase easier to scale and maintain, and lets different team members work on separate modules without stepping on each other's work.

## System Objectives

The primary objectives of FakeNet are:

- Detect automated bot accounts on social media platforms.
- Improve detection accuracy using metadata, textual, and graph-based information.
- Develop a multi-modal bot detection framework that is scalable and modular.
- Adapt to evolving bot behavior through reinforcement learning.
- Evaluate the framework using standard machine learning metrics.


## System Inputs

- User profile metadata
- User-generated posts and comments
- User interaction network
- Benchmark datasets (TwiBot-22, Cresci-2017, Reddit)

## System Outputs

- Bot probability score
- Human/Bot classification
- Evaluation metrics


---

## High-Level Architecture

```
                                        Social Media Datasets
      (Reddit, TwiBot-22, Cresci-2017, etc.)
                            │
                            ▼
                   Data Collection Module
                            │
                            ▼
                  Data Preprocessing Module
                            │
                            ▼
                  Feature Engineering Module
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼
 Metadata Features      NLP Analysis     Graph Neural Network
        │                   │                    │
        │                   │             Graph Embeddings
        └───────────────────┼────────────────────┘
                            ▼
                  Multi-modal Feature Fusion
                            │
                            ▼
                  Bot Detection Model
                            │
                            ▼
              Reinforcement Learning Module
                            │
                            ▼
               Human / Bot Prediction Output
```

---

## Module Descriptions

### 1. Data Collection Module

Purpose: Gather publicly available datasets that contain both genuine and bot accounts.

Input:
- Reddit Bot Dataset
- TwiBot-22
- Cresci-2017

Output: Raw datasets stored in `data/raw/`

---

### 2. Data Preprocessing Module

Purpose Get the raw datasets into shape for machine learning.

**Responsibilities:**
- Remove duplicate records
- Handle missing values
- Standardize labels
- Normalize data formats
- Merge datasets into one unified structure

**Output:** `data/processed/`

---

### 3. Feature Engineering Module

Purpose Pull meaningful signals out of the processed data. These fall into three categories:

**A. Metadata Features** — describe the user's profile
- Account age
- Number of followers
- Number of following
- Number of posts
- Posting frequency
- Reddit karma
- Profile completeness

**B. NLP Features** — pulled from the text itself
- Text embeddings
- Sentiment score
- Toxicity score
- Writing style
- Vocabulary richness
- Repetitive content detection

**C. Graph Features** — describe how users interact
- Reply network
- Mention network
- Community structure
- Degree centrality
- Betweenness centrality
- PageRank
- Clustering coefficient

---

### 4. Feature Fusion Module

Purpose Combine the metadata, NLP, and graph features into a single, unified feature representation the model can actually use.

**Input:** Metadata + NLP + Graph features
**Output:** Unified feature vector

---

### 5. Bot Detection Module

Purpose Classify each account as bot or human based on the unified feature vector.

**Example output:**
```
Bot Probability : 0.94
Prediction       : Bot
```

---

### 6. Reinforcement Learning Module

Purpose: Improve the adaptability of the bot detection system by refining detection strategies based on feedback and evolving bot behavior.

**Responsibilities:**

- Optimize the detection policy
- Adapt to new bot patterns
- Improve long-term detection performance

**Input:**

- Prediction scores
- Model feedback

**Output:**

- Updated detection policy
- Improved bot classification

### 7. Evaluation Module

Purpose Measure how well the model is actually performing.

**Metrics used:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

### 8. Deployment Module

Purpose Give users a way to actually run bot detection.

**Possible deployment options:**
- Web application
- REST API
- Browser extension

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Graph Learning | PyTorch Geometric |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git & GitHub |

---

## Data Flow

1. Collect social media datasets.
2. Preprocess and clean the data.
3. Extract metadata features.
4. Extract NLP features.
5. Construct interaction graphs.
6. Generate graph embeddings using Graph Neural Networks.
7. Fuse all feature representations.
8. Train the bot detection model.
9. Improve decision-making using reinforcement learning.
10. Predict whether an account is a bot or a human.


## Project Folder Structure

```
FakeNet/
├── data/
│   ├── raw/
│   ├── processed/
│   └── schemas/
│
├── docs/
│   ├── architecture/
│   ├── literature_review/
│   ├── research_gap/
│   ├── references/
│   └── dataset_report.md
│
├── notebooks/
├── src/
├── tests/
├── README.md
└── requirements.txt
```

---

## Phase Mapping

| Phase   | Deliverable                                         |
| ------- | --------------------------------------------------- |
| Phase 1 | Research, datasets, architecture                    |
| Phase 2 | Data preprocessing and metadata feature engineering |
| Phase 3 | NLP feature extraction                              |
| Phase 4 | Graph Neural Network implementation                 |
| Phase 5 | Multi-modal feature fusion                          |
| Phase 6 | Bot detection model training                        |
| Phase 7 | Reinforcement learning optimization and evaluation  |
| Phase 8 | Deployment and final documentation                  |


---

## Team Responsibilities

**Member 1 – Data Engineering**
- Collect datasets
- Clean datasets
- Prepare unified schema
- Dataset documentation

**Member 2 – Research**
- Literature review
- Existing bot detection systems
- Research gap analysis
- References

**Member 3 – Infrastructure & Architecture**
- GitHub repository management
- System architecture
- Project documentation
- Collaboration workflow
- Project management

---

## Conclusion

FakeNet's modular and scalable architecture combines metadata analysis, NLP, Graph Neural Networks, and Reinforcement Learning to build a robust multi-modal bot detection framework. By integrating multiple sources of information and enabling adaptive learning, the proposed system aims to improve detection accuracy while remaining maintainable, extensible, and suitable for future enhancements.

---
