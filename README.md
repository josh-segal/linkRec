# LinkRec: Contextual News Recommendation System

## Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [AI/ML Technical Report](#aiml-technical-report)
- [System Architecture](#system-architecture)
- [Future Improvements](#future-improvements)

# Overview
LinkRec is a contextual news recommendation system built on the MIND dataset. It frames the problem as a contextual bandit task, using real-time user and article context to serve ranked slates of articles. The goal is to maximize user engagement through adaptive, feedback-driven learning.

# Motivation
Recommendation systems drive engagement across digital platforms. News recommendation, in particular, is challenging due to fast-changing user preferences and partial feedback. Contextual bandits are well-suited here—they make one-shot decisions with limited supervision and adapt quickly. This project was an exploration of reinforcement learning techniques applied in a scalable, real-world setting.

Full report will be posted soon here.

# AI/ML Technical Report
## Data Processing
The MIND dataset was parsed into structured context-action-reward examples. Category and subcategory features were encoded using sentence transformers and clustered into compact representations. Context features included time of day, day of week, and click history length. All numerical features were standardized.

## Machine Learning Architecture
The system includes four bandit models: LinUCB, Thompson Sampling, Doubly Robust, and a custom Slate Ranking variant. Each uses context vectors to select and rank articles. Semantic embeddings provide rich input features. Models were evaluated both offline and interactively via the web app.

## Model Performance
Baseline random selection reached 2.7% top-1 accuracy. Bandit models performed between 10% and 13%, with the Slate Ranking and Doubly Robust approaches outperforming others in all key metrics: accuracy, NDCG, and precision. The models learned quickly, achieving most gains with just 5% of the available data.

# System Architecture
The backend is a lightweight Flask server exposing model inference and update APIs. The frontend is a React app that handles user interaction and visualization. Data flows from context generation to model scoring, then user feedback updates the model in real time.

# Future Improvements
Ingesting real-time news would test generalization beyond static datasets. Pretraining encoders on user-item histories could improve context representations. Deeper user modeling and comparisons with full RL agents like policy gradient or DDPG would further benchmark performance.