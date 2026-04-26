# Spam Message Classifier (NLP + Machine Learning)

An end-to-end Natural Language Processing project that classifies SMS/email messages as Spam or Ham (Not Spam) using classical machine learning techniques and an interactive Streamlit web app.

This project demonstrates the complete ML workflow — from data cleaning and exploratory analysis to model deployment.

# Project Highlights
End-to-end NLP pipeline
Exploratory Data Analysis (EDA) with visualizations
Feature engineering using TF-IDF
Model comparison and evaluation
Interactive Streamlit web application
Reproducible project structure

# Model Performance
| # Model             |  Accuracy  |  F1 Score (Spam)   |
| ------------------- | ---------- | ------------------ |
| Naive Bayes         | **97.2%**  | **95.1%**          |
| Logistic Regression | **98.4%**  | **97.3%**          |

*Why F1-Score?*
The dataset is imbalanced (87% ham vs 13% spam), making F1-score a more reliable metric than accuracy.

# Dataset
UCI SMS Spam Collection

5,572 labelled SMS messages
4,825 Ham (87%)
747 Spam (13%)

# Exploratory Insights
Key observations from the analysis:

Spam messages are ~2× longer than ham messages on average
Most frequent spam trigger words:
free
win
call
prize
claim
Text length is a strong predictive feature
TF-IDF significantly outperformed basic Bag-of-Words

# Machine Learning Pipeline
1. Text preprocessing
Lowercasing
Tokenization
Stopword removal
Punctuation cleaning
2. Feature Engineering
TF-IDF Vectorization
3. Model Training
Multinomial Naive Bayes
Logistic Regression
4. Evaluation
Precision, Recall, F1-Score
Confusion Matrix
5. Deployment
Streamlit Web App

# Streamlit App
The web app allows users to input a message and instantly receive a spam prediction.

**Run locally:**
git clone https://github.com/you/spam-classifier
cd spam-classifier
pip install -r requirements.txt
streamlit run app.py


# Key Learnings
Importance of proper text preprocessing in NLP tasks
Why TF-IDF often outperforms Bag-of-Words
Handling class imbalance using appropriate evaluation metrics
Building reproducible ML pipelines with scikit-learn
Deploying ML models using Streamlit

# Future Improvements
Try advanced models (SVM, Random Forest, XGBoost)
Add deep learning (LSTM / Transformers)
Deploy the app to the cloud
Add real-time email classification
