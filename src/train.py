import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import clean_text

df = pd.read_csv(r'D:\00_PortFolio\AI_ML\spam-classifier\data\spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
df['clean'] = df['text'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

# Model 1: Naive Bayes (fast, interpretable)
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

# Model 2: Logistic Regression (usually more accurate)
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

for name, model in [('Naive Bayes', nb_pipeline), ('Logistic Regression', lr_pipeline)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\n--- {name} ---')
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Save the best model (Logistic Regression)
joblib.dump(lr_pipeline, r'D:\00_PortFolio\AI_ML\spam-classifier\models\classifier.pkl')
print('Model saved to models/classifier.pkl')



import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, lr_pipeline.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix — Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(r'D:\00_PortFolio\AI_ML\spam-classifier\notebooks\assets\confusion_matrix.png')
plt.show()