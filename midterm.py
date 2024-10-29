import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def add_all_features(df):
 
    df = df.copy()
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)

    df['Year'] = pd.to_datetime(df['Time'], unit='s').dt.year

    df['Text'] = df['Text'].fillna('').astype(str)
    df['Summary'] = df['Summary'].fillna('').astype(str)
    df['LenText'] = df['Text'].apply(len)
    df['LenSummary'] = df['Summary'].apply(len)
    df['Text_information'] = df['ProductId'] + ' ' + df['Summary'] + ' ' + df['Text']
    return df

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df = add_all_features(train_df)


X_train = train_df[train_df['Score'].notnull()]

X_test = pd.merge(test_df[['Id']], train_df, on='Id', how='left')


X_test['Helpfulness'] = X_test['Helpfulness'].fillna(0)
X_test['Year'] = X_test['Year'].fillna(train_df['Year'].median())
X_test['LenText'] = X_test['LenText'].fillna(0)
X_test['LenSummary'] = X_test['LenSummary'].fillna(0)
X_test['Text_information'] = X_test['Text_information'].fillna('')

text_feature = 'Text_information'
numeric_features = ['Helpfulness', 'Year', 'LenText', 'LenSummary']
target = 'Score'

X = X_train[[text_feature] + numeric_features]
y = X_train[target]

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X, y, test_size=0.2
)

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            max_features=20000,
            stop_words='english',
            ngram_range=(1,2),
            min_df=5
        ), text_feature),
        ('num', StandardScaler(), numeric_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, n_jobs=-1))
])

start_time = time.time()
pipeline.fit(X_train_split, y_train_split)
end_time = time.time()
print(f"Model fitting took {round(end_time - start_time, 2)} seconds")


start_time = time.time()
y_val_predictions = pipeline.predict(X_val_split)
end_time = time.time()
print(f"Prediction on validation set took {round(end_time - start_time, 2)} seconds")

accuracy = accuracy_score(y_val_split, y_val_predictions)
print(f"Accuracy on validation set = {accuracy:.4f}")


cm = confusion_matrix(y_val_split, y_val_predictions, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

X_test_features = X_test[[text_feature] + numeric_features]

start_time = time.time()
X_test['Score'] = pipeline.predict(X_test_features)
end_time = time.time()
print(f"Prediction on test set took {round(end_time - start_time, 2)} seconds")

submission = X_test[['Id', 'Score']]
submission.to_csv('submission.csv', index=False)

classifier = pipeline.named_steps['classifier']
feature_names_text = pipeline.named_steps['preprocessor'].named_transformers_['text'].get_feature_names_out()
feature_names_num = numeric_features
feature_names = np.concatenate([feature_names_text, feature_names_num])

coefficients = classifier.coef_
importance = np.abs(coefficients).sum(axis=0)

top20_indices = importance.argsort()[-20:][::-1]
top20_features = feature_names[top20_indices]
top20_importance = importance[top20_indices]

plt.figure(figsize=(12, 8))
sns.barplot(x=top20_importance, y=top20_features, palette='viridis')
plt.title('Top 20 Important Features')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

