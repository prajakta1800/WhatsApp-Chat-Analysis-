import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("loading CSV data")
df = pd.read_csv('WhatsApp.csv')
print("data loaded successfully.")

print("start of chat")
print(df.head())

df = df.dropna(subset=['message'])
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['user']

print("Preprocessed Data:")
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Data Shape:", X_train.shape, y_train.shape)
print("Testing Data Shape:", X_test.shape, y_test.shape)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Random Forest Model Trained")
y_pred = rf.predict(X_test)

print("Predictions:")
print(y_pred)

print("Calculating word count")
word_count = df['message'].apply(lambda x: len(str(x).split()))
print(word_count.describe())

most_common_word = Counter(' '.join(map(str, df['message'].fillna(''))).split()).most_common(10)
print("most commmon words are : ", most_common_word)

print("Generating wordcloud")
wordcloud = WordCloud(width=800, height=400).generate(' '.join(map(str,df['message'])))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Analyzing timeline")
df['day'] = pd.to_datetime(df['day'])
plt.figure(figsize=(12, 6))
sns.lineplot(x='day', y='message', data=df)
plt.title('message Frequency Over Time')
plt.xlabel('day')
plt.ylabel('message')
plt.show()







