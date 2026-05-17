import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# loading data
df = pd.read_csv('Data/all-data.csv', names=['sentiment', 'text'], encoding='latin-1')

print("original data:")
print(df.head())

# initialize label encoder
encoder = LabelEncoder()

# convert text labels ('positive', 'negative', 'neutral') into numbers
df['label'] = encoder.fit_transform(df['sentiment'])

print("\ndata after label encoding:")
print(df[['sentiment', 'label', 'text']].head(10))

# encoder mapping
print("\nencoder mapping:")
for i, item in enumerate(encoder.classes_):
    print(f"{item}  -->  {i}")

# split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print(f"\ntraining on {len(X_train)} headlines. testing on {len(X_test)} headlines.")

# vectorize the text using tf-idf (turning words into math)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# train the baseline model (naive bayes)
print("\ntraining the baseline naive bayes model...")
baseline_model = MultinomialNB()
baseline_model.fit(X_train_vectorized, y_train)

# test the model and print the results
predictions = baseline_model.predict(X_test_vectorized)
print("\n--- baseline model results ---")
print(classification_report(y_test, predictions, target_names=encoder.classes_))