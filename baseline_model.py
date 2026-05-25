import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# loading data
df = pd.read_csv('Data/all-data.csv', names=['sentiment', 'text'], encoding='latin-1')

# initialize label encoder
encoder = LabelEncoder()

# convert text labels ('positive', 'negative', 'neutral') into numbers
df['label'] = encoder.fit_transform(df['sentiment'])

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

# automated demo of 10 custom statements

print("\n" + "=" * 40)
print("AUTOMATED DEMO: 10 CUSTOM HEADLINES")
print("=" * 40)

custom_headlines = [
    # positive
    "net sales surged by 25.5 % and operating profit doubled to eur 43.1 mn in the third quarter.",
    "the company has been awarded a significant order to supply machinery and equipment for a new plant in china.",
    "diluted earnings per share rose to eur 1.05 from eur 0.64, and the board proposed a raised dividend payout.",
    "the group's order book increased by 40 % year-on-year, showing strong recovery in demand.",
    # negative
    "operating loss widened to eur 12.7 mn as demand dropped and the company announced temporary lay-offs.",
    "the company issued a profit warning after pretax profit decreased by 69 %.",
    "due to weak market conditions, the factory will be closed and 150 employees will face redundancy.",
    # neutral
    "the annual general meeting will be held on tuesday at the helsinki headquarters.",
    "the share capital of the company is divided into 75,000 shares.",
    "the company announced the appointment of a new vice president of human resources starting next month."
]

for headline in custom_headlines:
    print(f"\nheadline: '{headline}'")

    # prep the text for naive bayes using tf-idf
    user_vectorized = vectorizer.transform([headline])

    # get probabilities
    raw_pred = baseline_model.predict_proba(user_vectorized)
    sentiment_index = np.argmax(raw_pred, axis=1)[0]
    confidence = raw_pred[0][sentiment_index] * 100

    sentiment_text = encoder.inverse_transform([sentiment_index])[0]

    print(f">> ai guess: {sentiment_text.upper()} (confidence: {confidence:.1f}%)")

# interactive live demo

print("\n" + "=" * 40)
print("LIVE DEMO: TYPE YOUR OWN HEADLINES")
print("type 'quit' to exit")
print("=" * 40)

while True:
    user_input = input("\nenter a financial headline: ")
    if user_input.lower() == 'quit':
        break

    # prep the text for naive bayes
    user_vectorized = vectorizer.transform([user_input])

    # get probabilities
    raw_pred = baseline_model.predict_proba(user_vectorized)
    sentiment_index = np.argmax(raw_pred, axis=1)[0]
    confidence = raw_pred[0][sentiment_index] * 100

    sentiment_text = encoder.inverse_transform([sentiment_index])[0]

    print(f">> ai guess: {sentiment_text.upper()} (confidence: {confidence:.1f}%)")