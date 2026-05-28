import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# load and prep the data
df = pd.read_csv('Data/all-data.csv', names=['sentiment', 'text'], encoding='latin-1')

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print(f"\ntraining on {len(X_train)} headlines. testing on {len(X_test)} headlines.")

# tokenize and pad
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=50, padding='post', truncating='post')
X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=50, padding='post', truncating='post')

# class weights
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(enumerate(weights))

# the champion model (using the nested cv winners)
print("building the model")
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    Bidirectional(LSTM(64)),  # winning units
    Dropout(0.5),  # winning dropout
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)  # winning learning rate
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# train it
print("\ntraining the final model...")
model.fit(X_train_padded, y_train, epochs=20, validation_data=(X_test_padded, y_test),
          class_weight=weight_dict, callbacks=[early_stop], verbose=1)

# final grade
print("\n" + "=" * 40)
print("SUMMARY OF MODEL")
print("=" * 40)
predictions = model.predict(X_test_padded)
predicted_classes = np.argmax(predictions, axis=1)
print(classification_report(y_test, predicted_classes, target_names=encoder.classes_))

# automated demo of 10 custom headlines

print("\n" + "=" * 40)
print("AUTOMATED DEMO: 10 CUSTOM HEADLINES")
print("=" * 40)

custom_headlines = [
    # Positive
    "Net sales surged by 25.5 % and operating profit doubled to eur 43.1 mn in the third quarter.",
    "The company has been awarded a significant order to supply machinery and equipment for a new plant in China.",
    "Diluted earnings per share rose to eur 1.05 from eur 0.64, and the board proposed a raised dividend payout.",
    "The group's order book increased by 40 % year-on-year, showing strong recovery in demand.",
    # Negative
    "Operating loss widened to eur 12.7 mn as demand dropped and the company announced temporary lay-offs.",
    "The company issued a profit warning after pretax profit decreased by 69 %.",
    "Due to weak market conditions, the factory will be closed and 150 employees will face redundancy.",
    # Neutral
    "The annual general meeting will be held on Tuesday at the Helsinki headquarters.",
    "The share capital of the company is divided into 75,000 shares.",
    "The company announced the appointment of a new Vice President of Human Resources starting next month."
]

for headline in custom_headlines:
    print(f"\nHeadline: '{headline}'")
    user_seq = tokenizer.texts_to_sequences([headline])
    user_padded = pad_sequences(user_seq, maxlen=50, padding='post', truncating='post')

    raw_pred = model.predict(user_padded, verbose=0)
    sentiment_index = np.argmax(raw_pred, axis=1)[0]
    confidence = raw_pred[0][sentiment_index] * 100
    sentiment_text = encoder.inverse_transform([sentiment_index])[0]

    print(f">> AI GUESS: {sentiment_text.upper()} (Confidence: {confidence:.1f}%)")

# interactive live demo

print("\n" + "=" * 40)
print("LIVE DEMO: TYPE YOUR OWN HEADLINES")
print("type 'quit' to exit")
print("=" * 40)

while True:
    user_input = input("\nenter a financial headline: ")
    if user_input.lower() == 'quit':
        break

    # prep the user's text just like we prepped the training data
    user_seq = tokenizer.texts_to_sequences([user_input])
    user_padded = pad_sequences(user_seq, maxlen=50, padding='post', truncating='post')

    # guess the sentiment
    raw_pred = model.predict(user_padded, verbose=0)
    sentiment_index = np.argmax(raw_pred, axis=1)[0]
    confidence = raw_pred[0][sentiment_index] * 100

    # translate the number back to text ('negative', 'neutral', 'positive')
    sentiment_text = encoder.inverse_transform([sentiment_index])[0]

    print(f">> AI GUESS: {sentiment_text.upper()} (Confidence: {confidence:.1f}%)")