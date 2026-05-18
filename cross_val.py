import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# load data
df = pd.read_csv('Data/all-data.csv', names=['sentiment', 'text'], encoding='latin-1')

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['sentiment'])

# we don't split 80/20 here, k-fold handles the splitting
text_data = df['text'].values
labels = df['label'].values

# tokenize the whole dataset
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(text_data)
padded_data = pad_sequences(tokenizer.texts_to_sequences(text_data), maxlen=50, padding='post', truncating='post')

# setup the 5-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

print("starting 5-fold cross validation on the winning parameters...\n")

fold_no = 1
for train_index, test_index in kfold.split(padded_data):
    # slice the data for this specific fold
    X_train, X_test = padded_data[train_index], padded_data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(enumerate(weights))

    # build the model with your winning parameters
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    print(f"training fold {fold_no}...", end=" ", flush=True)
    model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        class_weight=weight_dict,
        callbacks=[early_stop],
        verbose=0
    )

    # test this fold
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"score: {accuracy:.4f}")

    fold_scores.append(accuracy)
    fold_no += 1

print("\nfinal cross validation results:")
print(f"all scores: {[round(s, 4) for s in fold_scores]}")
print(f"average true accuracy: {np.mean(fold_scores):.4f}")