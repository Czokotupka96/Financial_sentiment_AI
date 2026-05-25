import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import itertools

# load and prep data exactly the same as before
df = pd.read_csv('../Data/all-data.csv', names=['sentiment', 'text'], encoding='latin-1')

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=50, padding='post', truncating='post')
X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=50, padding='post', truncating='post')

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(enumerate(weights))

# --- NEW: THE GRID SEARCH LOOP ---

# the parameters we want to test
lstm_units_options = [32, 64]
dropout_options = [0.3, 0.5]
learning_rates = [0.001, 0.0005]

# create a list of all possible combinations (2 x 2 x 2 = 8 runs)
combinations = list(itertools.product(lstm_units_options, dropout_options, learning_rates))

best_accuracy = 0
best_params = {}

print("starting the grid search\n")

for i, (units, dropout, lr) in enumerate(combinations):
    print(f"run {i + 1}/8 | units: {units}, dropout: {dropout}, learning rate: {lr}...", end=" ", flush=True)

    # build the model with the current loop's parameters
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64),
        Bidirectional(LSTM(units)),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    optimizer = Adam(learning_rate=lr)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # patience=2 to fail fast and save time
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # verbose=0 completely hides the epoch progress bars
    model.fit(
        X_train_padded, y_train,
        epochs=10,
        validation_data=(X_test_padded, y_test),
        class_weight=weight_dict,
        callbacks=[early_stop],
        verbose=0
    )

    # evaluate the model quietly
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"accuracy: {accuracy:.4f}")

    # save the winner
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {'units': units, 'dropout': dropout, 'lr': lr}

print("\ngrid search complete:")
print(f"best baseline to beat was: 0.6900")
print(f"best lstm accuracy: {best_accuracy:.4f}")
print(f"winning parameters: {best_params}")