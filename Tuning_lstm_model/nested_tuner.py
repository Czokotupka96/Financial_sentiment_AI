import pandas as pd
import numpy as np
import itertools
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# load and prep the data (using the entire dataset)
df = pd.read_csv('../Data/all-data.csv', names=['sentiment', 'text'], encoding='latin-1')

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['sentiment'])

text_data = df['text'].values
labels = df['label'].values

# tokenize and pad
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(text_data)
padded_data = pad_sequences(tokenizer.texts_to_sequences(text_data), maxlen=50, padding='post', truncating='post')

# set up the grid search parameters (the outer loop)
lstm_units_options = [32, 64]
dropout_options = [0.3, 0.5]
learning_rates = [0.001, 0.0005]
combinations = list(itertools.product(lstm_units_options, dropout_options, learning_rates))

# set up the cross-validation (the inner loop)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

best_f1_score = 0
best_params = {}
all_results = []

print(
    f"starting nested cross-validation: {len(combinations)} combinations x 5 folds = {len(combinations) * 5} total runs.")
start_time = time.time()

# --- THE OUTER LOOP (PARAMETERS) ---
for i, (units, dropout, lr) in enumerate(combinations):
    print(f"testing combo {i + 1}/{len(combinations)} | units: {units}, dropout: {dropout}, lr: {lr}")

    fold_accuracies = []
    fold_f1_scores = []

    # --- THE INNER LOOP (DATA SPLITS) ---
    for fold, (train_index, test_index) in enumerate(kfold.split(padded_data)):
        # slice the data for this fold
        X_train, X_test = padded_data[train_index], padded_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # calculate weights for this specific slice of training data
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weight_dict = dict(enumerate(weights))

        # build the brain
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=64),
            Bidirectional(LSTM(units)),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])

        optimizer = Adam(learning_rate=lr)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        # train silently (verbose=0)
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
                  class_weight=weight_dict, callbacks=[early_stop], verbose=0)

        # predict and grade this fold
        predictions = model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # calculate accuracy and weighted f1-score
        acc = accuracy_score(y_test, predicted_classes)
        # using weighted average to account for the imbalanced dataset
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted_classes, average='weighted',
                                                                   zero_division=0)

        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)

        print(f"  fold {fold + 1}/5 -> acc: {acc:.4f} | f1: {f1:.4f}")

    # calculate averages for this parameter combination
    avg_acc = np.mean(fold_accuracies)
    avg_f1 = np.mean(fold_f1_scores)

    print(f">> combo {i + 1} results -> avg accuracy: {avg_acc:.4f} | avg f1-score: {avg_f1:.4f}\n")

    # save to our leaderboard
    all_results.append({
        'units': units, 'dropout': dropout, 'lr': lr,
        'avg_acc': avg_acc, 'avg_f1': avg_f1
    })

    # check if this is the new champion (judged by f1-score, which is better for imbalanced data)
    if avg_f1 > best_f1_score:
        best_f1_score = avg_f1
        best_params = {'units': units, 'dropout': dropout, 'lr': lr}

total_time = (time.time() - start_time) / 60

# print the final leaderboard
print("=" * 50)
print(f"NESTED CROSS-VALIDATION COMPLETE IN {total_time:.1f} MINUTES")
print(f"the absolute best parameters (based on highest average f1-score):")
print(best_params)
print(f"champion average f1-score: {best_f1_score:.4f}")