import pandas as pd
from sklearn.preprocessing import LabelEncoder

# loading data
df = pd.read_csv('Data/all-data.csv', names=['sentiment', 'text'], encoding='latin-1')

print("original data:")
print(df.head())

# initialize label encoder:
encoder = LabelEncoder()

# convert text labels : ('positive', 'negative', 'neutral') into numbers
df['label'] = encoder.fit_transform(df['sentiment'])

print("\ndata after label encoding:")
print(df[['sentiment', 'label', 'text']].head(10))

# encoder mapping
print("\nencoder mapping:")
for i, item in enumerate(encoder.classes_):
    print(f"{item}  -->  {i}")