import pandas as pd

# loading the data adding column names 'sentiment' and 'text'
df = pd.read_csv('Data/all-data.csv', names=['sentiment', 'text'], encoding='latin-1')

# seeing what is in the data
print("first 5 rows of data:")
print(df.head())
print("\n")

# how many statements in each sentiment category
print("dataset breakdwon")
print(df['sentiment'].value_counts())