import pyprind
import pandas as pd
import numpy as np
import os

# Path to the dataset folder
basepath = 'aclImdb'


labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
rows = []

for s in ('train', 'test'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            rows.append({'review': txt, 'sentiment': labels[l]})
            pbar.update()


df = pd.DataFrame(rows, columns=['review', 'sentiment'])
print(f"Data loaded. Total reviews: {len(df)}")

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')  

df = pd.read_csv('movie_data.csv', encoding='utf-8')

try:
    df = df.rename(columns={"0": "review", "1": "sentiment"})
except KeyError:
    pass 

print("\nShuffled DataFrame:")
print(df.head(3))
