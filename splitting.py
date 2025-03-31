import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./dataset/with_negatives.csv")

# shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# split into Train (70%) and temp (30%)
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)

# split again Validation (15%) and Test (15%) from the temp
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

train_df.to_csv("./dataset/train.csv", index=False)
val_df.to_csv("./dataset/val.csv", index=False)
test_df.to_csv("./dataset/test.csv", index=False)

print("Split complete!")
