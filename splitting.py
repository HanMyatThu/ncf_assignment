import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./dataset/with_negatives.csv")

# Train (70%), Temp (30%)
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=42
)

# Val (15%), Test (15%)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)
# Save splits
train_df.to_csv("./dataset/train.csv", index=False)
val_df.to_csv("./dataset/val.csv", index=False)
test_df.to_csv("./dataset/test.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
