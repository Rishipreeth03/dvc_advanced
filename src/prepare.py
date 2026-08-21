
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml, os

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/iris.csv")

train, test = train_test_split(df, test_size=params["split"]["test_size"], random_state=42)

os.makedirs("data", exist_ok=True)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
