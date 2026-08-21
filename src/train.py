
import pandas as pd, yaml, pickle, json, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

with open("params.yaml") as f:
    params = yaml.safe_load(f)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = train.drop("target", axis=1)
y_train = train["target"]
X_test = test.drop("target", axis=1)
y_test = test["target"]

model = LogisticRegression(C=params["model"]["C"], max_iter=params["model"]["max_iter"])
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

# metrics
with open("metrics.json","w") as f:
    json.dump({"accuracy": acc}, f, indent=2)

# ROC plot data (one-vs-rest)
lb = LabelBinarizer()
y_bin = lb.fit_transform(y_test)
probs = model.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
os.makedirs("plots", exist_ok=True)
pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv("plots/roc.csv", index=False)

with open("model.pkl","wb") as f:
    pickle.dump(model,f)
