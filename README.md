# FakeNewsDetection

Binary classification demo using Logistic Regression to distinguish fake vs true news.

**Dataset**
https://drive.google.com/drive/folders/1uViefNaUP9xhgzR5O4xrAOWYJpuYBOjt
- **Fake:** [Fake.csv](Fake.csv)
- **True:** [True.csv](True.csv)

**Overview**
- **Goal:** Train a logistic regression classifier on the provided datasets and evaluate accuracy, precision, recall, and a confusion matrix.
- **Approach:** Use `pandas` to load and preprocess text data, `scikit-learn` for vectorization and logistic regression, and `matplotlib`/`seaborn` for visualizations.

**Recommended VS Code extensions**
- **Python:** Microsoft Python extension
- **Pylance:** fast type checking and language features
- **Jupyter:** run and edit notebooks
- **IntelliCode:** AI-assisted completions (optional)

**Setup (Windows PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Minimal `requirements.txt`**
```
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyterlab
ipykernel
```

Create a `requirements.txt` with the lines above, then run the `pip install` command shown.

**Quick usage**
- Start Jupyter Lab: `jupyter lab` and open a notebook such as `notebooks/logistic_regression.ipynb` (create one if missing).
- Or run a script: `python scripts/train_logistic.py` (example script should load `Fake.csv` + `True.csv`, preprocess, train, and save model).

**Example training snippet (scikit-learn)**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')
df = pd.concat([fake.assign(label=0), true.assign(label=1)], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
vec = TfidfVectorizer(max_features=20000)
X_train_t = vec.fit_transform(X_train)
X_test_t = vec.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_t, y_train)

pred = model.predict(X_test_t)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
```

**Next steps (suggested)**
- Add a `notebooks/logistic_regression.ipynb` with exploratory analysis and the training pipeline.
- Add `scripts/train_logistic.py` and `scripts/evaluate.py` for reproducible runs.

