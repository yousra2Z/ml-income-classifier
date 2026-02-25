import joblib
from preprocess import load_and_preprocess
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#  Charger les données
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

#  Charger le modèle sauvegardé
model = joblib.load('models/xgb_income_model.joblib')

#  Prédictions sur validation set
y_pred = model.predict(X_val)

#  Metrics principales
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_val, y_pred))

#  Confusion matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#  Explorer quelques exemples mal classés
misclassified_idx = [i for i, (yt, yp) in enumerate(zip(y_val, y_pred)) if yt != yp]
misclassified_examples = X_val.iloc[misclassified_idx]
misclassified_examples['true_label'] = y_val.iloc[misclassified_idx]
misclassified_examples['pred_label'] = y_pred[misclassified_idx]

print("\nQuelques exemples mal classés :")
print(misclassified_examples.head(10))