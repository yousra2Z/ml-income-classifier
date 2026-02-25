import joblib
from xgboost import XGBClassifier
from preprocess import load_and_preprocess
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Charger les données
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

# 2️⃣ Définir le modèle
model = XGBClassifier(
    n_estimators=100,       # nombre d’arbres
    max_depth=5,            # profondeur max des arbres
    learning_rate=0.1,      # taux d’apprentissage
    use_label_encoder=False, 
    eval_metric='logloss',  # métrique pour XGBoost
    random_state=42
)

# 3️⃣ Entraîner le modèle
model.fit(X_train, y_train)

# 4️⃣ Évaluer sur validation set
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# 5️⃣ Sauvegarder le modèle
joblib.dump(model, 'models/xgb_income_model.joblib')
print("Model saved to models/xgb_income_model.joblib")