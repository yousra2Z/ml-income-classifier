import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_and_preprocess():
    # Colonnes du dataset
    column_names = [
        'age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain','capital-loss',
        'hours-per-week','native-country','income'
    ]
    
    # Chemin vers les fichiers
    data_path = os.path.join('data', 'adult.data')
    
    # Charger le dataset
    df = pd.read_csv(data_path, names=column_names, sep=',', skipinitialspace=True)
    
    # Remplacer '?' par NaN et supprimer les lignes manquantes
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    # Encoder les features catégoriques
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Séparer features et target
    X = df.drop('income', axis=1)
    y = df['income']
    
    # Split train / val / test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Test rapide
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()
    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)