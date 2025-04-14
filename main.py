import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_path = "iris.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

# Oddzielenie cech od etykiet klasowych (zakładamy: ostatnia kolumna = klasa)
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# Kodowanie etykiet klasowych, jeśli są kategoryczne
if labels.dtype == 'object':
    labels = LabelEncoder().fit_transform(labels)

# Kodowanie cech kategorycznych (jeśli są)
for col in features.columns:
    if features[col].dtype == 'object':
        features[col] = LabelEncoder().fit_transform(features[col])

# Standaryzacja cech (x' = (x - μ) / σ)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features = pd.DataFrame(features_scaled, columns=features.columns)

print("Standaryzowane cechy:\n", features.head())
print("\nEtykiety klas:\n", labels[:5])


#   :)
