import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

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

class Interval:
    def __init__(self, indices):
        self.indices = indices  # lista indeksów oryginalnych próbek

def build_initial_intervals(n_samples):
    """Tworzy po jednym przedziale dla każdej próbki"""
    return [Interval([i]) for i in range(n_samples)]

def count_separated_pairs(intervals, labels):
    """
    Optymalna wersja: liczy liczbę par próbek z różnych klas, w różnych przedziałach.
    """
    class_counts = []

    # Liczymy ile jest próbek każdej klasy w każdym przedziale
    for interval in intervals:
        interval_labels = labels[interval.indices]
        class_count = Counter(interval_labels)
        class_counts.append(class_count)

    separated = 0
    # Porównujemy przedziały parami
    for i in range(len(class_counts)):
        for j in range(i + 1, len(class_counts)):
            for class_i, count_i in class_counts[i].items():
                for class_j, count_j in class_counts[j].items():
                    if class_i != class_j:
                        separated += count_i * count_j
    return separated

# --- Twoja część: budujemy przedziały i liczymy odseparowane pary ---
intervals = build_initial_intervals(len(features))
separated_count = count_separated_pairs(intervals, labels)
print(f"Liczba par odseparowanych w początkowym układzie: {separated_count}")