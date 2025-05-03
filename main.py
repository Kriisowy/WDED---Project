import pandas as pd
import functools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

def debug_decorator(func):
    """Dekorator logujący wywołania funkcji i jej wynik"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Wywołanie funkcji: {func.__name__} z argumentami: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"Wynik funkcji {func.__name__}: {result}")
        return result
    return wrapper

def load_dataset(path):
    """Wczytuje dane z pliku CSV i usuwa brakujące wartości"""
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df

def separate_features_and_labels(df):
    """Oddziela cechy od etykiet klasowych (ostatnia kolumna jako etykieta)"""
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    return features, labels

def encode_labels(labels):
    """Koduje etykiety klasowe, jeśli są kategoryczne"""
    if labels.dtype == 'object':
        labels = LabelEncoder().fit_transform(labels)
    return labels

def encode_features(features):
    """Koduje cechy kategoryczne, jeśli występują"""
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = LabelEncoder().fit_transform(features[col])
    return features

def standardize_features(features):
    """Standaryzuje cechy"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return pd.DataFrame(features_scaled, columns=features.columns)

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

def merge_intervals(intervals, i, j):
    """Łączy dwa przedziały o indeksach i i j"""
    merged_indices = intervals[i].indices + intervals[j].indices
    new_interval = Interval(merged_indices)
    new_intervals = intervals[:i] + [new_interval] + intervals[i+1:j] + intervals[j+1:]
    return new_intervals

def bottom_up_discretization(intervals, labels):
    """
    Zachłanny algorytm dyskretyzacji wstępującej
    """
    best_intervals = intervals.copy()
    best_score = count_separated_pairs(best_intervals, labels)
    improved = True

    while improved and len(best_intervals) > 1:
        improved = False
        best_merge = None
        best_merge_score = -1

        for i in range(len(best_intervals) - 1):
            merged = merge_intervals(best_intervals, i, i+1)
            score = count_separated_pairs(merged, labels)

            if score > best_merge_score:
                best_merge_score = score
                best_merge = merged

        # jeśli najlepsze połączenie poprawia wynik – zastosuj
        if best_merge_score >= best_score:
            best_intervals = best_merge
            best_score = best_merge_score
            improved = True

    return best_intervals, best_score

if __name__ == "__main__":

    # Zmień ścieżkę na właściwą
    file_path = "iris.csv"

    # Wczytanie i przygotowanie danych
    df = load_dataset(file_path)
    features, labels = separate_features_and_labels(df)
    features = encode_features(features)
    labels = encode_labels(labels)
    features = standardize_features(features)

    # Budowa przedziałów i liczenie par
    intervals = build_initial_intervals(len(features))
    separated_count = count_separated_pairs(intervals, labels)
    print(f"Liczba par odseparowanych w początkowym układzie: {separated_count}")

    # Dyskretyzacja i wynik końcowy
    final_intervals, final_score = bottom_up_discretization(intervals, labels)
    print(f"Liczba przedziałów po dyskretyzacji: {len(final_intervals)}")
    print(f"Liczba odseparowanych par po dyskretyzacji: {final_score}")
