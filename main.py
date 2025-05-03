import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(path):
    df = pd.read_csv(path).dropna()
    return df


def separate_features_and_labels(df):
    return df.iloc[:, :-1], df.iloc[:, -1]


def encode_labels(labels):
    return LabelEncoder().fit_transform(labels) if labels.dtype == 'object' else labels


def encode_features(features):
    return features.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)


def standardize_features(features):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(features), columns=features.columns)


class Interval:
    def __init__(self, indices):
        self.indices = indices


def build_initial_intervals(n_samples):
    return [Interval([i]) for i in range(n_samples)]


def count_separated_pairs(intervals, labels):
    class_counts = [np.bincount(labels[interval.indices], minlength=len(np.unique(labels))) for interval in intervals]
    return sum(np.dot(c1, c2) for i, c1 in enumerate(class_counts) for c2 in class_counts[i + 1:])


def merge_intervals(intervals, i, j):
    new_intervals = intervals[:i] + [Interval(intervals[i].indices + intervals[j].indices)] + intervals[
                                                                                              i + 1:j] + intervals[
                                                                                                         j + 1:]
    return new_intervals


def bottom_up_discretization(intervals, labels):
    best_intervals, best_score = intervals, count_separated_pairs(intervals, labels)

    while len(best_intervals) > 1:
        merge_candidates = [(i, i + 1, count_separated_pairs(merge_intervals(best_intervals, i, i + 1), labels)) for i
                            in range(len(best_intervals) - 1)]
        i, j, best_merge_score = max(merge_candidates, key=lambda x: x[2])

        if best_merge_score >= best_score:
            best_intervals = merge_intervals(best_intervals, i, j)
            best_score = best_merge_score
        else:
            break

    return best_intervals, best_score


if __name__ == "__main__":
    file_path = "iris.csv"
    df = load_dataset(file_path)
    features, labels = separate_features_and_labels(df)
    features, labels = encode_features(features), encode_labels(labels)
    features = standardize_features(features)

    intervals = build_initial_intervals(len(features))
    print(f"Początkowa liczba odseparowanych par: {count_separated_pairs(intervals, labels)}")

    final_intervals, final_score = bottom_up_discretization(intervals, labels)
    print(f"Po dyskretyzacji: liczba przedziałów = {len(final_intervals)}, liczba odseparowanych par = {final_score}")
