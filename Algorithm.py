import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple


class Algorithm:
    def read_and_preprocess(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)

        for col in df.columns[:-1].tolist():
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                except Exception:
                    df.drop(columns=[col], inplace=True)

        decision_col = df.columns[-1]
        if not pd.api.types.is_numeric_dtype(df[decision_col]):
            try:
                df[decision_col] = LabelEncoder().fit_transform(df[decision_col].astype(str))
            except Exception:
                pass

        df.reset_index(drop=True, inplace=True)
        return df

    def get_cut_candidates(self, values: np.ndarray, decisions: np.ndarray) -> List[float]:
        order = np.argsort(values)
        sorted_vals = values[order]
        sorted_decs = decisions[order]
        diffs = sorted_vals[1:] != sorted_vals[:-1]
        dec_diff = sorted_decs[1:] != sorted_decs[:-1]
        mask = diffs & dec_diff
        candidates = (sorted_vals[1:][mask] + sorted_vals[:-1][mask]) / 2.0
        return sorted(np.unique(candidates))

    def count_separated_pairs_attr(self, values: np.ndarray,
                                   decisions: np.ndarray,
                                   cuts: List[float]) -> int:
        N = len(values)
        classes, class_counts = np.unique(decisions, return_counts=True)
        total_pairs = N * (N - 1) // 2
        same_pairs = np.sum(class_counts * (class_counts - 1) // 2)
        diff_pairs = total_pairs - same_pairs

        if cuts:
            bins = np.digitize(values, cuts)
        else:
            bins = np.zeros_like(values, dtype=int)

        intra_diff = 0
        for b in np.unique(bins):
            mask = bins == b
            n_bin = np.sum(mask)
            if n_bin < 2:
                continue
            total_b = n_bin * (n_bin - 1) // 2
            _, cnts = np.unique(decisions[mask], return_counts=True)
            same_b = np.sum(cnts * (cnts - 1) // 2)
            intra_diff += (total_b - same_b)

        return diff_pairs - intra_diff

    def entropy(self, groups: List[np.ndarray]) -> float:
        total = sum(len(g) for g in groups)
        ent = 0.0
        for g in groups:
            if len(g) == 0:
                continue
            _, counts = np.unique(g, return_counts=True)
            probs = counts / counts.sum()
            ent -= (len(g) / total) * np.sum(probs * np.log2(probs))
        return ent

    def discretize_attribute(self, values: np.ndarray,
                             decisions: np.ndarray) -> List[float]:
        candidates = self.get_cut_candidates(values, decisions)
        chosen: List[float] = []
        best_sep = self.count_separated_pairs_attr(values, decisions, chosen)
        base_ent = self.entropy([decisions])

        for cut in candidates:
            test_cuts = sorted(chosen + [cut])
            sep = self.count_separated_pairs_attr(values, decisions, test_cuts)
            if sep > best_sep:
                best_sep, chosen = sep, test_cuts
            elif sep == best_sep:
                left = decisions[values <= cut]
                right = decisions[values > cut]
                if self.entropy([left, right]) < base_ent:
                    chosen = test_cuts
        return sorted(chosen)

    def discretize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
        decision_col = df.columns[-1]
        attrs = df.columns[:-1]
        cuts: Dict[str, List[float]] = {}

        for attr in attrs:
            vals = df[attr].values
            decs = df[decision_col].values
            cuts[attr] = self.discretize_attribute(vals, decs)

        new_df = pd.DataFrame()
        for attr in attrs:
            bins = [-np.inf] + cuts[attr] + [np.inf]
            intervals = pd.cut(df[attr], bins=bins, right=True)
            new_df[attr] = intervals.astype(str)
        new_df[decision_col] = df[decision_col].values

        return new_df, cuts

    def save_discretized(self, df: pd.DataFrame, in_filepath: str) -> str:
        dirpath, fname = os.path.split(in_filepath)
        out_name = "DISC_" + fname
        out_path = os.path.join(dirpath, out_name)
        df.to_csv(out_path, header=False, index=False)
        return out_path

    def discretize_file(self, filepath: str) -> Dict[str, any]:
        start = time.time()
        df = self.read_and_preprocess(filepath)
        n_objects = len(df)
        disc_df, cuts = self.discretize_dataframe(df)
        out_path = self.save_discretized(disc_df, filepath)
        duration = time.time() - start

        return {
            "input_file": filepath,
            "output_file": out_path,
            "num_objects": n_objects,
            "num_cuts": sum(len(v) for v in cuts.values()),
            "duration_sec": duration
        }
