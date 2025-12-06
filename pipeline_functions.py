"""
Pipeline helpers and custom transformers for the 'Cars 4 You' ML project.

Central place for:
- Group-based hierarchical imputation (GroupImputer)
- Mean estimation with M-estimate smoothing (m_estimate_mean)

Design goals
------------
- Keep all pipeline-related, sklearn-compatible helpers in one place.
- Make the main notebook focus on modelling rather than low-level plumbing.
- Make it easy for a new team member to understand how missing data and
  smoothed group statistics are handled.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Hierarchical imputer for numeric + categorical features.

    Idea
    ----
    For each row with a missing value, fill it using statistics from
    "similar" rows first, and only fall back to global statistics if needed.

    Hierarchy for numeric columns (num_cols):
        1) median per (group_cols[0], group_cols[1])     e.g. (Brand, model)
        2) median per group_cols[0]                      e.g. Brand
        3) global median across all rows

    Hierarchy for categorical columns (cat_cols):
        1) mode per (group_cols[0], group_cols[1])
        2) mode per group_cols[0]
        3) global mode across all rows

    Notes
    -----
    - `group_cols` are used only to define groups; they themselves are not imputed.
    - `num_cols` and `cat_cols` can be given explicitly (lists of column names).
      If None, they are inferred from the dtypes in `fit`.
    - The class is sklearn-compatible:
        * __init__ does NOT modify parameters (important for `clone`)
        * `fit` computes statistics
        * `transform` applies the imputation
    """

    def __init__(self,
                 group_cols=("Brand", "model"),
                 num_cols=None,
                 cat_cols=None,
                 fallback="__MISSING__"):
        """
        Parameters
        ----------
        group_cols : tuple/list of str
            Column names that define the hierarchy (e.g. ("Brand", "model")).

        num_cols : list[str] or None
            Numeric columns to impute. If None, inferred from dtypes in fit().

        cat_cols : list[str] or None
            Categorical columns to impute. If None, inferred as "the rest".

        fallback : str
            Value used if we cannot compute any mode for a categorical variable.
        """
        # IMPORTANT: do not modify params here (no list(...) etc).
        # Sklearn's clone() relies on these being exactly what the user passed.
        self.group_cols = group_cols
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.fallback = fallback

    # ---------- helpers ----------

    def _mode(self, s: pd.Series):
        """
        Deterministic mode helper.

        - Compute the most frequent non-null value.
        - If multiple values tie, Series.mode() returns them in order, we take .iloc[0].
        - If there is no valid mode (all NaN), return fallback token.
        """
        m = s.mode(dropna=True)
        if not m.empty:
            return m.iloc[0]
        return self.fallback

    def _get_group_series(self, df: pd.DataFrame, col_name: str) -> pd.Series:
        """
        Get the FIRST physical column with the given label from df.

        Reason
        ------
        - In some workflows, df.columns can contain duplicate labels
          (e.g. "Brand" appearing twice after some operations).
        - df["Brand"] would then raise "Grouper for 'Brand' not 1-dimensional".
        - By using np.where(df.columns == col_name)[0] we get *positions* and
          explicitly pick the first one.

        Raises
        ------
        ValueError if no column with that name exists.
        """
        matches = np.where(df.columns == col_name)[0]
        if len(matches) == 0:
            raise ValueError(f"GroupImputer: grouping column '{col_name}' not found in data.")
        return df.iloc[:, matches[0]]

    # ---------- fit ----------

    def fit(self, X, y=None):
        """
        Learn the group-level and global statistics from the training data.

        Steps
        -----
        1) Convert X to DataFrame and remember the original column order.
        2) Resolve which columns are numeric/categorical to impute.
        3) Build group keys (g0, g1) from group_cols (e.g. Brand, model).
        4) For numeric columns:
            - compute global medians
            - medians per g0 (e.g. per brand)
            - medians per (g0, g1) (e.g. per brand+model)
        5) For categorical columns:
            - global modes
            - modes per g0
            - modes per (g0, g1)
        """
        # Ensure we are working with a DataFrame
        df = pd.DataFrame(X).copy()

        # Remember the column order we saw during fit.
        # We will use this to align transform() inputs.
        self.feature_names_in_ = df.columns.to_list()

        # group_cols must contain at least one column name
        if self.group_cols is None or len(self.group_cols) == 0:
            raise ValueError("GroupImputer: at least one group column must be specified.")

        # Internal copy: always a simple list of group column names
        self.group_cols_ = list(self.group_cols)

        # Determine numeric columns to impute (internal num_cols_)
        if self.num_cols is None:
            # If not specified: all numeric columns except the group columns
            num_cols_all = df.select_dtypes(include="number").columns.tolist()
            self.num_cols_ = [c for c in num_cols_all if c not in self.group_cols_]
        else:
            # If specified: keep only those that exist in df
            self.num_cols_ = [c for c in self.num_cols if c in df.columns]

        # Determine categorical columns to impute (internal cat_cols_)
        if self.cat_cols is None:
            # If not specified: all non-group, non-numeric columns
            self.cat_cols_ = [
                c for c in df.columns
                if c not in self.group_cols_ + self.num_cols_
            ]
        else:
            # If specified: keep only those that exist in df
            self.cat_cols_ = [c for c in self.cat_cols if c in df.columns]

        # Build group key series based on the current df
        # g0 = first grouping column (e.g. Brand)
        g0 = self._get_group_series(df, self.group_cols_[0])

        # g1 = second grouping column (e.g. model), optional
        g1 = None
        if len(self.group_cols_) > 1:
            g1 = self._get_group_series(df, self.group_cols_[1])

        # ---- numeric statistics ----
        if self.num_cols_:
            # Extract the numeric columns to impute
            num_df = df[self.num_cols_].copy()

            # 3) Global median per numeric column (fallback for any group with no stats)
            self.num_global_ = num_df.median(numeric_only=True)

            # 2) Median per first-level group (g0, e.g. Brand)
            num_first = num_df.copy()
            num_first["_g0"] = g0.values  # temporary group key column
            self.num_first_ = (
                num_first
                .groupby("_g0", dropna=True)  # group by Brand (or group_cols_[0])
                .median(numeric_only=True)
            )

            # 1) Median per pair (g0, g1), e.g. (Brand, model)
            if g1 is not None:
                num_pair = num_df.copy()
                num_pair["_g0"] = g0.values
                num_pair["_g1"] = g1.values
                self.num_pair_ = (
                    num_pair
                    .groupby(["_g0", "_g1"], dropna=True)  # group by (Brand, model)
                    .median(numeric_only=True)
                )
            else:
                # If there is no second grouping column, we keep an empty DataFrame
                self.num_pair_ = pd.DataFrame()
        else:
            # If there are no numeric columns to impute, store empty stats
            self.num_global_ = pd.Series(dtype="float64")
            self.num_first_ = pd.DataFrame()
            self.num_pair_ = pd.DataFrame()

        # ---- categorical statistics ----
        if self.cat_cols_:
            # Extract the categorical columns to impute
            cat_df = df[self.cat_cols_].copy()

            # 3) Global mode per categorical column
            self.cat_global_ = pd.Series(
                {c: self._mode(cat_df[c]) for c in self.cat_cols_},
                dtype="object"
            )

            # 2) Mode per first-level group (g0)
            cat_first = cat_df.copy()
            cat_first["_g0"] = g0.values
            self.cat_first_ = (
                cat_first
                .groupby("_g0", dropna=True)
                .agg(lambda s: self._mode(s))  # apply deterministic mode per column
            )

            # 1) Mode per pair (g0, g1)
            if g1 is not None:
                cat_pair = cat_df.copy()
                cat_pair["_g0"] = g0.values
                cat_pair["_g1"] = g1.values
                self.cat_pair_ = (
                    cat_pair
                    .groupby(["_g0", "_g1"], dropna=True)
                    .agg(lambda s: self._mode(s))
                )
            else:
                 self.cat_pair_ = pd.DataFrame()
        else:
            # No categorical columns to impute
            self.cat_global_ = pd.Series(dtype="object")
            self.cat_first_ = pd.DataFrame()
            self.cat_pair_ = pd.DataFrame()

        return self

    # ---------- transform ----------

    def transform(self, X):
        """
        Apply hierarchical imputation to new data.

        Steps
        -----
        1) Convert input to DataFrame and align columns to what fit() saw.
        2) Rebuild group keys g0, g1 from the current data.
        3) For each numeric column with missing values:
            - try pair-level median (g0, g1)
            - then brand-level median (g0)
            - then global median
        4) Same for categorical columns with modes.
        """
        # Ensure DataFrame and align with training column order
        df = pd.DataFrame(X).copy()
        df = df.reindex(columns=self.feature_names_in_)

        # Build group key series again from current df
        g0 = self._get_group_series(df, self.group_cols_[0])
        g1 = None
        if len(self.group_cols_) > 1:
            g1 = self._get_group_series(df, self.group_cols_[1])

        # ---- numeric imputation ----
        if hasattr(self, "num_cols_") and self.num_cols_:
            # Ensure numeric dtype for numeric columns (to safely assign medians)
            df[self.num_cols_] = df[self.num_cols_].astype("float64")

            # Only consider columns that actually have missing values
            to_impute_num = [c for c in self.num_cols_ if df[c].isna().any()]

            if to_impute_num:
                # 1) pair-level imputation: per (g0, g1)
                if g1 is not None and not self.num_pair_.empty:
                    # Build a small DF with group keys for each row
                    key_df = pd.DataFrame({"_g0": g0.values, "_g1": g1.values})
                    # Turn the multi-index num_pair_ into a DF with _g0,_g1 as columns
                    med_df = self.num_pair_.reset_index()
                    # Left-join: each row gets the medians for its (g0,g1) if available
                    joined = key_df.merge(med_df, on=["_g0", "_g1"], how="left")

                    for col in to_impute_num:
                        if col not in self.num_pair_.columns:
                            # Column not part of pair-level stats, skip
                            continue
                        # Mask: rows where the original value is NaN but we have a pair-level median
                        mask = df[col].isna() & joined[col].notna()
                        # Fill with the corresponding pair-level median
                        df.loc[mask, col] = joined.loc[mask, col]

                # 2) first-level imputation: per g0 only
                if not self.num_first_.empty:
                    key1 = pd.DataFrame({"_g0": g0.values})
                    med1 = self.num_first_.reset_index()
                    joined1 = key1.merge(med1, on="_g0", how="left")

                    for col in to_impute_num:
                        if col not in self.num_first_.columns:
                            continue
                        # Now fill remaining NaNs using brand-level medians
                        mask = df[col].isna() & joined1[col].notna()
                        df.loc[mask, col] = joined1[col]

                # 3) global median fallback
                for col in to_impute_num:
                    if col in self.num_global_:
                        df[col] = df[col].fillna(self.num_global_[col])

        # ---- categorical imputation ----
        if hasattr(self, "cat_cols_") and self.cat_cols_:
            # Only consider columns that actually have missing values
            to_impute_cat = [c for c in self.cat_cols_ if df[c].isna().any()]

            if to_impute_cat:
                # 1) pair-level imputation: per (g0, g1)
                if g1 is not None and not self.cat_pair_.empty:
                    key_df = pd.DataFrame({"_g0": g0.values, "_g1": g1.values})
                    mode_df = self.cat_pair_.reset_index()
                    joined = key_df.merge(mode_df, on=["_g0", "_g1"], how="left")

                    for col in to_impute_cat:
                        if col not in self.cat_pair_.columns:
                            continue
                        mask = df[col].isna() & joined[col].notna()
                        df.loc[mask, col] = joined[col]

                # 2) first-level imputation: per g0 only
                if not self.cat_first_.empty:
                    key1 = pd.DataFrame({"_g0": g0.values})
                    mode1 = self.cat_first_.reset_index()
                    joined1 = key1.merge(mode1, on="_g0", how="left")

                    for col in to_impute_cat:
                        if col not in self.cat_first_.columns:
                            continue
                        mask = df[col].isna() & joined1[col].notna()
                        df.loc[mask, col] = joined1[col]

                # 3) global mode fallback (or fallback token)
                for col in to_impute_cat:
                    if col in self.cat_global_:
                        df[col] = df[col].fillna(self.cat_global_[col])
                    else:
                        df[col] = df[col].fillna(self.fallback)

        # Return a DataFrame (sklearn will accept this, or you can call .values() if needed)
        return df

    def get_feature_names_out(self, input_features=None):
        """
        Make the transformer compatible with sklearn's feature-name getting.

        - If called without arguments, return the original feature names seen in fit().
        - This is mostly useful when GroupImputer is at the top of a Pipeline and
          later steps want to introspect feature names.
        """
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)


def m_estimate_mean(sum_, prior, count, m=50):
    """
    Posterior mean with M-estimate smoothing.

    This function implements the standard M-estimator used to smooth noisy
    category-wise statistics (e.g. average price per model). Small groups are
    pulled towards a prior, large groups are closer to their empirical mean.

    Parameters
    ----------
    sum_ : float or pd.Series
        Sum of target values per group.
    prior : float or pd.Series
        Prior "pseudo-sum". In this project, we use `global_mean * count`,
        mirroring the original notebook implementation.
    count : float or pd.Series
        Number of observations in the group.
    m : int, default 50
        Smoothing strength. Larger `m` means stronger pull towards the prior.

    Returns
    -------
    float or pd.Series
        Smoothed mean estimate.
    """
    return (sum_ + m * prior) / (count + m)
