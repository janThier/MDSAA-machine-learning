"""
Pipeline helpers and custom transformers for the 'Cars 4 You' ML project.

Central place for:
- Several feature engineering steps encapsulated in a transformer (CarFeatureEngineer)
- Group-based hierarchical imputation (GroupImputer)
- Mean estimation with M-estimate smoothing (m_estimate_mean)

Design goals
------------
- Keep all pipeline-related, sklearn-compatible helpers in one place.
- Make the main notebook focus on structure and modelling rather than lots of detailed code.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV, KFold


################################################################################
##################### Handle missing values (GroupImputer) #####################
################################################################################


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


# TODO this method is not used anywhere right? Can we remove it @Samu ~J
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


################################################################################
######################## Feature Engineering #######################
################################################################################

# docstring adden TODO
class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    This class calculates the metrics for the specific X (X = a fold within CV) and computes the engineered features with these metrics.
    """
    def __init__(self, ref_year=None):
        self.ref_year = ref_year

    def fit(self, X, y=None): # y is necessary because 3 arguments are given in pipeline # TODO figure out why this is the case
        X_ = X.copy()
        if self.ref_year is None:
            self.ref_year_ = X_['year'].max()
        else:
            self.ref_year_ = self.ref_year
        self.brand_median_age_ = (
            (self.ref_year_ - X_['year'])
            .groupby(X_['Brand'])
            .median()
            .to_dict()
        )
        self.model_freq_ = X_['model'].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        
        # # 1. Base Feature Creation: Car Age - Newer cars usually have higher prices, models prefer linear features
        age = self.ref_year_ - X['year']
        X['age'] = age

        # Miles per Year: Normalizes mileage by age (solves multicollinearity between year and mileage)
        X['miles_per_year'] = X['mileage'] / age.replace({0: np.nan})
        X['miles_per_year'] = X['miles_per_year'].fillna(X['mileage']) # if age is 0, just use mileage because that's the mileage it has driven so far in that year

        # Interaction Terms: Capture non-linear effects between engine and other numeric features
        X['age_x_engine'] = X['age'] * X['engineSize']
        X['mpg_x_engine']  = X['mpg'] * X['engineSize']

        # tax per engine
        X['tax_per_engine'] = X['tax'] / X['engineSize'].replace({0: np.nan}) # Catch Edge Case if engineSize=0 occurs in test set (e.g. for EVs)

        # MPG per engineSize to represent the efficiency
        X['mpg_per_engine'] = X['mpg'] / X['engineSize'].replace({0: np.nan}) # Catch Edge Case if engineSize=0 occurs in test set (e.g. for EVs)

        # 2. Model Frequency: Popular models tend to have stable demand and prices
        X['model_freq'] = X['model'].map(self.model_freq_).fillna(0.0)

        # 3. Create Interaction Features for anchor (relative positioning within brand/model)
        X['brand_fuel'] = X['Brand'].astype(str) + "_" + X['fuelType'].astype(str)
        X['brand_trans'] = X['Brand'].astype(str) + "_" + X['transmission'].astype(str)
        
        # 4. Relative Age (within brand): newer/older than brand median year
        X['age_rel_brand'] = X['age'] - X['Brand'].map(self.brand_median_age_)
        return X


################################################################################
######################## Helpers for FunctionTransformer #######################
################################################################################

# Adjust FunctionTransformer to expose feature names
class NamedFunctionTransformer(FunctionTransformer):
    def __init__(self, func=None, feature_names=None, **kwargs):
        # store as attribute so sklearn.get_params can access it
        self.feature_names = feature_names
        super().__init__(func=func, **kwargs)

    def get_feature_names_out(self, input_features=None):
        # if custom names specified, use them
        if self.feature_names is not None:
            return np.asarray(self.feature_names, dtype=object)
        # otherwise just pass through the input feature names
        return np.asarray(input_features, dtype=object)


# Callable function which uses the NamedFunctionTransformer to get feature names from a preprocessor
def get_feature_names_from_preprocessor(pre):
    feature_names = []
    for name, trans, cols in pre.transformers_:
        if name != 'remainder':
            if hasattr(trans, 'get_feature_names_out'):
                # for categorical OHE
                try:
                    feature_names.extend(trans.get_feature_names_out(cols))
                except:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
    return feature_names


# TODO warum benutzen wir das => auch bei pipeline adden
def to_float_array(x):
    """Convert input to float array."""
    return np.array(x, dtype=float)


######


# Define a function to use it here and potentially use it later for a final hyperparameter tuning after feature selection again
def model_hyperparameter_tuning(X_train, y_train, pipeline, param_dist, n_iter=100, splits=5):
    
    cv = KFold(n_splits=splits, shuffle=True, random_state=42) # 5 folds for more robust estimation

    # Randomized search setup
    model_random = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,                      # number of different hyperparameter combinations that will be randomly sampled and evaluated (more iterations = more thorough search but longer runtime)
        scoring={
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'r2': 'r2'
        },
        refit='mae', # Refit the best model based on MAE on the whole training set
        cv=cv,
        n_jobs=-2,
        random_state=42,
        verbose=3,
    )

    # Fit the search
    model_random.fit(X_train, y_train)

    mae = -model_random.cv_results_['mean_test_mae'][model_random.best_index_]
    mse = -model_random.cv_results_['mean_test_mse'][model_random.best_index_]
    rmse = np.sqrt(mse)
    r2 = model_random.cv_results_['mean_test_r2'][model_random.best_index_]

    print("Model Results (CV metrics):")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("Best Model params:", model_random.best_params_)

    return model_random.best_estimator_, model_random # return the best model