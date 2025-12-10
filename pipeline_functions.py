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

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.feature_selection import mutual_info_regression
from sklearn.exceptions import NotFittedError

import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport




class DebugTransformer(BaseEstimator, TransformerMixin):
    """Transformer that prints the data shape and optionally the data itself for easier debugging and understanding"""
    
    def __init__(self, name="Debug", show_data=False, y_data_profiling=False, n_rows=5):
        self.name = name
        self.show_data = show_data
        self.y_data_profiling = y_data_profiling
        self.n_rows = n_rows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print(f"\n--- {self.name} ---")
        print(f"Shape: {X.shape}")
        print(f"Type: {type(X)}")
        
        if self.show_data:
            if isinstance(X, pd.DataFrame):
                print(f"\nFirst {self.n_rows} rows:")
                display(X.head(self.n_rows))
                display(X.describe(include='all').T)
                if(self.y_data_profiling):
                    print("\nGenerating data profiling report...")
                    profile = ProfileReport(
                        X,
                        title='Car Data Profiling Report',
                        correlations={
                            "pearson": {"calculate": True},
                            "spearman": {"calculate": False},
                            "kendall": {"calculate": False},
                            "phi_k": {"calculate": False},
                            "cramers": {"calculate": False},
                        },
                    )
                    profile.to_notebook_iframe()

            else: # Edge-case for numpy array after column transformer
                print(f"\nFirst {self.n_rows} rows:")
                display(X[:self.n_rows])
        
        return X


################################################################################
##################### Handle missing values (GroupImputer) #####################
################################################################################


class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Hierarchical imputer for numeric + categorical features.

    Idea
    ----
    We have to  compute the median value for the train dataset and fill the missing values in train, validation and test set with the median from the train dataset.

    For each row with a missing value, fill it using statistics from "similar" rows first, and only fall back to global statistics if needed.

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
        
        self.brand_mean_age_ = (
            (self.ref_year_ - X_['year'])
            .groupby(X_['Brand'])
            .mean()
            .to_dict()
        )
        self.model_mean_age_ = (
            (self.ref_year_ - X_['year'])
            .groupby(X_['model'])
            .mean()
            .to_dict()
        )

        self.model_mean_mileage_ = (
            X_['mileage']
            .groupby(X_['model'])
            .mean()
            .to_dict()
        )

        self.model_mean_engineSize_ = (
            X_['engineSize']
            .groupby(X_['model'])
            .mean()
            .to_dict()
        )

        self.model_freq_ = X_['model'].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()

        # Available num features:
        # orig_numeric_features = ["year", "mileage", "tax", "mpg", "engineSize", "previousOwners"] # though previousOwners has now correlations
        # orig_categorical_features = ["Brand", "model", "transmission", "fuelType"]
        # unused_features = ['hasDamage', 'paintQuality']
        
        ############ 1. Base Feature Creation:
        # Car Age - Newer cars usually have higher prices, models prefer linear features
        age = self.ref_year_ - X['year']
        X['age'] = age


        ############ 2. Interaction effects to capture non-additive information (learn conditional relationships and potentially skyrocket their importance):
        ############ - It helps to solve multicolinearity between features by combining them into one feature creating a new signal
        ############ => Only spearman correlations > 0.2 are regarded # TODO is that a good approach or is pearson maybe more suited in this case?
        ############ - Use Multiplication if we think two features "boost" each other (e.g., Length*Width = Area).
        ############ - Use Division if we need to "fairly compare" items of different sizes (e.g., Cost/Weight = Price per kg)
        ############ -> Mult or Div has to be chosen based on the logic of the relationship
        ###### Multiplication: The Amplifier (model synergy or joint occurrence: "The presence of A makes B more effective") -> capture the simultaneous impact of two things

        X['mpg_x_engine'] = X['mpg'] * X['engineSize']        # TODO multiplication kind of cancels the signal (10mpg * 9es = 90 , 45mpg * 2es = 90 -> big and small cars treated the same) (However, it improves performance)
        
        # Removed because of high multicolinearity and lower corr with price: X['mileage_x_mpg']          = X['mileage'] * X[s'mpg'] # Higher mileage cars tend to have lower MPG (people drive lower mpg cars more often) -> amplify effect
        # Add 1 to age because if age is 0 (this year) the value would be lost otherwise
        X['engine_x_age'] = X['engineSize'] * (X['age']+1)      # Highlight the aspect of old cars with big engines for that time which were very valuable and might therefore still be valuable
        X['mileage_x_age'] = X['mileage'] * (X['age']+1)        # Both are negatively correlated with price -> amplify effect to identify a stronger signal of old abused cars that are probably less valuable
        X['mpg_x_age'] = X['mpg'] * (X['age']+1)                # Older cars tend to have higher MPG -> amplify effect 
        X['tax_x_age'] = X['tax'] * (X['age']+1)                

        ###### Division: The Normalizer (create ratios, rates, or efficiency metrics: "How much of A do we have per unit of B?") -> removes the influence of the divisor        
        ### Normalize by Age to capture how features behave relative to the car's age
        
        # Miles per Year: Normalizes mileage by age -> reveals how much a car was really driven per year
        X['miles_per_year'] = X['mileage'] / (X['age']+1)               # Add 1 to age because if age is 0 (this year) the division would fail (dont impute with 1 bc then its the same as 1 year old instead of being from this year)

        # tax normalized by engine and/or per mpg to focus on the tax of the car regardless of the other factor (prefered to keep engine because engine is the cause and mpg the effect but corr with price of mpg was higher (0.46 and -0.06))
        X['tax_per_mpg'] = X['tax'] / X['mpg']                          # No 0-handling necessary because mpg cannot be 0 (we only keep values from 5-150 and impute the others)

        # engine per mpg creates a signal for sports/luxury cars that have a high engine size but low mpg (high performance cars) -> these cars are usually more valuable
        X['engine_per_mpg'] = X['engineSize'] / X['mpg']              # No 0-handling necessary because engineSize cannot be 0 (we only keep values from 0.6–9.0 and impute the others)


        ############ Create Interaction Features for anchor (relative positioning within brand/model)
        X['brand_fuel'] = X['Brand'].astype(str) + "_" + X['fuelType'].astype(str)
        X['brand_trans'] = X['Brand'].astype(str) + "_" + X['transmission'].astype(str)


        ############ Features based on learned statistics from the available data fold in the fit() method:
        X['model_freq'] = X['model'].map(self.model_freq_).fillna(0.0) # Model Frequency: Popular models tend to have stable demand and prices
        

        ############ Relative Age (within brand): newer/older than brand median year
        X['age_rel_brand'] = X['age'] - X['Brand'].map(self.brand_mean_age_) # use mean instead of median because most of the values were 0 otherwise
        X['age_rel_model'] = X['age'] - X['model'].map(self.model_mean_age_)

        X['engine_rel_model'] = X['engineSize'] / X['model'].map(self.model_mean_engineSize_) # engine size relative to model mean engine size

        # TODO tax divided by mean model price (affordability within model) # Before that: check whether road tax varies per model
        return X

################################################################################
######################## Feature Selection #######################
################################################################################
# Custom Majority Voter Transformer to prevent data leakage
# Explanation of Leakage Prevention:
# 0) Function call: pass the preprocessor_pipe (which contains the Majority Voter) into RandomizedSearchCV:
# 1) Splitting: The search CV splits the data into Train and Validation folds.
# 2) Fitting: It calls .fit() on your pipeline using only the Train fold.
# 3) Voting: The custom MajorityVoteSelectorTransformer runs inside the pipeline. It sees only the Train fold. It calculates votes and selects features based only on that fold.
# 4) Transformation: It transforms the Validation fold based on the features selected in step 3.
# ==> Leakage Free: Since the Validation fold was never used to decide which features to keep, there is no leakage.
#
# Explanation of why it is not a problem for the final refit that different features might have been selected in different folds:
# 0) Final refit is called on best hyperparameters found during CV.
# 1) The MajorityVoteSelectorTransformer sees the entire training data during final refit.
# 2) It calculates votes and selects features based on the entire training data. (This is done without hp-tuning now because the hps are fixed.)
# 3) It transforms the entire training data based on the features selected in step 2.
# ==> No Problem: Although different folds might have selected different features during CV, the final refit uses the entire training data to select only one final set of features (which might vary from previous features selected in the folds but thats not a problem).


class MajorityVoteSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that fits multiple fs algorithms and keeps features selected by at least 'min_votes' selectors.
    """
    def __init__(self, selectors=None, min_votes=2):
        """
        args:
            selectors: list of sklearn feature selector objects.
            min_votes: int, minimum number of selectors that must agree to keep a feature.
        """
        self.selectors = selectors
        self.min_votes = min_votes
        self.fitted_selectors_ = []
        self.support_mask_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Validate inputs
        if not self.selectors:
            raise ValueError("You must provide a list of selectors.")
        
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns) # store input feature names if available for the get_feature_names_out method
        
        self.fitted_selectors_ = []
        votes = np.zeros(X.shape[1])

        # Loop through each selector, fit it, and tally votes
        for selector in self.selectors:
            # Clone to ensure we don't modify the original objects
            sel = clone(selector)
            sel.fit(X, y)
            self.fitted_selectors_.append(sel)
            
            # Get boolean mask of selected features (True means that they are selected)
            mask = sel.get_support()
            votes += mask.astype(int)
        
        # Create the final mask: True if votes >= threshold
        self.support_mask_ = votes >= self.min_votes
        
        return self

    def transform(self, X):
        if self.support_mask_ is None:
            raise NotFittedError("This MajorityVoteSelectorTransformer instance is not fitted yet.")
        
        # If X is a DataFrame, keep column names for better debugging. Otherwise return numpy array
        if hasattr(X, "loc"):
            return X.loc[:, self.support_mask_]
        
        return X[:, self.support_mask_]

    def get_feature_names_out(self, input_features=None):
        """This method is called by sklearn when set_output(transform='pandas') is on (like in our debug-transformer)"""
        # If we stored names during fit, use them as default
        if input_features is None and self.feature_names_in_ is not None:
            input_features = self.feature_names_in_
        
        # If input_features is still None, sklearn generates x0, x1...
        if input_features is None:
            # If no names provided, generate generic indices
            return np.array([f"x{i}" for i in range(len(self.support_mask_))])[self.support_mask_]
        
        return np.array(input_features)[self.support_mask_]


# This custom Correlation Threshold Selector selects features based on their absolute correlation with the target variable.
class CorrelationThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.mask_ = None

    def fit(self, X, y):
        
        correlations = []
        # Handle both cases whether X is DataFrame or Numpy
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        for i in range(X_arr.shape[1]):
            # Simple absolute Pearson correlation
            corr = np.corrcoef(X_arr[:, i], y_arr)[0, 1]
            correlations.append(abs(corr))
            
        self.mask_ = np.array(correlations) > self.threshold
        return self

    def get_support(self):
        return self.mask_

    def transform(self, X):
        return X[:, self.mask_]


class MutualInfoThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01, n_neighbors=10, random_state=42):
        """
        threshold: Minimum MI score required to keep a feature.
        n_neighbors: Parameter for the internal MI calculation.
        """
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.mask_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        # Save input feature names if available (for pandas output)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array([str(c) for c in X.columns])
        else:
            self.feature_names_in_ = None

        # Calculate Mutual Information Scores (assume X is all numeric here (handled by your previous pipeline steps))
        self.scores_ = mutual_info_regression(
            X, 
            y, 
            n_neighbors=self.n_neighbors, 
            random_state=self.random_state
        )
        
        # 3. Create Mask based on Threshold
        self.mask_ = self.scores_ > self.threshold
        return self

    def transform(self, X):
        if self.mask_ is None:
            raise NotFittedError("Selector not fitted.")
        
        # Handle DataFrame vs Numpy (depending on the setting in the pipeline (e.g. df in the debug transformer))
        if hasattr(X, "loc"):
            return X.loc[:, self.mask_]
        return X[:, self.mask_]

    def get_support(self):
        return self.mask_
    

def plot_selector_agreement(majority_selector, feature_names):
    """"Function to plot a heatmap showing which features were selected by which voters in the MajorityVoteSelectorTransformer."""

    feature_names = np.asarray(feature_names)
    
    # Collect masks from each fitted selector and convert to int for boolean values
    data = {}
    for i, selector in enumerate(majority_selector.fitted_selectors_):
        
        selector_name = selector.__class__.__name__
        
        # If it's a SelectFromModel, add the base estimator name
        if hasattr(selector, 'estimator'):
            base_estimator_name = selector.estimator.__class__.__name__
            selector_name = f"{selector_name}({base_estimator_name})"
        
        # Convert boolean mask to int (0/1) for heatmap
        data[selector_name] = selector.get_support().astype(int)
    
    # Create DataFrame and add the kept and total votes columns
    df_votes = pd.DataFrame(data, index=feature_names)
    df_votes['Total Votes'] = df_votes.sum(axis=1).astype(int)
    df_votes['KEPT'] = (df_votes['Total Votes'] >= majority_selector.min_votes).astype(int)
    
    # Plot Heatmap (all columns are now int, so seaborn can handle them)
    plt.figure(figsize=(10, len(feature_names) * 0.4))
    sns.heatmap(df_votes, annot=True, cbar=False, cmap="Blues", linewidths=0.5, fmt='d')
    plt.title("Feature Selection Agreement")
    plt.tight_layout()
    plt.show()




################################################################################
######################## Helpers for FunctionTransformer #######################
################################################################################

# Adjust FunctionTransformer to expose feature names
class NamedFunctionTransformer(FunctionTransformer): # TODO check if this is really necessary or can be removed when implementing get_feature_names_out everywhere clean
    def __init__(self, func=None, feature_names=None, **kwargs):
        # store as attribute so sklearn.get_params can access it
        self.feature_names = feature_names
        super().__init__(func=func, **kwargs)

    def get_feature_names_out(self, input_features=None):
        # Fixed: use self.feature_names instead of self._feature_names
        if self.feature_names is not None:
            return np.asarray(self.feature_names, dtype=object)
        # IMPORTANT: Return input_features if no custom names provided
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        # Last resort: return generic names based on n_features_in_
        if hasattr(self, 'n_features_in_'):
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        # Cannot determine feature names
        return None


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