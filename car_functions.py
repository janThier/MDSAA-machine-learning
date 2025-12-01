"""
Utility functions and custom transformers for the 'Cars 4 You' ML project.

Central place for:
- Raw data cleaning (clean_car_dataframe)
- Target encoding helpers (cv_target_encode)
- Custom imputers (GroupMedianImputer)
- Group-based price anchor features (add_price_anchor_features)
- Generic metric printing helper (print_metrics)

Design goals
------------
- Keep all deterministic, non-leakage preprocessing steps in one place.
- Make the main notebook focus on modelling.
- Make it easy for a new team member or outsider to understand what each step does.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

##########################################################################################################################################################

def clean_car_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw car data. This function is meant to be the single entry point for turning the raw CSV into a consistent, analysis-ready dataframe. It does:
    - Type coercion for numeric columns
    - Simple outlier removal via range checks (values outside are set to NaN)
    - Canonicalisation of messy string categories (e.g. 'bmw', 'BM' -> 'BMW')
    - Basic brand inference from model when Brand is missing

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with the original project columns (carID, Brand, model, year,
        mileage, tax, fuelType, mpg, engineSize, paintQuality%, previousOwners,
        hasDamage, price).

    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with:
        - `carID` set as index (no duplicates expected)
        - Numeric columns coerced to numeric and obvious outliers set to NaN
        - Integer-like columns stored as nullable Int64 (can hold NaN)
        - Brand/model/transmission/fuelType spellings normalised to canonical labels
        - `Brand` inferred from `model` where possible

    Notes
    -----
    - This function deliberately does not touch the target column (price),
      so it can be safely used before any train/validation split without causing
      target leakage.
    - Thresholds and ranges (e.g. mpg 5-150) are based on domain knowledge +
      the project EDA and can be adjusted in one place here if needed.
    """
    
    df = df.copy()

    # NUMERICAL COLUMNS:

    # column carID (set as index, has no duplicates)
    df = df.set_index('carID')

    # column year: 1970 to 2020
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df.loc[~df['year'].between(1970, 2020), 'year'] = np.nan  # TODO undo to 2024 for best model
    df['year'] = np.floor(df['year']).astype('Int64')

    # column mileage: -58.000 to 323.000
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df.loc[df['mileage'] < 0, 'mileage'] = np.nan
    df['mileage'] = np.floor(df['mileage']).astype('Int64')

    # column tax: -91 to 580
    df['tax'] = pd.to_numeric(df['tax'], errors='coerce')
    df.loc[df['tax'] < 0, 'tax'] = np.nan
    df['tax'] = np.floor(df['tax']).astype('Int64')

    # column mpg: -43 to 470
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')
    df.loc[~df['mpg'].between(5, 150), 'mpg'] = np.nan
    df['mpg'] = np.floor(df['mpg']).astype('Int64')

    # column engineSize: -0.1 to 6.6
    df['engineSize'] = pd.to_numeric(df['engineSize'], errors='coerce')
    df.loc[~df['engineSize'].between(0.6, 9.0), 'engineSize'] = np.nan
    df['engineSize'] = df['engineSize'].round(1)

    # column paintQuality%: 1.6 to 125
    # rename once here so the rest of the code can consistently use `paintQuality`
    df = df.rename(columns={'paintQuality%': 'paintQuality'})
    df['paintQuality'] = pd.to_numeric(df['paintQuality'], errors='coerce')
    df.loc[~df['paintQuality'].between(5, 100), 'paintQuality'] = np.nan  # TODO undo to (70,100) for best model
    df['paintQuality'] = np.floor(df['paintQuality']).astype('Int64')

    # column previousOwners: -2.3 to 6.2
    df['previousOwners'] = pd.to_numeric(df['previousOwners'], errors='coerce')
    df.loc[df['previousOwners'] < 0, 'previousOwners'] = np.nan
    df['previousOwners'] = np.floor(df['previousOwners']).astype('Int64')

    # column hasDamage (0/nan -> we cannot say that nan means damaged so this feature will be ignored, convert to Int)
    df['hasDamage'] = pd.to_numeric(df['hasDamage'], errors='coerce').astype('Int64')


    # CATEGORICAL COLUMNS:
    #   the idea is always:
    #   - normalise case / whitespace
    #   - map any known typo/variant into a canonical label using a reverse dict

    # column brand: map all incorrect spellings to the right brand
    brand_map = {
        'VW': ['VW', 'V', 'W', 'vw', 'v', 'w'],
        'Toyota': ['Toyota', 'TOYOTA', 'Toyot', 'toyota', 'oyota', 'TOYOT', 'OYOTA'],
        'Audi': ['Audi', 'AUDI', 'A', 'udi', 'Aud', 'audi', 'AUD', 'UDI'],
        'Ford': ['Ford', 'FORD', 'For', 'ord', 'for', 'ORD', 'or', 'FOR'],
        'BMW': ['BMW', 'bmw', 'MW', 'BM', 'mw'],
        'Skoda': ['Skoda', 'SKODA', 'Skod', 'koda', 'SKOD', 'kod', 'skoda', 'skod', 'KODA'],
        'Opel': ['Opel', 'OPEL', 'Ope', 'opel', 'OPE', 'pel', 'pe', 'PEL', 'ope'],
        'Mercedes': ['Mercedes', 'MERCEDES', 'mercedes', 'Mercede', 'ercedes',
                     'ERCEDES', 'MERCEDE', 'ercede', 'mercede'],
        'Hyundai': ['Hyundai', 'HYUNDAI', 'hyundai', 'Hyunda', 'yundai', 'yunda',
                    'HYUNDA', 'hyunda', 'yundai', 'yunda']
    }
    reverse_brand = {v.lower(): k for k, vals in brand_map.items() for v in vals}
    df['Brand'] = df['Brand'].astype(str).str.strip().str.lower().map(reverse_brand)
    df['Brand'] = df['Brand'].replace({None: np.nan})


    # column model: map all incorrect spellings to the right model
    # NOTE: this mapping encodes project-specific model names.
    model_map = {
        # VW
        'golf': ['golf', 'gol', 'golf s', 'golf sv'],
        'passat': ['passat', 'passa'],
        'polo': ['polo', 'pol'],
        'tiguan': ['tiguan', 'tigua', 'tiguan allspace', 'tiguan allspac'],
        'touran': ['touran', 'toura'],
        'up': ['up', 'u'],
        'sharan': ['sharan', 'shara'],
        'scirocco': ['scirocco', 'sciroc'],
        'amarok': ['amarok', 'amaro'],
        'arteon': ['arteon', 'arteo'],
        'beetle': ['beetle', 'beetl'],

        # Toyota
        'yaris': ['yaris', 'yari'],
        'corolla': ['corolla', 'corol', 'coroll'],
        'aygo': ['aygo', 'ayg'],
        'rav4': ['rav4', 'rav', 'rav-4'],
        'auris': ['auris', 'auri'],
        'avensis': ['avensis', 'avens'],
        'c-hr': ['c-hr', 'chr', 'c-h'],
        'verso': ['verso', 'verso-s'],
        'hilux': ['hilux', 'hilu'],
        'land cruiser': ['land cruiser', 'land cruise'],

        # Audi
        'a1': ['a1', 'a 1'],
        'a3': ['a3', 'a 3'],
        'a4': ['a4', 'a 4'],
        'a5': ['a5', 'a 5'],
        'a6': ['a6', 'a 6'],
        'a7': ['a7', 'a 7'],
        'a8': ['a8', 'a 8'],
        'q2': ['q2'],
        'q3': ['q3', 'q 3'],
        'q5': ['q5', 'q 5'],
        'q7': ['q7', 'q 7'],
        'q8': ['q8'],
        'tt': ['tt'],
        'r8': ['r8', 'r 8'],

        # Ford
        'fiesta': ['fiesta', 'fiest'],
        'focus': ['focus', 'focu'],
        'mondeo': ['mondeo', 'monde'],
        'kuga': ['kuga', 'kug'],
        'ecosport': ['ecosport', 'eco sport', 'ecospor'],
        'puma': ['puma', 'pum'],
        'edge': ['edge', 'edg'],
        's-max': ['s-max', 's-ma', 'smax'],
        'c-max': ['c-max', 'c-ma', 'cmax'],
        'b-max': ['b-max', 'b-ma', 'bmax'],
        'ka+': ['ka+', 'ka', 'streetka'],

        # BMW
        '1 series': ['1 series', '1 serie', '1 ser', '1series'],
        '2 series': ['2 series', '2 serie', '2series'],
        '3 series': ['3 series', '3 serie', '3series'],
        '4 series': ['4 series', '4 serie', '4series'],
        '5 series': ['5 series', '5 serie', '5series'],
        '6 series': ['6 series', '6 serie', '6series'],
        '7 series': ['7 series', '7 serie', '7series'],
        '8 series': ['8 series', '8 serie', '8series'],
        'x1': ['x1'], 'x2': ['x2'], 'x3': ['x3'], 'x4': ['x4'], 'x5': ['x5'], 'x6': ['x6'], 'x7': ['x7'],
        'z3': ['z3'], 'z4': ['z4'], 'm3': ['m3'], 'm4': ['m4'], 'm5': ['m5'], 'm6': ['m6'],

        # Skoda
        'fabia': ['fabia', 'fabi'], 'octavia': ['octavia', 'octavi', 'octa'],
        'superb': ['superb', 'super'], 'scala': ['scala', 'scal'],
        'karoq': ['karoq', 'karo'], 'kodiaq': ['kodiaq', 'kodia', 'kodi'],
        'kamiq': ['kamiq', 'kami'], 'yeti': ['yeti', 'yeti outdoor', 'yeti outdoo'],

        # Opel
        'astra': ['astra', 'astr'], 'corsa': ['corsa', 'cors'], 'insignia': ['insignia', 'insigni'],
        'mokka': ['mokka', 'mokk', 'mokka x', 'mokkax'], 'zafira': ['zafira', 'zafir'], 'meriva': ['meriva', 'meriv'],
        'adam': ['adam', 'ad'], 'vectra': ['vectra', 'vectr'], 'antara': ['antara', 'anta'],
        'combo life': ['combo life', 'combo lif'], 'grandland x': ['grandland x', 'grandland'], 'crossland x': ['crossland x', 'crossland'],

        # Mercedes
        'a class': ['a class', 'a clas', 'a-class'], 'b class': ['b class', 'b clas', 'b-class'],
        'c class': ['c class', 'c clas', 'c-class'], 'e class': ['e class', 'e clas', 'e-class'],
        's class': ['s class', 's clas', 's-class'], 'glc class': ['glc class', 'glc clas'],
        'gle class': ['gle class', 'gle clas'], 'gla class': ['gla class', 'gla clas'],
        'cls class': ['cls class', 'cls clas'], 'glb class': ['glb class'], 'gls class': ['gls class'],
        'm class': ['m class'], 'sl class': ['sl class'], 'cl class': ['cl class'], 'v class': ['v class'], 'x-class': ['x-class'], 'g class': ['g class'],

        # Hyundai
        'i10': ['i10', 'i 10'], 'i20': ['i20', 'i 20'], 'i30': ['i30', 'i 30'], 'i40': ['i40', 'i 40'],
        'ioniq': ['ioniq', 'ioni'], 'ix20': ['ix20', 'ix 20'], 'ix35': ['ix35', 'ix 35'],
        'kona': ['kona', 'kon'], 'tucson': ['tucson', 'tucso'], 'santa fe': ['santa fe', 'santa f']
    }
    reverse_model = {v.lower(): k for k, vals in model_map.items() for v in vals}
    df['model'] = df['model'].astype(str).str.strip().str.lower().map(reverse_model)
    df['model'] = df['model'].replace({None: np.nan})

    # column transmission: map all incorrect spellings to the right transmission type
    trans_map = {
        'Manual': ['manual', 'manua', 'anual', 'emi-auto', 'MANUAL'],
        'Semi-Auto': ['semi-auto', 'semi-aut', 'semi-aut', 'semi-aut', 'emi-auto'],
        'Automatic': ['automatic', 'automati', 'AUTOMATIC', 'utomatic', 'Automati'],
        'Unknown': ['unknown', 'unknow', 'nknown'],
        'Other': ['Other']
    }
    reverse_trans = {v.lower(): k for k, vals in trans_map.items() for v in vals}
    df['transmission'] = df['transmission'].astype(str).str.strip().str.lower().map(reverse_trans)
    df['transmission'] = df['transmission'].replace({None: np.nan})

    # column fuelType: map all incorrect spellings to the right fuelType
    fuel_map = {
        'Petrol': ['petrol', 'petro', 'etrol', 'etro'],
        'Diesel': ['diesel', 'dies', 'iesel', 'diese', 'iese', 'diesele'],
        'Hybrid': ['hybrid', 'ybri', 'hybri', 'ybrid', 'hybridd'],
        'Electric': ['electric'],
        'Other': ['other', 'ther', 'othe']
    }
    reverse_fuel = {v.lower(): k for k, vals in fuel_map.items() for v in vals}
    df['fuelType'] = df['fuelType'].astype(str).str.strip().str.lower().map(reverse_fuel)
    df['fuelType'] = df['fuelType'].replace({None: np.nan})

    # build model -> brand mapping: there are rows where `model` is filled but `Brand` is not.
    # We can back-fill brand from model via this mapping.
    model_to_brand = {}
    for brand, models in {
        'VW': ['golf', 'passat', 'polo', 'tiguan', 'touran', 'up', 'sharan', 'scirocco', 'amarok', 'arteon', 'beetle'],
        'Toyota': ['yaris', 'corolla', 'aygo', 'rav4', 'auris', 'avensis', 'c-hr', 'verso', 'hilux', 'land cruiser'],
        'Audi': ['a1', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'q2', 'q3', 'q5', 'q7', 'q8', 'tt', 'r8'],
        'Ford': ['fiesta', 'focus', 'mondeo', 'kuga', 'ecosport', 'puma', 'edge', 's-max', 'c-max', 'b-max', 'ka+'],
        'BMW': ['1 series', '2 series', '3 series', '4 series', '5 series', '6 series', '7 series', '8 series',
                'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'z3', 'z4', 'm3', 'm4', 'm5', 'm6'],
        'Skoda': ['fabia', 'octavia', 'superb', 'scala', 'karoq', 'kodiaq', 'kamiq', 'yeti'],
        'Opel': ['astra', 'corsa', 'insignia', 'mokka', 'zafira', 'meriva', 'adam', 'vectra', 'antara',
                 'combo life', 'grandland x', 'crossland x'],
        'Mercedes': ['a class', 'b class', 'c class', 'e class', 's class', 'glc class', 'gle class', 'gla class',
                     'cls class', 'glb class', 'gls class', 'm class', 'sl class', 'cl class', 'v class', 'x-class', 'g class'],
        'Hyundai': ['i10', 'i20', 'i30', 'i40', 'ioniq', 'ix20', 'ix35', 'kona', 'tucson', 'santa fe']
    }.items():
        for m in models:
            model_to_brand[m] = brand

    # fill missing Brand from model where possible
    df.loc[df['Brand'].isna() & df['model'].notna(), 'Brand'] = (
        df.loc[df['Brand'].isna() & df['model'].notna(), 'model'].map(model_to_brand)
    )

    return df

##########################################################################################################################################################

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing numeric values using hierarchical medians.

    The goal is to use local structure first (e.g. a specific brand + model median), 
    and fall back to more global statistics only when necessary.

    Hierarchy
    ---------
    For each numeric column (except grouping columns themselves):

    1. (group_cols[0], group_cols[1]) median
       e.g. (Brand_te, model_te)
    2. group_cols[0] median
       e.g. Brand_te
    3. Global median over all rows

    Parameters
    ----------
    group_cols : list of str, optional
        Column names used for the grouping hierarchy, in order of specificity.
        Default is ["Brand_te", "model_te"], which matches how this transformer
        is used in the main project: it runs after target-encoded features
        were added to the dataframe.

    Usage pattern
    -------------
    - This transformer is meant to sit inside a sklearn Pipeline.
    - It expects that the columns in `group_cols` are present in X.
    - In this project, it is plugged into the numeric/log pipelines before scaling.

    Attributes
    ----------
    medians_ : pd.DataFrame
        Pair-level medians indexed by group_cols (e.g. Brand_te, model_te).
    first_level_medians_ : pd.DataFrame
        Medians grouped by group_cols[0] (e.g. Brand_te).
    global_median_ : pd.Series
        Median of all numeric columns in X.
    """

    def __init__(self, group_cols=["Brand", "model"]):
        self.group_cols = group_cols

    def fit(self, X, y=None):
        """
        Learn group-level and global medians from the input data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Data containing the grouping columns and the numeric columns to impute.
        y : Ignored
            Included for sklearn compatibility.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = X.columns

        # Step 1 — pair-level medians (e.g. Brand_te + model_te)
        if all(c in X.columns for c in self.group_cols):
            self.medians_ = X.groupby(self.group_cols).median(numeric_only=True)
        else:
            self.medians_ = pd.DataFrame()

        # Step 2 — first group col only (e.g. Brand_te)
        if self.group_cols and self.group_cols[0] in X.columns:
            self.first_level_medians_ = X.groupby(self.group_cols[0]).median(numeric_only=True)
        else:
            self.first_level_medians_ = pd.DataFrame()

        # Step 3 — global medians
        self.global_median_ = X.median(numeric_only=True)
        return self

    def transform(self, X):
        """
        Apply the hierarchical median imputation to new data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Data to transform. Must contain the same columns as seen in `fit`,
            including the `group_cols`.

        Returns
        -------
        np.ndarray
            Imputed values as a NumPy array (sklearn convention).
        """
        X = pd.DataFrame(X).copy()

        # columns we are allowed to impute (ignore grouping columns themselves)
        cols_to_impute = [
            c for c in X.columns
            if c not in self.group_cols and X[c].isna().any()
        ]
        if not cols_to_impute:
            return X.values

        # IMPORTANT: cast imputed columns to float so medians like 17416.5 are valid
        X[cols_to_impute] = X[cols_to_impute].astype("float64")

        # -------------------------
        # 1) Pair-level medians: (group_cols[0], group_cols[1])
        # -------------------------
        if (
            self.group_cols
            and all(c in X.columns for c in self.group_cols)
            and not getattr(self, "medians_", pd.DataFrame()).empty
        ):
            key_df = X[self.group_cols].copy()
            med_df = self.medians_.reset_index()  # group_cols + numeric medians

            joined = key_df.merge(
                med_df,
                on=self.group_cols,
                how="left",
                suffixes=("", "_gpair")
            )

            for col in cols_to_impute:
                if col not in self.medians_.columns:
                    continue
                mask = X[col].isna() & joined[col].notna()
                X.loc[mask, col] = joined.loc[mask, col]

        # -------------------------
        # 2) First-level medians: group_cols[0]
        # -------------------------
        if (
            self.group_cols
            and self.group_cols[0] in X.columns
            and not getattr(self, "first_level_medians_", pd.DataFrame()).empty
        ):
            gcol = self.group_cols[0]
            med1 = self.first_level_medians_.reset_index()  # gcol + numeric medians

            joined1 = X[[gcol]].merge(
                med1[[gcol] + [c for c in cols_to_impute if c in med1.columns]],
                on=gcol,
                how="left",
                suffixes=("", "_g1")
            )

            for col in cols_to_impute:
                if col not in med1.columns:
                    continue
                mask = X[col].isna() & joined1[col].notna()
                X.loc[mask, col] = joined1.loc[mask, col]

        # -------------------------
        # 3) Global median fallback
        # -------------------------
        for col in cols_to_impute:
            if col in self.global_median_:
                X[col] = X[col].fillna(self.global_median_[col])

        return X.values

##########################################################################################################################################################

class GroupModeImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing categorical values hierarchically:
    1) mode per (group_cols[0], group_cols[1])
    2) mode per group_cols[0]
    3) global mode
    """

    def __init__(self, group_cols=["Brand", "model"], fallback="__MISSING__"):
        self.group_cols = group_cols
        self.fallback = fallback

    def _mode(self, series):
        # deterministic mode: first entry if same score
        m = series.mode(dropna=True)
        return m.iloc[0] if not m.empty else self.fallback

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = X.columns

        # Level 1: Group mode per (group_cols[0], group_cols[1])
        if all(c in X.columns for c in self.group_cols):
            self.modes_ = (
                X.groupby(self.group_cols)
                 .agg(lambda s: self._mode(s))
            )
        else:
            self.modes_ = pd.DataFrame()

        # Level 2: First-level group mode per group_cols[0]
        if self.group_cols and self.group_cols[0] in X.columns:
            self.first_level_modes_ = (
                X.groupby(self.group_cols[0])
                 .agg(lambda s: self._mode(s))
            )
        else:
            self.first_level_modes_ = pd.DataFrame()

        # Level 3: Global mode per column
        self.global_mode_ = X.apply(self._mode)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Which columns need imputing?
        cols_to_impute = [
            c for c in X.columns
            if c not in self.group_cols and X[c].isna().any()
        ]

        if not cols_to_impute:
            return X.values

        # 1) Group-Level Imputation
        if (
            self.group_cols
            and all(c in X.columns for c in self.group_cols)
            and not self.modes_.empty
        ):
            key_df = X[self.group_cols]
            mode_df = self.modes_.reset_index()
            joined = key_df.merge(mode_df, on=self.group_cols, how="left")

            for col in cols_to_impute:
                if col not in self.modes_.columns:
                    continue
                mask = X[col].isna() & joined[col].notna()
                X.loc[mask, col] = joined.loc[mask, col]

        # 2) First-level group imputation
        if (
            self.group_cols
            and self.group_cols[0] in X.columns
            and not self.first_level_modes_.empty
        ):
            g = self.group_cols[0]
            mode1 = self.first_level_modes_.reset_index()

            joined1 = X[[g]].merge(
                mode1[[g] + [c for c in cols_to_impute if c in mode1.columns]],
                on=g,
                how="left",
            )

            for col in cols_to_impute:
                if col not in mode1.columns:
                    continue
                mask = X[col].isna() & joined1[col].notna()
                X.loc[mask, col] = joined1.loc[mask, col]

        # 3) Global mode imputation
        for col in cols_to_impute:
            X[col] = X[col].fillna(self.global_mode_.get(col, self.fallback))

        return X.values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)


##########################################################################################################################################################

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

##########################################################################################################################################################

# TODO remove for final submission because its currently not used anywhere
def cv_target_encode(train_df, test_df, col, ycol='price', m=50, n_splits=5, random_state=42):
    """
    Leakage-safe KFold target encoding for a single categorical feature.

    Idea
    ----
    Replace a category (e.g. model) by a smoothed estimate of the target mean
    for that category, but computed in a **cross-validation** fashion so that
    each row only sees statistics from *other* rows (no target leakage).

    Steps
    -----
    1. Split train_df into K folds.
    2. For each fold:
       - compute group stats on the remaining K-1 folds,
       - map those encodings onto the held-out fold.
    3. For test_df:
       - compute group stats on the full train_df,
       - map those encodings onto test_df.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe containing feature column `col` and target `ycol`.
    test_df : pd.DataFrame
        Test dataframe containing feature column `col`.
    col : str
        Categorical column to encode.
    ycol : str, default 'price'
        Target column.
    m : int, default 50
        Smoothing parameter for M-estimate.
    n_splits : int, default 5
        Number of folds for KFold.
    random_state : int, default 42
        Reproducibility of the CV splits.

    Returns
    -------
    (pd.Series, pd.Series)
        Encoded training series, encoded test series (both float32).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_mean = train_df[ycol].mean()
    tr_encoded = pd.Series(index=train_df.index, dtype=float)

    # Encode training data using CV folds
    for tr_idx, val_idx in kf.split(train_df):
        tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        stats = tr.groupby(col)[ycol].agg(['sum', 'count'])
        # Keep formula identical to the notebook version
        stats['enc'] = m_estimate_mean(stats['sum'], global_mean * stats['count'], stats['count'], m=m)
        tr_encoded.iloc[val_idx] = val[col].map(stats['enc']).fillna(global_mean)

    # Encode test data using full training statistics
    full_stats = train_df.groupby(col)[ycol].agg(['sum', 'count'])
    full_stats['enc'] = m_estimate_mean(full_stats['sum'], global_mean * full_stats['count'], full_stats['count'], m=m)
    te_map = full_stats['enc'].to_dict()

    te_train = tr_encoded.fillna(global_mean).astype('float32')
    te_test = test_df[col].map(te_map).fillna(global_mean).astype('float32')

    return te_train, te_test

##########################################################################################################################################################

# TODO remove for final submission because its currently not used anywhere
def print_metrics(y_true, y_pred):
    """
    Utility wrapper for the 3 main regression metrics used throughout the project.

    The intention is to:
    - keep metric reporting consistent across experiments, and
    - avoid duplicating metric code in every notebook cell.

    Metrics
    -------
    - MAE  : Mean Absolute Error, same unit as the target (here: GBP)
    - RMSE : Mean Squared Error (note: currently **not** square-rooted on purpose
             to match the original notebook code and keep behaviour identical)
    - R²   : Coefficient of determination

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted target values from a model.

    Returns
    -------
    None
        Prints the metrics to stdout.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")