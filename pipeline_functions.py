"""
pipeline_functions.py — Cars4You

Reading guide:
-----------------------------
This file contains small building blocks ("transformers") that are chained into a
single sklearn Pipeline. Each transformer does one job:

1) CarDataCleaner       : fixes obvious issues (typos, impossible numeric ranges) without dropping rows
2) OutlierHandler       : reduces the impact of extreme numeric values (winsorization / clipping)
3) GroupImputer         : fills missing values using statistics from similar cars first
4) CarFeatureEngineer   : creates additional signals (age, ratios, interactions, relative positioning)
5) Feature selection    : keeps only helpful signals and drops redundant noise

Important rule: transformers must NOT drop rows inside transform(),
otherwise X and y get misaligned during cross-validation.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.feature_selection import SelectorMixin, mutual_info_regression
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, check_X_y

from scipy.stats import spearmanr

from difflib import get_close_matches

# !pip install ydata-profiling
from ydata_profiling import ProfileReport



################################################################################
# Debug Transformer 
################################################################################

class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    DebugTransformer (optional)
    ---------------------------
    Prints shape/type and optionally shows a small preview of the data flowing
    through the pipeline.

    Why this exists:
    - Pipelines can be hard to inspect because intermediate outputs change types
      (DataFrame → array → sparse matrix).
    - This step makes the pipeline "transparent" during development.

    Safe defaults:
    - show_data=False (no messy printing unless wanted)
    - y_data_profiling=False
    """

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
            if display is None:
                print("display() not available in this environment.")
            elif isinstance(X, pd.DataFrame):
                print(f"\nFirst {self.n_rows} rows:")
                display(X.head(self.n_rows))
                display(X.describe(include="all").T)
                if self.y_data_profiling:
                    print("\nGenerating data profiling report...")
                    profile = ProfileReport(
                        X,
                        title="Car Data Profiling Report",
                        correlations={
                            "pearson": {"calculate": True},
                            "spearman": {"calculate": False},
                            "kendall": {"calculate": False},
                            "phi_k": {"calculate": False},
                            "cramers": {"calculate": False},
                        },
                    )
                    profile.to_notebook_iframe()
            else:  # Edge-case for numpy array after column transformer
                print(f"\nFirst {self.n_rows} rows:")
                display(X[: self.n_rows])

        return X



################################################################################
# Data Cleaning (CarDataCleaner)
################################################################################

class CarDataCleaner(BaseEstimator, TransformerMixin):
    """
    CarDataCleaner

    What it does:
    ---------------------------
    1) Schema normalization:
       - rename Brand -> brand 
       - rename paintQuality% -> paintQuality (then dropped; not available at prediction time)

    2) Numeric sanity checks:
       - coercion to numeric types
       - invalid ranges are set to NaN (NOT dropped) so imputers can handle them

    3) Categorical canonicalization:
       - normalize case/whitespace
       - map known typos/variants into one canonical label using explicit dictionaries

    4) Conservative fuzzy fallback:
       - Only tries to fill values that are still missing AFTER deterministic mapping
       - Vocabulary is learned in fit() from the training fold -> leakage-safe in CV
       - Guardrails prevent dangerous guessing (e.g., 1-letter tokens like "a")

    Notes
    -----
    - paintQuality is dropped because it is not available for predictions (filled by mechanic).
    - Electric vehicles are recoded (not dropped) due to very small / inconsistent presence.
    """

    def __init__(
        self,
        year_min=1886,
        year_max=2020,
        mpg_min=5,
        mpg_max=150,
        engine_min=0.6,
        engine_max=9.0,
        paint_min=5,
        paint_max=100,
        handle_electric="other",  # {"keep","other","nan"}
        set_carid_index=False,    # keep False for pipeline stability

        # fuzzy fallback (must be leakage-safe -> vocab learned in fit())
        use_fuzzy=True,
        fuzzy_cutoff_brand=0.94,
        fuzzy_cutoff_model=0.94,
        fuzzy_cutoff_trans=0.94,
        fuzzy_cutoff_fuel=0.94,
        fuzzy_min_token_len=2,              # prevents "a" -> "a3" guessing
        fuzzy_require_brand_for_model=True  # model fuzzy restricted to brand vocab if possible
    ):
        # numeric checks
        self.year_min = year_min
        self.year_max = year_max
        self.mpg_min = mpg_min
        self.mpg_max = mpg_max
        self.engine_min = engine_min
        self.engine_max = engine_max
        self.paint_min = paint_min
        self.paint_max = paint_max

        # other behavior
        self.handle_electric = handle_electric
        self.set_carid_index = set_carid_index

        # fuzzy behavior
        self.use_fuzzy = use_fuzzy
        self.fuzzy_cutoff_brand = fuzzy_cutoff_brand
        self.fuzzy_cutoff_model = fuzzy_cutoff_model
        self.fuzzy_cutoff_trans = fuzzy_cutoff_trans
        self.fuzzy_cutoff_fuel = fuzzy_cutoff_fuel
        self.fuzzy_min_token_len = fuzzy_min_token_len
        self.fuzzy_require_brand_for_model = fuzzy_require_brand_for_model


    # Helpers (string normalization)
    @staticmethod
    def _norm_str_series(s: pd.Series) -> pd.Series:
        """Lowercase + strip without turning NaN into the string 'nan'."""
        return s.astype(str).str.strip().str.lower().replace('nan', np.nan)

    @staticmethod
    def _canon_map(series: pd.Series, reverse_map: dict) -> pd.Series:
        """
        Map known variants to canonical labels using a reverse dictionary:
          reverse_map[variant_lower] = canonical_label
        """
        s = series.astype(str).str.strip().str.lower().replace('nan', np.nan)
        out = s.map(reverse_map)
        return out


    # Helpers (fuzzy matching)
    @staticmethod
    def _fuzzy_one(token: str, choices: list[str], cutoff: float) -> str | None:
        """
        Return the closest match from 'choices' if similarity >= cutoff.
        Uses Python's stdlib difflib (fast, no extra dependency).
        """
        tok = str(token).strip().lower()
        if tok == "" or tok == "<na>":
            return None
        match = get_close_matches(tok, choices, n=1, cutoff=cutoff)
        return match[0] if match else None

    def _fuzzy_fill_missing(self, s: pd.Series, choices: list[str], cutoff: float) -> pd.Series:
        """
        Conservative fuzzy fill:
        - only fills NaNs
        - only attempts if token length >= fuzzy_min_token_len
        """
        out = s.copy()

        def _try(x):
            if pd.isna(x):
                return None
            tok = str(x).strip().lower()
            if len(tok) < self.fuzzy_min_token_len:
                return None
            m = self._fuzzy_one(tok, choices, cutoff=cutoff)
            return m

        miss = out.isna()
        if miss.any() and choices:
            out.loc[miss] = out.loc[miss].apply(_try)
        return out


    # fit (learn fuzzy vocab on train fold only)
    def fit(self, X, y=None):
        """
        Learn training-fold vocab for optional fuzzy fallback.
        This keeps fuzzy matching leakage-safe during CV.
        """
        df = pd.DataFrame(X).copy()

        # normalize schema so fit vocab matches transform behavior
        if "Brand" in df.columns and "brand" not in df.columns:
            df = df.rename(columns={"Brand": "brand"})

        # store global vocab
        if "brand" in df.columns:
            self.brand_vocab_ = sorted(self._norm_str_series(df["brand"]).dropna().unique().tolist())
        else:
            self.brand_vocab_ = []

        if "model" in df.columns:
            self.model_vocab_ = sorted(self._norm_str_series(df["model"]).dropna().unique().tolist())
        else:
            self.model_vocab_ = []

        if "transmission" in df.columns:
            self.trans_vocab_ = sorted(self._norm_str_series(df["transmission"]).dropna().unique().tolist())
        else:
            self.trans_vocab_ = []

        if "fuelType" in df.columns:
            self.fuel_vocab_ = sorted(self._norm_str_series(df["fuelType"]).dropna().unique().tolist())
        else:
            self.fuel_vocab_ = []

        # brand -> models vocab (safer fuzzy matching for model)
        self.brand_to_models_ = {}
        if "brand" in df.columns and "model" in df.columns:
            b = self._norm_str_series(df["brand"])
            m = self._norm_str_series(df["model"])
            tmp = pd.DataFrame({"brand": b, "model": m}).dropna()
            for brand_val, g in tmp.groupby("brand"):
                self.brand_to_models_[brand_val] = sorted(g["model"].unique().tolist())

        return self


    # transform
    def transform(self, X):
        df = pd.DataFrame(X).copy()

        # column carID (set as index, has no duplicates) (in pipelines, keeping it as a column is often easier)
        if self.set_carid_index and "carID" in df.columns:
            df = df.set_index("carID")

        # rename once here so the rest of the code can consistently use `brand`
        if "Brand" in df.columns and "brand" not in df.columns:
            df = df.rename(columns={"Brand": "brand"})

        # rename once here so the rest of the code can consistently use `paintQuality`
        if "paintQuality%" in df.columns and "paintQuality" not in df.columns:
            df = df.rename(columns={"paintQuality%": "paintQuality"})

        # NUMERICAL COLUMNS
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df.loc[~df["year"].between(self.year_min, self.year_max), "year"] = np.nan
            df["year"] = np.floor(df["year"]).astype("float64")

        if "mileage" in df.columns:
            df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
            df.loc[df["mileage"] < 0, "mileage"] = np.nan
            df["mileage"] = np.floor(df["mileage"]).astype("float64")

        if "tax" in df.columns:
            df["tax"] = pd.to_numeric(df["tax"], errors="coerce")
            df.loc[df["tax"] < 0, "tax"] = np.nan
            df["tax"] = np.floor(df["tax"]).astype("float64")

        if "mpg" in df.columns:
            df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")
            df.loc[~df["mpg"].between(self.mpg_min, self.mpg_max), "mpg"] = np.nan
            df["mpg"] = np.floor(df["mpg"]).astype("float64")

        if "engineSize" in df.columns:
            df["engineSize"] = pd.to_numeric(df["engineSize"], errors="coerce")
            df.loc[~df["engineSize"].between(self.engine_min, self.engine_max), "engineSize"] = np.nan
            df["engineSize"] = df["engineSize"].round(1)

        if "paintQuality" in df.columns:
            df["paintQuality"] = pd.to_numeric(df["paintQuality"], errors="coerce")
            df.loc[~df["paintQuality"].between(self.paint_min, self.paint_max), "paintQuality"] = np.nan
            df["paintQuality"] = np.floor(df["paintQuality"]).astype("float64")

        if "previousOwners" in df.columns:
            df["previousOwners"] = pd.to_numeric(df["previousOwners"], errors="coerce")
            df.loc[df["previousOwners"] < 0, "previousOwners"] = np.nan
            df["previousOwners"] = np.floor(df["previousOwners"]).astype("float64")

        # column hasDamage (we cannot safely assume NaN means damaged or not damaged)
        if "hasDamage" in df.columns:
            df["hasDamage"] = pd.to_numeric(df["hasDamage"], errors="coerce").astype("float64")

        # Drop paintQuality because we cannot use it for predictions (filled by mechanic)
        if "paintQuality" in df.columns:
            df = df.drop(columns=["paintQuality"])


        # CATEGORICAL COLUMNS
        # The idea is always:
        # - normalise case / whitespace
        # - map known typos/variants into a canonical label using a reverse dict

        # column brand: map all incorrect spellings to the right brand
        brand_map = {
            "VW": ["VW", "V", "W", "vw", "v", "w"],
            "Toyota": ["Toyota", "TOYOTA", "Toyot", "toyota", "oyota", "TOYOT", "OYOTA"],
            "Audi": ["Audi", "AUDI", "A", "udi", "Aud", "audi", "AUD", "UDI"],
            "Ford": ["Ford", "FORD", "For", "ord", "for", "ORD", "or", "FOR"],
            "BMW": ["BMW", "bmw", "MW", "BM", "mw"],
            "Skoda": ["Skoda", "SKODA", "Skod", "koda", "SKOD", "kod", "skoda", "skod", "KODA"],
            "Opel": ["Opel", "OPEL", "Ope", "opel", "OPE", "pel", "pe", "PEL", "ope"],
            "Mercedes": ["Mercedes", "MERCEDES", "mercedes", "Mercede", "ercedes", "ERCEDES", "MERCEDE", "ercede", "mercede"],
            "Hyundai": ["Hyundai", "HYUNDAI", "hyundai", "Hyunda", "yundai", "yunda", "HYUNDA", "hyunda", "yundai", "yunda"],
        }
        reverse_brand = {v.lower(): k for k, vals in brand_map.items() for v in vals}

        if "brand" in df.columns:
            df["brand"] = self._canon_map(df["brand"], reverse_brand)

        # Store raw model token BEFORE deterministic mapping (used for special cases + optional fuzzy fallback)
        if "model" in df.columns:
            df["_raw_model"] = self._norm_str_series(df["model"])
        else:
            df["_raw_model"] = pd.Series([np.nan] * len(df), index=df.index)

        # If brand is Audi and model token is exactly "a", we do not want do guessing (a1/a3/a4/...) -> map to dedicated category a_unknown.
        if "brand" in df.columns and "model" in df.columns:
            audi_a_mask = (df["brand"] == "Audi") & (df["_raw_model"] == "a")
            df.loc[audi_a_mask, "model"] = "a_unknown"

        # column model: map all incorrect spellings to the right model
        model_map = {
            # VW
            "golf": ["golf", "gol", "golf s", "golf sv"],
            "passat": ["passat", "passa"],
            "polo": ["polo", "pol"],
            "tiguan": ["tiguan", "tigua", "tiguan allspace", "tiguan allspac"],
            "touran": ["touran", "toura"],
            "up": ["up", "u"],
            "sharan": ["sharan", "shara"],
            "scirocco": ["scirocco", "sciroc"],
            "amarok": ["amarok", "amaro"],
            "arteon": ["arteon", "arteo"],
            "beetle": ["beetle", "beetl"],

            # Toyota
            "yaris": ["yaris", "yari"],
            "corolla": ["corolla", "corol", "coroll"],
            "aygo": ["aygo", "ayg"],
            "rav4": ["rav4", "rav", "rav-4"],
            "auris": ["auris", "auri"],
            "avensis": ["avensis", "avens"],
            "c-hr": ["c-hr", "chr", "c-h"],
            "verso": ["verso", "verso-s"],
            "hilux": ["hilux", "hilu"],
            "land cruiser": ["land cruiser", "land cruise"],

            # Audi
            "a_unknown": ["a_unknown"],
            "a1": ["a1", "a 1"],
            "a3": ["a3", "a 3"],
            "a4": ["a4", "a 4"],
            "a5": ["a5", "a 5"],
            "a6": ["a6", "a 6"],
            "a7": ["a7", "a 7"],
            "a8": ["a8", "a 8"],
            "q2": ["q2"],
            "q3": ["q3", "q 3"],
            "q5": ["q5", "q 5"],
            "q7": ["q7", "q 7"],
            "q8": ["q8"],
            "tt": ["tt"],
            "r8": ["r8", "r 8"],

            # Ford
            "fiesta": ["fiesta", "fiest"],
            "focus": ["focus", "focu"],
            "mondeo": ["mondeo", "monde"],
            "kuga": ["kuga", "kug"],
            "ecosport": ["ecosport", "eco sport", "ecospor"],
            "puma": ["puma"], 
            "edge": ["edge", "edg"],
            "s-max": ["s-max", "s-ma", "smax"],
            "c-max": ["c-max", "c-ma", "cmax"],
            "b-max": ["b-max", "b-ma", "bmax"],
            "ka+": ["ka+", "ka", "streetka"],

            # BMW
            "1 series": ["1 series", "1 serie", "1 ser", "1series"],
            "2 series": ["2 series", "2 serie", "2series"],
            "3 series": ["3 series", "3 serie", "3series"],
            "4 series": ["4 series", "4 serie", "4series"],
            "5 series": ["5 series", "5 serie", "5series"],
            "6 series": ["6 series", "6 serie", "6series"],
            "7 series": ["7 series", "7 serie", "7series"],
            "8 series": ["8 series", "8 serie", "8series"],
            "x1": ["x1"],
            "x2": ["x2"],
            "x3": ["x3"],
            "x4": ["x4"],
            "x5": ["x5"],
            "x6": ["x6"],
            "x7": ["x7"],
            "z3": ["z3"],
            "z4": ["z4"],
            "m3": ["m3"],
            "m4": ["m4"],
            "m5": ["m5"],
            "m6": ["m6"],

            # Skoda
            "fabia": ["fabia", "fabi"],
            "octavia": ["octavia", "octavi", "octa"],
            "superb": ["superb", "super"],
            "scala": ["scala", "scal"],
            "karoq": ["karoq", "karo"],
            "kodiaq": ["kodiaq", "kodia", "kodi"],
            "kamiq": ["kamiq", "kami"],
            "yeti": ["yeti", "yeti outdoor", "yeti outdoo"],

            # Opel
            "astra": ["astra", "astr"],
            "corsa": ["corsa", "cors"],
            "insignia": ["insignia", "insigni"],
            "mokka": ["mokka", "mokk", "mokka x", "mokkax"],
            "zafira": ["zafira", "zafir"],
            "meriva": ["meriva", "meriv"],
            "adam": ["adam", "ad"],
            "vectra": ["vectra", "vectr"],
            "antara": ["antara", "anta"],
            "combo life": ["combo life", "combo lif"],
            "grandland x": ["grandland x", "grandland"],
            "crossland x": ["crossland x", "crossland"],

            # Mercedes
            "a class": ["a class", "a clas", "a-class"],
            "b class": ["b class", "b clas", "b-class"],
            "c class": ["c class", "c clas", "c-class"],
            "e class": ["e class", "e clas", "e-class"],
            "s class": ["s class", "s clas", "s-class"],
            "glc class": ["glc class", "glc clas"],
            "gle class": ["gle class", "gle clas"],
            "gla class": ["gla class", "gla clas"],
            "cls class": ["cls class", "cls clas"],
            "glb class": ["glb class"],
            "gls class": ["gls class"],
            "m class": ["m class"],
            "sl class": ["sl class"],
            "cl class": ["cl class"],
            "v class": ["v class"],
            "x-class": ["x-class"],
            "g class": ["g class"],

            # Hyundai
            "i10": ["i10", "i 10"],
            "i20": ["i20", "i 20"],
            "i30": ["i30", "i 30"],
            "i40": ["i40", "i 40"],
            "ioniq": ["ioniq", "ioni"],
            "ix20": ["ix20", "ix 20"],
            "ix35": ["ix35", "ix 35"],
            "kona": ["kona", "kon"],
            "tucson": ["tucson", "tucso"],
            "santa fe": ["santa fe", "santa f"],
        }

        reverse_model = {v.lower(): k for k, vals in model_map.items() for v in vals}

        if "model" in df.columns:
            # Important: do NOT overwrite the Audi special case a_unknown
            # We only map those rows where model is still not already set to a_unknown
            model_is_unknown_bucket = df["model"].astype(str) == "a_unknown"
            mapped = self._canon_map(df["model"], reverse_model)
            df.loc[~model_is_unknown_bucket, "model"] = mapped.loc[~model_is_unknown_bucket]

        # column transmission
        trans_map = {
            "Manual": ["manual", "manua", "anual", "emi-auto", "MANUAL"],
            "Semi-Auto": ["semi-auto", "semi-aut", "emi-auto"],
            "Automatic": ["automatic", "automati", "AUTOMATIC", "utomatic", "Automati"],
            "Unknown": ["unknown", "unknow", "nknown"],
            "Other": ["Other", "other"],
        }
        reverse_trans = {v.lower(): k for k, vals in trans_map.items() for v in vals}
        if "transmission" in df.columns:
            df["transmission"] = self._canon_map(df["transmission"], reverse_trans)

        # column fuelType
        fuel_map = {
            "Petrol": ["petrol", "petro", "etrol", "etro"],
            "Diesel": ["diesel", "dies", "iesel", "diese", "iese", "diesele"],
            "Hybrid": ["hybrid", "ybri", "hybri", "ybrid", "hybridd"],
            "Electric": ["electric"],
            "Other": ["other", "ther", "othe"],
        }
        reverse_fuel = {v.lower(): k for k, vals in fuel_map.items() for v in vals}
        if "fuelType" in df.columns:
            df["fuelType"] = self._canon_map(df["fuelType"], reverse_fuel)

            # Remove Electric vehicles due to too few samples (do NOT drop rows inside pipeline)
            if self.handle_electric == "other":
                df.loc[df["fuelType"] == "Electric", "fuelType"] = "Other"
            elif self.handle_electric == "nan":
                df.loc[df["fuelType"] == "Electric", "fuelType"] = np.nan


        # FUZZY FALLBACK
        # Only try to fill values that are STILL missing after deterministic mapping.
        # This avoids fuzzy overriding correct deterministic mappings.
        if self.use_fuzzy:
            # brand fuzzy
            if "brand" in df.columns and getattr(self, "brand_vocab_", []):
                df["brand"] = self._fuzzy_fill_missing(df["brand"], self.brand_vocab_, self.fuzzy_cutoff_brand)

            # transmission fuzzy
            if "transmission" in df.columns and getattr(self, "trans_vocab_", []):
                df["transmission"] = self._fuzzy_fill_missing(df["transmission"], self.trans_vocab_, self.fuzzy_cutoff_trans)

            # fuel fuzzy
            if "fuelType" in df.columns and getattr(self, "fuel_vocab_", []):
                df["fuelType"] = self._fuzzy_fill_missing(df["fuelType"], self.fuel_vocab_, self.fuzzy_cutoff_fuel)

            # model fuzzy (restricted to brand-specific vocab when possible)
            if "model" in df.columns:
                # only attempt fuzzy on missing model values
                miss_model = df["model"].isna()
                if miss_model.any():
                    def _choices_for_row(idx):
                        if self.fuzzy_require_brand_for_model and "brand" in df.columns and pd.notna(df.loc[idx, "brand"]):
                            b = str(df.loc[idx, "brand"]).strip().lower()
                            return self.brand_to_models_.get(b, [])
                        return getattr(self, "model_vocab_", [])

                    filled = []
                    for idx in df.index[miss_model]:
                        raw_tok = df.loc[idx, "_raw_model"]
                        if pd.isna(raw_tok) or len(str(raw_tok)) < self.fuzzy_min_token_len:
                            filled.append(np.nan)
                            continue
                        choices = _choices_for_row(idx)
                        m = self._fuzzy_one(raw_tok, choices, cutoff=self.fuzzy_cutoff_model)
                        filled.append(m if m is not None else np.nan)

                    df.loc[miss_model, "model"] = pd.Series(filled, index=df.index[miss_model])

        # clean helper column
        df = df.drop(columns=["_raw_model"], errors="ignore")

        # build model -> brand mapping: there are rows where model is filled but brand is not.
        # We can back-fill brand from model via this mapping.
        model_to_brand = {}
        for brand, models in {
            "VW": ["golf", "passat", "polo", "tiguan", "touran", "up", "sharan", "scirocco", "amarok", "arteon", "beetle"],
            "Toyota": ["yaris", "corolla", "aygo", "rav4", "auris", "avensis", "c-hr", "verso", "hilux", "land cruiser"],
            "Audi": ["a_unknown","a1", "a3", "a4", "a5", "a6", "a7", "a8", "q2", "q3", "q5", "q7", "q8", "tt", "r8"],
            "Ford": ["fiesta", "focus", "mondeo", "kuga", "ecosport", "puma", "edge", "s-max", "c-max", "b-max", "ka+"],
            "BMW": ["1 series","2 series","3 series","4 series","5 series","6 series","7 series","8 series","x1","x2","x3","x4","x5","x6","x7","z3","z4","m3","m4","m5","m6"],
            "Skoda": ["fabia", "octavia", "superb", "scala", "karoq", "kodiaq", "kamiq", "yeti"],
            "Opel": ["astra", "corsa", "insignia", "mokka", "zafira", "meriva", "adam", "vectra", "antara", "combo life", "grandland x", "crossland x"],
            "Mercedes": ["a class","b class","c class","e class","s class","glc class","gle class","gla class","cls class","glb class","gls class","m class","sl class","cl class","v class","x-class","g class"],
            "Hyundai": ["i10","i20","i30","i40","ioniq","ix20","ix35","kona","tucson","santa fe"],
        }.items():
            for m in models:
                model_to_brand[m] = brand

        # fill missing brand from model where possible
        if "brand" in df.columns and "model" in df.columns:
            mask = df["brand"].isna() & df["model"].notna()
            df.loc[mask, "brand"] = df.loc[mask, "model"].map(model_to_brand)

        # Fallback to ensure sklearn compatibility (convert all string dtypes to object and replace pd.NA with np.nan)
        for col in df.columns:
            if str(df[col].dtype) == 'string':
                # First replace pd.NA explicitly, then convert to object
                df[col] = df[col].replace({pd.NA: np.nan}).astype('object')

        return df
    


################################################################################
# Outlier Handling (OutlierHandler)
################################################################################

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    OutlierHandler

    After deterministic cleaning (range checks, typos, impossible values), we still can have:
      1) wrong values for that specific model but missed in the data cleaning
      2) extreme but valid values that distort the distribution

    Approach:
    --------
    - Detect outliers using multiple robust univariate rules and only flag points that
      are supported by more than one rule (voting).
    - Then either:
        > set them to NaN (recommended BEFORE GroupImputer), or
        > clip them to bounds (winsorize) (recommended AFTER imputation / for linear models)

    Default: vote between
      - Tukey IQR fences (k=1.5)
      - Modified Z-score using median + MAD (threshold=3.5)
    """

    def __init__(
        self,
        cols=None,
        methods=("iqr", "mod_z"),
        min_votes=2,
        iqr_k=1.5,
        z_thresh=3.5,
        action="nan",           # "nan" or "clip"
        verbose=False,
    ):
        self.cols = cols
        self.methods = methods
        self.min_votes = min_votes
        self.iqr_k = iqr_k
        self.z_thresh = z_thresh
        self.action = action
        self.verbose = verbose

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        self.feature_names_in_ = df.columns.to_list()

        # Decide which columns to handle
        if self.cols is None:
            # Only numeric columns by default
            self.cols_ = df.select_dtypes(include="number").columns.to_list()
        else:
            self.cols_ = [c for c in self.cols if c in df.columns]

        self.stats_ = {}

        for col in self.cols_:
            s = pd.to_numeric(df[col], errors="coerce").astype(float)
            s = s.dropna()
            if s.empty:
                self.stats_[col] = {}
                continue

            col_stats = {}

            # --- IQR fences ---
            if "iqr" in self.methods:
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - self.iqr_k * iqr
                upper = q3 + self.iqr_k * iqr
                col_stats["iqr"] = {"lower": lower, "upper": upper, "q1": q1, "q3": q3, "iqr": iqr}

            # --- Modified Z-score (median + MAD) ---
            if "mod_z" in self.methods:
                med = float(s.median())
                mad = float(np.median(np.abs(s - med)))
                # Avoid division-by-zero (constant-ish feature)
                if mad <= 0:
                    col_stats["mod_z"] = {"lower": -np.inf, "upper": np.inf, "med": med, "mad": mad}
                else:
                    # |0.6745*(x-med)/MAD| > z_thresh  ==>  x outside [med ± z_thresh*MAD/0.6745]
                    delta = (self.z_thresh * mad) / 0.6745
                    lower = med - delta
                    upper = med + delta
                    col_stats["mod_z"] = {"lower": lower, "upper": upper, "med": med, "mad": mad}

            self.stats_[col] = col_stats

        return self

    def transform(self, X):
        check_is_fitted(self, "stats_")
        df = pd.DataFrame(X).copy()
        df = df.reindex(columns=self.feature_names_in_)

        n_total_flagged = 0
        self.n_outliers_by_col_ = {}

        for col in getattr(self, "cols_", []):
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors="coerce").astype(float)

            # build vote counter
            votes = np.zeros(len(df), dtype=int)

            # IQR flags
            if "iqr" in self.stats_.get(col, {}):
                b = self.stats_[col]["iqr"]
                votes += ((s < b["lower"]) | (s > b["upper"])).astype(int)

            # Modified Z flags
            if "mod_z" in self.stats_.get(col, {}):
                b = self.stats_[col]["mod_z"]
                votes += ((s < b["lower"]) | (s > b["upper"])).astype(int)

            out_mask = votes >= self.min_votes
            n_flagged = int(out_mask.sum())
            self.n_outliers_by_col_[col] = n_flagged
            n_total_flagged += n_flagged

            if n_flagged == 0:
                continue

            if self.action == "nan":
                df.loc[out_mask, col] = np.nan

            elif self.action == "clip":
                # Winsorize: cap values instead of removing rows (keeps rare but valid cars)
                # For clipping we use the intersection of available bounds (most conservative):
                lowers = []
                uppers = []
                if "iqr" in self.stats_[col]:
                    lowers.append(self.stats_[col]["iqr"]["lower"])
                    uppers.append(self.stats_[col]["iqr"]["upper"])
                if "mod_z" in self.stats_[col]:
                    lowers.append(self.stats_[col]["mod_z"]["lower"])
                    uppers.append(self.stats_[col]["mod_z"]["upper"])

                lower_clip = max(lowers) if lowers else -np.inf
                upper_clip = min(uppers) if uppers else np.inf

                df[col] = s.clip(lower_clip, upper_clip)

            else:
                raise ValueError(f"OutlierHandler: unknown action='{self.action}'. Use 'nan' or 'clip'.")

        self.n_outliers_total_ = n_total_flagged
        if self.verbose:
            print(f"OutlierHandler: flagged total outliers (cells): {self.n_outliers_total_}")
            # print(self.n_outliers_by_col_)  # uncomment for detailed logging

        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)



################################################################################
# Missing Values (GroupImputer) 
################################################################################

class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Hierarchical imputer for numeric + categorical features.

    Idea:
    ----
    We have to  compute the median value for the train dataset and fill the missing values in train, validation and test set with the median from the train dataset.
    For each row with a missing value, fill it using statistics from "similar" rows first, and only fall back to global statistics if needed.

    Hierarchy for numeric columns (num_cols):
        1) median per (group_cols[0], group_cols[1])    > we use brand, model
        2) median per group_cols[0]                     > we use brand
        3) global median across all rows

    Hierarchy for categorical columns (cat_cols):
        1) mode per (group_cols[0], group_cols[1])      > we use brand, model
        2) mode per group_cols[0]                       > we use brand
        3) global mode across all rows

    Notes:
    -----
    - `group_cols` are used only to define groups; they themselves are not imputed.
    - `num_cols` and `cat_cols` can be given explicitly (lists of column names). If None, they are inferred from the dtypes in `fit`.
    """

    def __init__(self, group_cols=("brand", "model"), num_cols=None, cat_cols=None, fallback="__MISSING__"):
        self.group_cols = group_cols
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.fallback = fallback

    # helpers
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
          (e.g. "brand" appearing twice after some operations).
        - df["brand"] would then raise "Grouper for 'brand' not 1-dimensional".
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


    def fit(self, X, y=None):
        """
        Learn the group-level and global statistics from the training data.

        Steps
        -----
        1) Convert X to DataFrame and remember the original column order.
        2) Resolve which columns are numeric/categorical to impute.
        3) Build group keys (g0, g1) from group_cols (e.g. brand, model).
        4) For numeric columns:
            - compute global medians
            - medians per g0 (e.g. per brand)
            - medians per (g0, g1) (e.g. per brand+model)
        5) For categorical columns:
            - global modes
            - modes per g0
            - modes per (g0, g1)
        """

        df = pd.DataFrame(X).copy()
        self.feature_names_in_ = df.columns.to_list()

        # group_cols must contain at least one column name
        if self.group_cols is None or len(self.group_cols) == 0:
            raise ValueError("GroupImputer: at least one group column must be specified.")
        
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
            self.cat_cols_ = [c for c in df.columns if c not in self.group_cols_ + self.num_cols_]
        else:
            # If specified: keep only those that exist in df
            self.cat_cols_ = [c for c in self.cat_cols if c in df.columns]

        # Build group key series based on the current df
        # g0 = first grouping column (e.g. brand)
        g0 = self._get_group_series(df, self.group_cols_[0])

        # g1 = second grouping column (e.g. model), optional
        g1 = None
        if len(self.group_cols_) > 1:
            g1 = self._get_group_series(df, self.group_cols_[1])

        # numeric statistics
        if self.num_cols_:
            # Extract the numeric columns to impute
            num_df = df[self.num_cols_].copy()

            # 3) Global median per numeric column (fallback for any group with no stats)
            self.num_global_ = num_df.median(numeric_only=True)

            # 2) Median per first-level group (g0, e.g. brand)
            num_first = num_df.copy()
            num_first["_g0"] = g0.values  # temporary group key column
            self.num_first_ = num_first.groupby("_g0", dropna=True).median(numeric_only=True)

            # 1) Median per pair (g0, g1), e.g. (brand, model)
            if g1 is not None:
                num_pair = num_df.copy()
                num_pair["_g0"] = g0.values
                num_pair["_g1"] = g1.values
                self.num_pair_ = num_pair.groupby(["_g0", "_g1"], dropna=True).median(numeric_only=True)
            else:
                self.num_pair_ = pd.DataFrame()
        else:
            self.num_global_ = pd.Series(dtype="float64")
            self.num_first_ = pd.DataFrame()
            self.num_pair_ = pd.DataFrame()

        # categorical statistics
        if self.cat_cols_:
            cat_df = df[self.cat_cols_].copy()

            # 3) Global mode per categorical column
            self.cat_global_ = pd.Series({c: self._mode(cat_df[c]) for c in self.cat_cols_}, dtype="object")

            # 2) Mode per first-level group (g0)
            cat_first = cat_df.copy()
            cat_first["_g0"] = g0.values
            self.cat_first_ = cat_first.groupby("_g0", dropna=True).agg(lambda s: self._mode(s))

            # 1) Mode per pair (g0, g1)
            if g1 is not None:
                cat_pair = cat_df.copy()
                cat_pair["_g0"] = g0.values
                cat_pair["_g1"] = g1.values
                self.cat_pair_ = cat_pair.groupby(["_g0", "_g1"], dropna=True).agg(lambda s: self._mode(s))
            else:
                self.cat_pair_ = pd.DataFrame()
        else:
            self.cat_global_ = pd.Series(dtype="object")
            self.cat_first_ = pd.DataFrame()
            self.cat_pair_ = pd.DataFrame()

        return self


    def transform(self, X):
        """
        Apply hierarchical imputation to new data.
            1) Convert input to DataFrame and align columns to what fit() saw.
            2) Rebuild group keys g0, g1 from the current data.
            3) For each numeric column with missing values:
                - try pair-level median (g0, g1)
                - then brand-level median (g0)
                - then global median
            4) Same for categorical columns with modes.
        """
        df = pd.DataFrame(X).copy()
        df = df.reindex(columns=self.feature_names_in_)

        g0 = self._get_group_series(df, self.group_cols_[0])
        g1 = None
        if len(self.group_cols_) > 1:
            g1 = self._get_group_series(df, self.group_cols_[1])

        # numeric imputation
        if hasattr(self, "num_cols_") and self.num_cols_:
            df[self.num_cols_] = df[self.num_cols_].astype("float64")
            to_impute_num = [c for c in self.num_cols_ if df[c].isna().any()]

            if to_impute_num:
                # 1) pair-level imputation: per (g0, g1)
                if g1 is not None and not self.num_pair_.empty:
                    key_df = pd.DataFrame({"_g0": g0.values, "_g1": g1.values})
                    med_df = self.num_pair_.reset_index()
                    joined = key_df.merge(med_df, on=["_g0", "_g1"], how="left")

                    for col in to_impute_num:
                        if col not in self.num_pair_.columns:
                            continue
                        mask = df[col].isna() & joined[col].notna()
                        df.loc[mask, col] = joined.loc[mask, col]

                # 2) first-level imputation: per g0 only
                if not self.num_first_.empty:
                    key1 = pd.DataFrame({"_g0": g0.values})
                    med1 = self.num_first_.reset_index()
                    joined1 = key1.merge(med1, on="_g0", how="left")

                    for col in to_impute_num:
                        if col not in self.num_first_.columns:
                            continue
                        mask = df[col].isna() & joined1[col].notna()
                        df.loc[mask, col] = joined1.loc[mask, col]

                # 3) global median fallback
                for col in to_impute_num:
                    if col in self.num_global_:
                        df[col] = df[col].fillna(self.num_global_[col])

        # categorical imputation
        if hasattr(self, "cat_cols_") and self.cat_cols_:
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
                        df.loc[mask, col] = joined.loc[mask, col]

                # 2) first-level imputation: per g0 only
                if not self.cat_first_.empty:
                    key1 = pd.DataFrame({"_g0": g0.values})
                    mode1 = self.cat_first_.reset_index()
                    joined1 = key1.merge(mode1, on="_g0", how="left")

                    for col in to_impute_cat:
                        if col not in self.cat_first_.columns:
                            continue
                        mask = df[col].isna() & joined1[col].notna()
                        df.loc[mask, col] = joined1.loc[mask, col]

                # 3) global mode fallback (or fallback token)
                for col in to_impute_cat:
                    if col in self.cat_global_:
                        df[col] = df[col].fillna(self.cat_global_[col])
                    else:
                        df[col] = df[col].fillna(self.fallback)

        return df

    def get_feature_names_out(self, input_features=None):
        """
        Make the transformer compatible with sklearn's get feature-name.

        - If called without arguments, return the original feature names seen in fit().
        - This is mostly useful when GroupImputer is at the top of a Pipeline and
          later steps want to introspect feature names.
        """
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)



################################################################################
# Feature Engineering (CarFeatureEngineer)
################################################################################

class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates domain-informed numeric features inside the pipeline (leakage-safe).

    Why fit() exists:
    - We learn fold-specific reference statistics (e.g., average age per brand)
      using only the training fold. This prevents leakage in cross-validation.

    Note about `y` in fit():
    - sklearn pipelines call fit(X, y) on every step. Even if we don't use y,
      the signature must accept it.
    """

    def __init__(self, ref_year=None):
        self.ref_year = ref_year

    def fit(self, X, y=None):  # y is necessary because 3 arguments are given in pipeline
        X_ = X.copy()
        if self.ref_year is None:
            self.ref_year_ = X_["year"].max()
        else:
            self.ref_year_ = self.ref_year

        self.brand_mean_age_ = ((self.ref_year_ - X_["year"]).groupby(X_["brand"]).mean().to_dict())
        self.model_mean_age_ = ((self.ref_year_ - X_["year"]).groupby(X_["model"]).mean().to_dict())

        self.model_mean_mileage_ = (X_["mileage"].groupby(X_["model"]).mean().to_dict())
        self.model_mean_engineSize_ = (X_["engineSize"].groupby(X_["model"]).mean().to_dict())

        self.model_freq_ = X_["model"].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()

        # Available num features:
        # orig_numeric_features = ["year", "mileage", "tax", "mpg", "engineSize", "previousOwners"] # though previousOwners has now correlations
        # orig_categorical_features = ["brand", "model", "transmission", "fuelType"]
        # unused_features = ['hasDamage', 'paintQuality']

        # 1. Base Feature Creation:
        #       - Car Age - Newer cars usually have higher prices, models prefer linear features
        age = self.ref_year_ - X["year"]
        X["age"] = age

        # 2. Interaction effects to capture non-additive information (learn conditional relationships and potentially skyrocket their importance):
        #       - It helps to solve multicolinearity between features by combining them into one feature creating a new signal
        #       => Only spearman correlations > 0.2 are regarded
        #       - Use Multiplication if we think two features "boost" each other (e.g., Length*Width = Area).
        #       - Use Division if we need to "fairly compare" items of different sizes (e.g., Cost/Weight = Price per kg)
        #       -> Mult or Div has to be chosen based on the logic of the relationship
        #       Multiplication: The Amplifier (model synergy or joint occurrence: "The presence of A makes B more effective") -> capture simultaneous impact of two things

        X["mpg_x_engine"] = X["mpg"] * X["engineSize"] 

        # Removed because of high multicolinearity and lower corr with price: X['mileage_x_mpg']          = X['mileage'] * X[s'mpg'] # Higher mileage cars tend to have lower MPG (people drive lower mpg cars more often) -> amplify effect

        # Add 1 to age because if age is 0 (this year) the value would be lost otherwise
        X["engine_x_age"] = X["engineSize"] * (X["age"] + 1)  # Highlight the aspect of old cars with big engines for that time which were very valuable and might therefore still be valuable

        X["mileage_x_age"] = X["mileage"] * (X["age"] + 1)  # Both are negatively correlated with price -> amplify effect to identify a stronger signal of old abused cars that are probably less valuable

        X["mpg_x_age"] = X["mpg"] * (X["age"] + 1)  # Older cars tend to have higher MPG -> amplify effect
        X["tax_x_age"] = X["tax"] * (X["age"] + 1)

        ###### Division: The Normalizer (create ratios, rates, or efficiency metrics: "How much of A do we have per unit of B?") -> removes the influence of the divisor
        ### Normalize by Age to capture how features behave relative to the car's age

        # Miles per Year: Normalizes mileage by age -> reveals how much a car was really driven per year
        X["miles_per_year"] = X["mileage"] / (X["age"] + 1)  # Add 1 to age because if age is 0 (this year) the division would fail (dont impute with 1 bc then its the same as 1 year old instead of being from this year)

        # tax normalized by engine and/or per mpg to focus on the tax of the car regardless of the other factor (prefered to keep engine because engine is the cause and mpg the effect but corr with price of mpg was higher (0.46 and -0.06))
        X["tax_per_mpg"] = X["tax"] / X["mpg"]  # No 0-handling necessary because mpg cannot be 0 (we only keep values from 5-150 and impute the others)

        # engine per mpg creates a signal for sports/luxury cars that have a high engine size but low mpg (high performance cars) -> these cars are usually more valuable
        X["engine_per_mpg"] = X["engineSize"] / X["mpg"]  # No 0-handling necessary because engineSize cannot be 0 (we only keep values from 0.6–12.7 and impute the others)

        ############ Create Interaction Features for anchor (relative positioning within brand/model)
        X["brand_fuel"] = X["brand"].astype(str) + "_" + X["fuelType"].astype(str)
        X["brand_trans"] = X["brand"].astype(str) + "_" + X["transmission"].astype(str)

        ############ Features based on learned statistics from the available data fold in the fit() method:
        X["model_freq"] = X["model"].map(self.model_freq_).fillna(0.0)  # Model Frequency: Popular models tend to have stable demand and prices

        ############ Relative Age (within brand): newer/older than brand median year (fill with 0 if brand or model was not seen during training)
        X["age_rel_brand"] = X["age"] - X["brand"].map(self.brand_mean_age_).fillna(0.0)  # use mean instead of median because most of the values were 0 otherwise
        X["age_rel_model"] = X["age"] - X["model"].map(self.model_mean_age_).fillna(0.0)
        
        # fill with 1 if model was not seen during training
        X["engine_rel_model"] = (X["engineSize"] / X["model"].map(self.model_mean_engineSize_).fillna(1.0)).fillna(1.0)  # engine size relative to model mean engine size

        # TODO tax divided by mean model price (affordability within model) # Before that: check whether road tax varies per model
        return X



################################################################################
# Feature Selection
################################################################################

# Explanation of Leakage Prevention:
#       0) Function call: pass the preprocessor_pipe (which contains the Majority Voter) into RandomizedSearchCV:
#       1) Splitting: The search CV splits the data into Train and Validation folds.
#       2) Fitting: It calls .fit() on your pipeline using only the Train fold.
#       3) Voting: The custom MajorityVoteSelectorTransformer runs inside the pipeline. It sees only the Train fold. It calculates votes and selects features based only on that fold.
#       4) Transformation: It transforms the Validation fold based on the features selected in step 3.
#       ==> Leakage Free: Since the Validation fold was never used to decide which features to keep, there is no leakage.

# Explanation of why it is not a problem for the final refit that different features might have been selected in different folds:
#       0) Final refit is called on best hyperparameters found during CV.
#       1) The MajorityVoteSelectorTransformer sees the entire training data during final refit.
#       2) It calculates votes and selects features based on the entire training data. (This is done without hp-tuning now because the hps are fixed.)
#       3) It transforms the entire training data based on the features selected in step 2.
#       ==> No Problem: Although different folds might have selected different features during CV, the final refit uses the entire training data to select only one final set of features (which might vary from previous features selected in the folds but thats not a problem).


class MajorityVoteSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Runs multiple feature selectors and keeps a feature if at least `min_votes` selectors agree.
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
            self.feature_names_in_ = np.array(X.columns)  # store input feature names if available for the get_feature_names_out method

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

################################################################################

class SpearmanRelevancyRedundancySelector(BaseEstimator, SelectorMixin):
    """
    Selects features based on:
      1. Relevance: High Spearman correlation with the target.
      2. Non-Redundancy: Low Spearman correlation with already selected features (drops redundant variable with lower correlation to target).

    More detailed (Algorithm: “Maximum Relevance, Minimum Redundancy (mRMR)-style pruning”):
    1. Sort features by relevance.
    2. Start with an empty list of selected features.
    3. For each feature (in order of relevance):
    4. Compare it with all already-selected features.
    5. If its |corr| with any selected feature > threshold, skip it.
    6. Otherwise, keep the feature.

    Parameters:
    ----------
    relevance_threshold : float
        Minimum absolute Spearman correlation with target to consider a feature 'relevant'.
    redundancy_threshold : float
        Maximum absolute Spearman correlation allowed between a new feature and already selected features.
    """

    def __init__(self, relevance_threshold=0.1, redundancy_threshold=0.85):
        self.relevance_threshold = relevance_threshold
        self.redundancy_threshold = redundancy_threshold
        self.selected_indices_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        # 1. Input Validation and Feature Name Capture
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array([str(c) for c in X.columns])

        # Convert to Numpy for speed, ensure y is correct shape
        X_arr, y_arr = check_X_y(X, y, dtype=None)
        n_features = X_arr.shape[1]

        # TODO maybe use Kendalls tau or cramers V for categorical features?

        # 1) Relevance Filtering (Filter out weak features first)
        relevance_scores = []
        for i in range(n_features):
            # Calculate spearman with target
            corr, _ = spearmanr(X_arr[:, i], y_arr)
            relevance_scores.append(abs(corr))

        self.relevance_scores_ = np.array(relevance_scores)
        relevance_indices = np.where(self.relevance_scores_ > self.relevance_threshold)[0]

        # Sort candidates by relevance (Best feature first (descending) -> argsort gives ascending, so we take [::-1])
        sorted_candidates = relevance_indices[np.argsort(self.relevance_scores_[relevance_indices])[::-1]]

        # 2) Eliminate Redundant Features (Remove the one with lower relevance)
        selected_indices = []

        # Optimization: Create a DataFrame of just the candidates for fast matrix corr
        if len(sorted_candidates) > 0:
            X_candidates = pd.DataFrame(X_arr[:, sorted_candidates])  # pandas for the correlation matrix as it handles Spearman efficiently
            corr_matrix = X_candidates.corr(method="spearman").abs().values

            # Map: corr_matrix index [i] corresponds to sorted_candidates[i]
            kept_local_indices = []

            for i in range(len(sorted_candidates)):
                # Always keep the single most relevant feature (i=0) because there is no possible redundant feature here yet
                if i == 0:
                    kept_local_indices.append(i)
                    continue

                # Check correlation with al features that passed relevance threshold (corr_matrix[i, kept_local_indices] gives array of corrs)
                current_corrs = corr_matrix[i, kept_local_indices]

                # If the max correlation with any selected feature is too high, drop it
                if np.max(current_corrs) < self.redundancy_threshold:
                    kept_local_indices.append(i)

            # Convert local kept indices back to original feature indices
            selected_indices = sorted_candidates[kept_local_indices]

        self.selected_indices_ = np.array(selected_indices)
        return self

    def _get_support_mask(self):
        """
        Required by SelectorMixin. Returns boolean mask of selected features.
        """
        check_is_fitted(self, "selected_indices_")
        n_features = len(self.relevance_scores_)
        mask = np.zeros(n_features, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        """
        Ensures proper feature names are passed to the next step.
        """
        if input_features is not None:
            names = np.array(input_features)
        elif self.feature_names_in_ is not None:
            names = self.feature_names_in_
        else:
            names = np.array([f"x{i}" for i in range(len(self.relevance_scores_))])

        return names[self._get_support_mask()]

################################################################################

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
        self.scores_ = mutual_info_regression(X, y, n_neighbors=self.n_neighbors, random_state=self.random_state)

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



################################################################################
# Helpers for FunctionTransformer
################################################################################

class NamedFunctionTransformer(FunctionTransformer):  # TODO check if this is really necessary or can be removed when implementing get_feature_names_out everywhere clean
    """
    FunctionTransformer variant that can expose feature names downstream.

    This is helpful if:
    - You use set_output(transform="pandas") and want meaningful column names
    - You use DebugTransformer and want readable outputs
    """

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
        if hasattr(self, "n_features_in_"):
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        # Cannot determine feature names
        return None


# TODO warum benutzen wir das => auch bei pipeline adden
def to_float_array(x):
    """Convert input to float array."""
    return np.array(x, dtype=float)



################################################################################
# Hyperparameter tuning
################################################################################

def model_hyperparameter_tuning(X_train, y_train, cv, pipeline, param_dist, n_iter=100):
    """
    Helper for RandomizedSearchCV with consistent scoring output.

    Parameters
    ----------
    X_train, y_train:
        Training data (already split; no leakage handling is done here).
    pipeline:
        sklearn Pipeline (should include cleaning/imputation/feature engineering).
    param_dist:
        dict of parameter distributions for RandomizedSearchCV.
    n_iter:
        number of random hyperparameter combinations.
    splits:
        number of CV folds (KFold).

    Returns
    -------
    best_estimator_ : fitted pipeline
    model_random : RandomizedSearchCV object (fitted)
    model_scores : dict with train/val metrics
    """

    # Randomized search setup
    model_random = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,  # number of different hyperparameter combinations that will be randomly sampled and evaluated (more iterations = more thorough search but longer runtime)
        scoring={"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error", "r2": "r2"},
        refit="mae",  # Refit the best model based on MAE on the whole training set
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=3,
        return_train_score=True,
    )

    # Fit the search
    model_random.fit(X_train, y_train)

    val_mae = -model_random.cv_results_["mean_test_mae"][model_random.best_index_]
    val_mse = -model_random.cv_results_["mean_test_mse"][model_random.best_index_]
    val_rmse = np.sqrt(val_mse)
    val_r2 = model_random.cv_results_["mean_test_r2"][model_random.best_index_]

    train_mae = -model_random.cv_results_["mean_train_mae"][model_random.best_index_]
    train_mse = -model_random.cv_results_["mean_train_mse"][model_random.best_index_]
    train_rmse = np.sqrt(train_mse)
    train_r2 = model_random.cv_results_["mean_train_r2"][model_random.best_index_]

    model_scores = {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
    }

    print("Model Train Results")
    print(f"MAE: {train_mae:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"R²: {train_r2:.4f}")

    print("Model Results (CV metrics):")
    print(f"MAE: {val_mae:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"R²: {val_r2:.4f}")
    print("Best Model params:", model_random.best_params_)

    return model_random.best_estimator_, model_random, model_scores
