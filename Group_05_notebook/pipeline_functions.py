"""
pipeline_functions.py — Cars4You

Reading guide:
-----------------------------
This file contains helper functions (e.g. for tuning) and small building blocks ("transformers") 
that are chained into a single sklearn Pipeline. Each transformer does one job:

1) CarDataCleaner               : fixes obvious issues (typos, impossible numeric ranges) without dropping rows
2) OutlierHandler               : reduces the impact of extreme numeric values (winsorization / clipping)
3) IndividualHierarchyImputer       : fills missing values using statistics from similar cars first
4) CarFeatureEngineer           : creates additional signals (age, ratios, interactions, relative positioning)
5) Feature selection            : keeps only helpful signals and drops redundant noise

Important rule: transformers must NOT drop rows inside transform(),
otherwise X and y get misaligned during cross-validation.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectorMixin, mutual_info_regression
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, check_X_y
import time

from scipy.stats import spearmanr

from difflib import get_close_matches

#!pip install ydata_profiling
from ydata_profiling import ProfileReport

from collections import Counter
from IPython.display import display
import matplotlib.pyplot as plt


def _print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _maybe_display(df_or_obj, max_rows=10):
    """Show a small table in notebooks; fall back to print in scripts."""
    if display is not None:
        display(df_or_obj if not hasattr(df_or_obj, "head") else df_or_obj.head(max_rows))
    else:
        print(df_or_obj if not hasattr(df_or_obj, "head") else df_or_obj.head(max_rows))


################################################################################
# Canonical maps for Cleaning (shared between fit() and transform())
################################################################################
 
BRAND_MAP = {
    "VW": ["V", "W", "vw", "v", "w"],
    "Toyota": ["toyot", "toyota", "oyota",],
    "Audi": ["au", "a", "udi", "aud", "audi"],
    "Ford": ["ford", "ord", "for", "or", "rd", "fo"],
    "BMW": ["bmw", "b", "bm", "mw"],
    "Skoda": ["sk", "koda", "sko", "kod", "skoda", "skod", "oda"],
    "Opel": ["op", "opel", "pel", "pe", "ope"],
    "Mercedes": ["mercedes", "mercede", "ercedes", "merc", "cedes", "ercede", "edes"],
    "Hyundai": ["hyundai", "hyun", "yundai", "yunda", "yund", "hyunda", "yun", "hyund"],
}
REVERSE_BRAND = {v.lower(): k for k, vals in BRAND_MAP.items() for v in vals}
BRAND_NORM_TO_CANON = {k.lower(): k for k in BRAND_MAP.keys()}
BRAND_CANON_VOCAB = sorted(list(BRAND_NORM_TO_CANON.keys()))
 
TRANS_MAP = {
    "Manual": ["manual", "manua", "anual", "nual", "manu"],
    "Semi-Auto": ["semi-auto", "semi-aut", "emi-auto", "sem", "semi", "semiauto"],
    "Automatic": ["automatic", "automati", "auto", "utomatic", "autom"],
    "Unknown": ["unknown", "unknow", "nknown", "unkno"],
    "Other": ["other", "ther", "othe"],
}
REVERSE_TRANS = {v.lower(): k for k, vals in TRANS_MAP.items() for v in vals}
TRANS_NORM_TO_CANON = {k.lower(): k for k in TRANS_MAP.keys()}
TRANS_CANON_VOCAB = sorted([k.lower() for k in TRANS_MAP.keys() if k != "Unknown"])
 
FUEL_MAP = {
    "Petrol": ["petrol", "petro", "etrol", "etro"],
    "Diesel": ["diesel", "dies", "iesel", "diese", "iese", "diesele"],
    "Hybrid": ["hybrid", "ybri", "hybri", "ybrid", "hybridd"],
    "Electric": ["electric", "elec", "electronic", "tric"],
    "Other": ["other", "ther", "othe"],
}
REVERSE_FUEL = {v.lower(): k for k, vals in FUEL_MAP.items() for v in vals}
FUEL_NORM_TO_CANON = {k.lower(): k for k in FUEL_MAP.keys()}
FUEL_CANON_VOCAB = sorted([k.lower() for k in FUEL_MAP.keys()])
 
MODEL_MAP = {
    # VW
    "golf": ["golf", "gol", "golf s", "golf sv"],
    "passat": ["passat", "passa", "pass"],
    "polo": ["polo", "pol", "olo"],
    "tiguan": ["tiguan", "tigua", "tiguan allspace", "tiguan allspac"],
    "touran": ["touran", "toura", "ouran"],
    "tourneo connect": ["tourneo connect"],
    "up": ["up", "u", "p"],
    "sharan": ["sharan", "shara", "shar"],
    "scirocco": ["scirocco", "sciroc", "scirocc"],
    "amarok": ["amarok", "amaro", "amar"],
    "arteon": ["arteon", "arteo", "teon"],
    "beetle": ["beetle", "beetl", "beet"],
    "fox": ["fox", "ox", "fo"],
    "california": ["california", "alifornia", "cali"],
    "t-roc": ["t-roc", "t-ro", "t roc"],
    "t-cross": ["t-cross", "t-cros", "t cross"],
    "touareg": ["touareg", "touare", "touare g"],
    "jetta": ["jetta", "etta", "jet"],
    "cc": ["cc", "c"],
    "eos": ["eos", "os", "eo"],
    "caravelle": ["caravelle", "caravell"],
    "caddy": ["caddy", "addy", "cad"],
    "caddy life": ["caddy life"],
    "caddy maxi": ["caddy maxi"],
    "caddy maxi life": ["caddy maxi life", "caddy maxi lif"],
 
    # Toyota
    "yaris": ["yaris", "yari", "yar"],
    "corolla": ["corolla", "corol", "coroll"],
    "aygo": ["aygo", "ayg", "ygo"],
    "rav4": ["rav4", "rav", "rav-4"],
    "auris": ["auris", "auri", "uris"],
    "avensis": ["avensis", "avens", "ensis"],
    "c-hr": ["c-hr", "chr", "c-h"],
    "verso": ["verso", "verso-s", "vers"],
    "hilux": ["hilux", "hilu", "lux"],
    "land cruiser": ["land cruiser", "land cruise"],
    "prius": ["prius", "rius"],
    "proace verso": ["proace verso"],
    "gt86": ["gt86", "gt"],
    "supra": ["supra", "sup", "supr"],
    "camry": ["camry", "amry", "mry"],
    "urban cruiser": ["urban cruiser", "urban cruise"],
    "iq": ["iq"],  
    "shuttle": ["shuttle", "uttle", "shut"],  
 
    # Audi
    "a_unknown": ["a_unknown"],
    "q_unknown": ["q_unknown"],
    "a1": ["a1", "a 1"],
    "a2": ["a2", "a 2"],
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
    "sq5": ["sq5"],
    "sq7": ["sq7"],
    "tt": ["tt"],
    "r8": ["r8", "r 8"],
    "rs3": ["rs3", "rs 3"],
    "rs4": ["rs4", "rs 4"],
    "rs5": ["rs5", "rs 5"],
    "rs6": ["rs6", "rs 6"],
    "s3": ["s3", "s 3"],
    "s4": ["s4", "s 4"],
    "s5": ["s5", "s 5"],
    "s8": ["s8", "s 8"],
 
    # Ford
    "fiesta": ["fiesta", "fiest"],
    "focus": ["focus", "focu"],
    "mondeo": ["mondeo", "monde"],
    "kuga": ["kuga", "kug"],
    "ecosport": ["ecosport", "eco sport", "ecospor"],
    "puma": ["puma", "pum"],  
    "edge": ["edge", "edg"],
    "s-max": ["s-max", "s-ma", "smax"],
    "c-max": ["c-max", "c-ma", "cmax"],
    "b-max": ["b-max", "b-ma", "bmax"],
    "ka+": ["ka+", "ka", "streetka"],
    "fusion": ["fusion", "fusio"],
    "galaxy": ["galaxy", "galax"],
    "tourneo custom": ["tourneo custom", "tourneo custo"],
    "grand tourneo connect": ["grand tourneo connect", "grand tourneo connec"],
    "mustang": ["mustang", "ustang"],
    "ranger": ["ranger", "anger"],
    "escort": ["escort", "cord"],
    "grand c-max": ["grand c-max", "grand c-ma"],
    "rapid": ["rapid", "rapi"],
 
    # BMW
    "1 series": ["1 series", "1 serie", "1 ser", "1series"],
    "2 series": ["2 series", "2 serie", "2series"],
    "3 series": ["3 series", "3 serie", "3series"],
    "4 series": ["4 series", "4 serie", "4series"],
    "5 series": ["5 series", "5 serie", "5series"],
    "6 series": ["6 series", "6 serie", "6series"],
    "7 series": ["7 series", "7 serie", "7series"],
    "8 series": ["8 series", "8 serie", "8series"],
    "x_unknown": ["x_unknown"],
    "x1": ["x1"],
    "x2": ["x2"],
    "x3": ["x3"],
    "x4": ["x4"],
    "x5": ["x5"],
    "x6": ["x6"],
    "x7": ["x7"],
    "z3": ["z3"],
    "z4": ["z4"],
    "m2": ["m2"],
    "m3": ["m3"],
    "m4": ["m4"],
    "m5": ["m5"],
    "m6": ["m6"],
    "i1": ["i1"],  
    "i2": ["i2"],
    "i3": ["i3"],
    "i8": ["i8"],
 
    # Skoda
    "fabia": ["fabia", "fabi"],
    "octavia": ["octavia", "octavi", "octa"],
    "superb": ["superb", "super", "sup"],
    "scala": ["scala", "scal", "cala"],
    "karoq": ["karoq", "karo", "aroq"],
    "kodiaq": ["kodiaq", "kodia", "kodi"],
    "kamiq": ["kamiq", "kami", "miq"],
    "yeti": ["yeti", "yeti outdoor", "yeti outdoo", "yet"],
    "citigo": ["citigo", "citig", "itigo"],
    "roomster": ["roomster", "roomste", "room"],
 
    # Opel
    "astra": ["astra", "astr", "gtc"],
    "corsa": ["corsa", "cors", "orsa"],
    "insignia": ["insignia", "insigni"],
    "mokka": ["mokka", "mokk", "mokka x", "mokkax"],
    "zafira": ["zafira", "zafir", "fira"],
    "zafira tourer": ["zafira tourer", "zafira toure"],
    "meriva": ["meriva", "meriv", "riva"],
    "adam": ["adam", "ad", "ada"],
    "vectra": ["vectra", "vectr", "ctra"],
    "antara": ["antara", "anta", "tara"],
    "combo life": ["combo life", "combo lif"],
    "grandland x": ["grandland x", "grandland"],
    "crossland x": ["crossland x", "crossland"],
    "cascada": ["cascada", "ascada"],
    "ampera": ["ampera", "mpera"],
    # you wont find leakage here *** :-)
    "tigra": ["tigra", "igra"],
    "vivaro": ["vivaro", "varo"],
    "viva": ["viva", "viv"],
    "agila": ["agila", "agil"],
 
    # Mercedes
    "a class": ["a class", "a clas", "a-class"],
    "b class": ["b class", "b clas", "b-class"],
    "c class": ["c class", "c clas", "c-class"],
    "e class": ["e class", "e clas", "e-class"],
    "s class": ["s class", "s clas", "s-class"],
    "glc class": ["glc class"],
    "gle class": ["gle class", "gle clas"],
    "gla class": ["gla class", "gla clas"],
    "cls class": ["cls class", "cls clas"],
    "glb class": ["glb class", "glb"],
    "gls class": ["gls class", "gls clas"],
    "gl class": ["gl class", "gl clas"],
    "cla class": ["cla class", "cla"],
    "m class": ["m class", "m clas"],
    "sl class": ["sl class", "sl clas", "sl"],  
    "cl class": ["cl class", "cl clas"],
    "clc class": ["clc class", "clc"],
    "v class": ["v class", "v clas"],
    "x-class": ["x-class", "x-clas"],
    "g class": ["g class", "g-clas"],
    "slk": ["slk"],
    "clk": ["clk"],
 
    # Hyundai
    "i10": ["i10", "i 10"],
    "i20": ["i20", "i 20"],
    "i30": ["i30", "i 30"],
    "i40": ["i40", "i 40"],
    "ioniq": ["ioniq", "ioni"],
    "ix20": ["ix20", "ix 20", "ix2"],  
    "ix35": ["ix35", "ix 35"],
    "kona": ["kona", "kon", "ona"],
    "tucson": ["tucson", "tucso", "tucs"],
    "santa fe": ["santa fe", "santa f", "santa"],
    "getz": ["getz", "etz"],
    "accent": ["accent", "acce", "ccent"],
    "i800": ["i800", "i80"],
    "veloster": ["veloster", "veloste", "eloster"],
    "terracan": ["terracan", "terra"],
 
}

REVERSE_MODEL = {v.lower(): k for k, vals in MODEL_MAP.items() for v in vals}

# model -> brand mapping: there are rows where model is filled but brand is not.
# We can back-fill brand from model via this mapping.
MODEL_TO_BRAND = {}
for brand, models in {
    "VW": [
        "golf", "passat", "polo", "tiguan", "touran", "up", "sharan", "scirocco", "amarok", "arteon", "beetle", "fox",
        "t-roc", "t-cross", "touareg", "jetta", "cc", "eos", "caravelle",
        "caddy", "caddy life", "caddy maxi", "caddy maxi life", "california"
    ],
    "Toyota": [
        "yaris", "corolla", "aygo", "rav4", "auris", "avensis", "c-hr", "verso", "hilux", "land cruiser",
        "prius", "proace verso", "gt86", "supra", "camry", "urban cruiser", "iq", "shuttle",
    ],
    "Audi": [
        "a_unknown", "q_unknown",
        "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8",
        "q2", "q3", "q5", "q7", "q8", "sq5", "sq7",
        "tt", "r8", "rs3", "rs4", "rs5", "rs6", "s3", "s4", "s5", "s8",
    ],
    "Ford": [
        "fiesta", "focus", "mondeo", "kuga", "ecosport", "puma", "edge", "s-max", "c-max", "b-max", "ka+", "fusion",
        "galaxy", "tourneo custom", "tourneo connect", "grand tourneo connect", "mustang", "ranger", "escort",
        "grand c-max", "rapid",
    ],
    "BMW": [
        "1 series", "2 series", "3 series", "4 series", "5 series", "6 series", "7 series", "8 series",
        "x_unknown", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "z3", "z4",
        "m2", "m3", "m4", "m5", "m6",
        "i1", "i2", "i3", "i8",
    ],
    "Skoda": ["fabia", "octavia", "superb", "scala", "karoq", "kodiaq", "kamiq", "yeti", "citigo", "roomster"],
    "Opel": [
        "astra", "corsa", "insignia", "mokka", "zafira", "zafira tourer", "meriva", "adam", "vectra", "antara",
        "combo life", "grandland x", "crossland x", "cascada", "ampera", "tigra", "vivaro", "viva", "agila", "gtc",
    ],
    "Mercedes": [
        "a class", "b class", "c class", "e class", "s class",
        "glc class", "gle class", "gla class", "cls class", "glb class", "gls class", "gl class", "cla class",
        "m class", "sl class", "cl class", "clc class", "v class", "x-class", "g class", "slk", "clk",
    ],
    "Hyundai": [
        "i10", "i20", "i30", "i40", "ioniq", "ix20", "ix35", "kona", "tucson", "santa fe",
        "getz", "accent", "i800", "veloster", "terracan",
    ],
}.items():
    for m in models:
        MODEL_TO_BRAND[m] = brand

################################################################################
# Data Cleaning (CarDataCleaner)
################################################################################

class CarDataCleaner(BaseEstimator, TransformerMixin):
    """
    CarDataCleaner

    What it does:

    1) Numeric sanity checks:
       - coercion to numeric types
       - invalid ranges are set to NaN (NOT dropped) so imputers can handle them

    2) Categorical canonicalization:
       - normalize case/whitespace
       - map known typos/variants into one canonical label using explicit dictionaries

    3) Conservative fuzzy fallback:
       - Only tries to fill values that are still missing AFTER deterministic mapping
       - Vocabulary is learned in fit() from the training fold -> leakage-safe in CV
       - Guardrails prevent dangerous guessing (e.g., 1-letter tokens like "a")

    Notes
    -----
    - paintQuality is dropped because it is not available for predictions (filled by mechanic).
    - electricVehicles are handled because there are only 4 cars and 2 entries are wrong. 
      In addition, their price and characteristics structure differs strongly (cannot generalize) from combustion cars.
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

        # STRICT model policy: accept ONLY static canonical keys as "valid"
        strict_static_model_vocab=True,

        # fuzzy fallback (very strict; only rescues obvious typos)
        use_fuzzy=True,
        fuzzy_cutoff_brand=0.97,
        fuzzy_cutoff_model=0.97,
        fuzzy_cutoff_trans=0.97,
        fuzzy_cutoff_fuel=0.97,
        fuzzy_min_token_len=2,

        # model-only “unique prefix” rescue (safe + makes fuzzy visible)
        model_prefix_min_len=3,
        fuzzy_require_brand_for_model=True,

        # verbosity controls
        verbose=False,
        verbose_top_n=15,
        verbose_plot=False,
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

        # model strictness
        self.strict_static_model_vocab = strict_static_model_vocab

        # fuzzy behavior
        self.use_fuzzy = use_fuzzy
        self.fuzzy_cutoff_brand = fuzzy_cutoff_brand
        self.fuzzy_cutoff_model = fuzzy_cutoff_model
        self.fuzzy_cutoff_trans = fuzzy_cutoff_trans
        self.fuzzy_cutoff_fuel = fuzzy_cutoff_fuel
        self.fuzzy_min_token_len = fuzzy_min_token_len

        self.model_prefix_min_len = model_prefix_min_len
        self.fuzzy_require_brand_for_model = fuzzy_require_brand_for_model

        # verbose
        self.verbose = verbose
        self.verbose_top_n = verbose_top_n
        self.verbose_plot = verbose_plot


    # Helpers (string normalization)
    @staticmethod
    def _norm_str_series(s: pd.Series) -> pd.Series:
        """Lowercase + strip without turning NaN into the string 'nan'."""
        return s.astype(str).str.strip().str.lower().replace("nan", np.nan)

    @staticmethod
    def _canon_map(series: pd.Series, reverse_map: dict, keep_unmapped: bool = False) -> pd.Series:
        """
        Map known variants to canonical labels using a reverse dictionary:
          reverse_map[variant_lower] = canonical_label

        keep_unmapped=False:
          - strict mode: if token not in reverse_map -> NaN
        keep_unmapped=True:
          - permissive mode: if token not in reverse_map -> keep normalized token
        """
        s = series.astype(str).str.strip().str.lower().replace("nan", np.nan)
        mapped = s.map(reverse_map)
        return mapped.fillna(s) if keep_unmapped else mapped

    @staticmethod
    def _fuzzy_one(token: str, choices: list[str], cutoff: float) -> str | None:
        tok = str(token).strip().lower()
        if tok == "" or tok == "<na>":
            return None
        match = get_close_matches(tok, choices, n=1, cutoff=cutoff)
        return match[0] if match else None

    @staticmethod
    def _unique_prefix_match(token: str, choices: list[str], min_len: int) -> str | None:
        """
        Safe model fuzzy:
        - Accept only if token is a UNIQUE prefix of exactly one choice.
        - This is why “pum -> puma” can work without enabling risky fuzzy.
        """
        tok = str(token).strip().lower()
        if len(tok) < int(min_len):
            return None
        cands = [c for c in choices if c.startswith(tok)]
        return cands[0] if len(cands) == 1 else None


    def fit(self, X, y=None):
        """
        Learn training-fold vocab for optional fuzzy fallback (leakage-safe).
        For this project we keep categorical fuzzy vocab STRICT and canonical.
        """
        self.brand_vocab_ = BRAND_CANON_VOCAB
        self.trans_vocab_ = TRANS_CANON_VOCAB
        self.fuel_vocab_ = FUEL_CANON_VOCAB

        # model vocab: strict static canonical keys (lowercase)
        self.model_static_vocab_ = sorted([k.strip().lower() for k in MODEL_MAP.keys()])
        static_set = set(self.model_static_vocab_)

        # brand -> model choices (only canonical models that are in static vocab)
        df = pd.DataFrame(X).copy()
        if "Brand" in df.columns and "brand" not in df.columns:
            df = df.rename(columns={"Brand": "brand"})

        self.brand_to_models_ = {}
        if "brand" in df.columns and "model" in df.columns:
            b = self._canon_map(df["brand"], REVERSE_BRAND, keep_unmapped=False)
            m = self._canon_map(df["model"], REVERSE_MODEL, keep_unmapped=True).astype(str).str.strip().str.lower()
            tmp = pd.DataFrame({"brand": b, "model": m}).dropna()
            tmp = tmp[tmp["model"].isin(static_set)]

            for brand_val, g in tmp.groupby("brand"):
                self.brand_to_models_[str(brand_val).strip().lower()] = sorted(g["model"].unique().tolist())

        return self


    def transform(self, X):
        df = pd.DataFrame(X).copy()

        self.clean_report_ = {"numeric_new_nans": {}, "categorical_changes": {}, "notes": []}
        self.fuzzy_matches_ = []

        if self.set_carid_index and "carID" in df.columns:
            df = df.set_index("carID")

        if "Brand" in df.columns and "brand" not in df.columns:
            df = df.rename(columns={"Brand": "brand"})

        if "paintQuality%" in df.columns and "paintQuality" not in df.columns:
            df = df.rename(columns={"paintQuality%": "paintQuality"})

        # NUMERICAL COLUMNS
        def _track_new_nans(col, before, after):
            before_na = int(pd.isna(before).sum())
            after_na = int(pd.isna(after).sum())
            self.clean_report_["numeric_new_nans"][col] = max(0, after_na - before_na)

        if "year" in df.columns:
            before = df["year"].copy()
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df.loc[~df["year"].between(self.year_min, self.year_max), "year"] = np.nan
            df["year"] = np.floor(df["year"]).astype("float64")
            _track_new_nans("year", before, df["year"])

        if "mileage" in df.columns:
            before = df["mileage"].copy()
            df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
            df.loc[df["mileage"] < 0, "mileage"] = np.nan
            df["mileage"] = np.floor(df["mileage"]).astype("float64")
            _track_new_nans("mileage", before, df["mileage"])

        if "tax" in df.columns:
            before = df["tax"].copy()
            df["tax"] = pd.to_numeric(df["tax"], errors="coerce")
            df.loc[df["tax"] < 0, "tax"] = np.nan
            df["tax"] = np.floor(df["tax"]).astype("float64")
            _track_new_nans("tax", before, df["tax"])

        if "mpg" in df.columns:
            before = df["mpg"].copy()
            df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")
            df.loc[~df["mpg"].between(self.mpg_min, self.mpg_max), "mpg"] = np.nan
            df["mpg"] = np.floor(df["mpg"]).astype("float64")
            _track_new_nans("mpg", before, df["mpg"])

        if "engineSize" in df.columns:
            before = df["engineSize"].copy()
            df["engineSize"] = pd.to_numeric(df["engineSize"], errors="coerce")
            df.loc[~df["engineSize"].between(self.engine_min, self.engine_max), "engineSize"] = np.nan
            df["engineSize"] = df["engineSize"].round(1)
            _track_new_nans("engineSize", before, df["engineSize"])

        if "paintQuality" in df.columns:
            before = df["paintQuality"].copy()
            df["paintQuality"] = pd.to_numeric(df["paintQuality"], errors="coerce")
            df.loc[~df["paintQuality"].between(self.paint_min, self.paint_max), "paintQuality"] = np.nan
            df["paintQuality"] = np.floor(df["paintQuality"]).astype("float64")
            _track_new_nans("paintQuality", before, df["paintQuality"])

        if "previousOwners" in df.columns:
            before = df["previousOwners"].copy()
            df["previousOwners"] = pd.to_numeric(df["previousOwners"], errors="coerce")
            df.loc[df["previousOwners"] < 0, "previousOwners"] = np.nan
            df["previousOwners"] = np.floor(df["previousOwners"]).astype("float64")
            _track_new_nans("previousOwners", before, df["previousOwners"])

        if "hasDamage" in df.columns:
            before = df["hasDamage"].copy()
            df["hasDamage"] = pd.to_numeric(df["hasDamage"], errors="coerce").astype("float64")
            _track_new_nans("hasDamage", before, df["hasDamage"])

        # Drop paintQuality because we cannot use it for predictions (filled by mechanic)
        if "paintQuality" in df.columns:
            df = df.drop(columns=["paintQuality"])

        # CATEGORICAL COLUMNS
        raw_brand = df["brand"].copy() if "brand" in df.columns else None
        raw_trans = df["transmission"].copy() if "transmission" in df.columns else None
        raw_fuel = df["fuelType"].copy() if "fuelType" in df.columns else None

        # brand
        if "brand" in df.columns:
            before = df["brand"].copy()
            df["brand"] = self._canon_map(df["brand"], REVERSE_BRAND, keep_unmapped=False)
            changed = self._norm_str_series(before).fillna("__nan__") != self._norm_str_series(df["brand"]).fillna("__nan__")
            self.clean_report_["categorical_changes"]["brand_changed"] = int(changed.sum())

        # Store raw model token BEFORE deterministic mapping
        if "model" in df.columns:
            df["_raw_model"] = self._norm_str_series(df["model"])
        else:
            df["_raw_model"] = pd.Series([np.nan] * len(df), index=df.index)

        # special buckets for single-letter tokens (only if brand known)
        if "brand" in df.columns and "model" in df.columns:
            audi_a_mask = (df["brand"] == "Audi") & (df["_raw_model"] == "a")
            df.loc[audi_a_mask, "model"] = "a_unknown"
            if audi_a_mask.any():
                self.clean_report_["notes"].append(f"Audi model 'a' mapped to 'a_unknown' for {int(audi_a_mask.sum())} rows.")

            audi_q_mask = (df["brand"] == "Audi") & (df["_raw_model"] == "q")
            df.loc[audi_q_mask, "model"] = "q_unknown"
            if audi_q_mask.any():
                self.clean_report_["notes"].append(f"Audi model 'q' mapped to 'q_unknown' for {int(audi_q_mask.sum())} rows.")

            bmw_x_mask = (df["brand"] == "BMW") & (df["_raw_model"] == "x")
            df.loc[bmw_x_mask, "model"] = "x_unknown"
            if bmw_x_mask.any():
                self.clean_report_["notes"].append(f"BMW model 'x' mapped to 'x_unknown' for {int(bmw_x_mask.sum())} rows.")

        # model deterministic + strict vocab gating
        if "model" in df.columns:
            before = df["model"].copy()

            model_is_bucket = df["model"].astype(str).isin(["a_unknown", "q_unknown", "x_unknown"])
            mapped = self._canon_map(df["model"], REVERSE_MODEL, keep_unmapped=True)
            df.loc[~model_is_bucket, "model"] = mapped.loc[~model_is_bucket]

            if self.strict_static_model_vocab and getattr(self, "model_static_vocab_", None):
                m_norm = self._norm_str_series(df["model"])
                static_set = set(self.model_static_vocab_)
                valid_mask = m_norm.isna() | m_norm.isin(static_set) | model_is_bucket

                # never guess a/q/x if brand missing
                tok_bad = m_norm.isin(["a", "q", "x"]) & df.get("brand", pd.Series([np.nan] * len(df))).isna()
                if tok_bad.any():
                    df.loc[tok_bad, "model"] = np.nan
                    self.clean_report_["notes"].append(
                        f"Model token in {{'a','q','x'}} with missing brand forced to NaN: {int(tok_bad.sum())} rows."
                    )

                n_forced = int((~valid_mask).sum())
                if n_forced > 0:
                    df.loc[~valid_mask, "model"] = np.nan
                    self.clean_report_["notes"].append(
                        f"Model tokens not in static vocab forced to NaN: {n_forced} rows (to enable fuzzy/imputation)."
                    )

            changed = self._norm_str_series(before).fillna("__nan__") != self._norm_str_series(df["model"]).fillna("__nan__")
            self.clean_report_["categorical_changes"]["model_changed"] = int(changed.sum())

        # transmission
        if "transmission" in df.columns:
            before = df["transmission"].copy()
            df["transmission"] = self._canon_map(df["transmission"], REVERSE_TRANS, keep_unmapped=False)
            df.loc[df["transmission"] == "Unknown", "transmission"] = np.nan
            changed = self._norm_str_series(before).fillna("__nan__") != self._norm_str_series(df["transmission"]).fillna("__nan__")
            self.clean_report_["categorical_changes"]["transmission_changed"] = int(changed.sum())

        # fuelType
        if "fuelType" in df.columns:
            before = df["fuelType"].copy()
            df["fuelType"] = self._canon_map(df["fuelType"], REVERSE_FUEL, keep_unmapped=False)

            if self.handle_electric == "other":
                df.loc[df["fuelType"] == "Electric", "fuelType"] = "Other"
            elif self.handle_electric == "nan":
                df.loc[df["fuelType"] == "Electric", "fuelType"] = np.nan

            changed = self._norm_str_series(before).fillna("__nan__") != self._norm_str_series(df["fuelType"]).fillna("__nan__")
            self.clean_report_["categorical_changes"]["fuelType_changed"] = int(changed.sum())

        # FUZZY (only fills NaNs)
        if self.use_fuzzy:
            # brand/trans/fuel: only fill NaNs with canonical vocab (strict)
            if "brand" in df.columns and getattr(self, "brand_vocab_", []) and raw_brand is not None:
                miss = df["brand"].isna() & raw_brand.notna()
                for idx in df.index[miss]:
                    tok = str(raw_brand.loc[idx]).strip().lower()
                    if len(tok) < self.fuzzy_min_token_len:
                        continue
                    m = self._fuzzy_one(tok, self.brand_vocab_, cutoff=self.fuzzy_cutoff_brand)
                    if m is not None and m != tok:
                        df.loc[idx, "brand"] = BRAND_NORM_TO_CANON.get(str(m).lower(), m)
                        self.fuzzy_matches_.append({"column": "brand", "raw": tok, "match": df.loc[idx, "brand"]})

            if "transmission" in df.columns and getattr(self, "trans_vocab_", []) and raw_trans is not None:
                miss = df["transmission"].isna() & raw_trans.notna()
                for idx in df.index[miss]:
                    tok = str(raw_trans.loc[idx]).strip().lower()
                    if len(tok) < self.fuzzy_min_token_len:
                        continue
                    m = self._fuzzy_one(tok, self.trans_vocab_, cutoff=self.fuzzy_cutoff_trans)
                    if m is not None and m != tok:
                        df.loc[idx, "transmission"] = TRANS_NORM_TO_CANON.get(str(m).lower(), m)
                        self.fuzzy_matches_.append({"column": "transmission", "raw": tok, "match": df.loc[idx, "transmission"]})

            if "fuelType" in df.columns and getattr(self, "fuel_vocab_", []) and raw_fuel is not None:
                miss = df["fuelType"].isna() & raw_fuel.notna()
                for idx in df.index[miss]:
                    tok = str(raw_fuel.loc[idx]).strip().lower()
                    if len(tok) < self.fuzzy_min_token_len:
                        continue
                    m = self._fuzzy_one(tok, self.fuel_vocab_, cutoff=self.fuzzy_cutoff_fuel)
                    if m is not None and m != tok:
                        df.loc[idx, "fuelType"] = FUEL_NORM_TO_CANON.get(str(m).lower(), m)
                        self.fuzzy_matches_.append({"column": "fuelType", "raw": tok, "match": df.loc[idx, "fuelType"]})

            # model fuzzy: rescue only to static canonical keys (and preferably within-brand)
            if "model" in df.columns:
                miss_model = df["model"].isna() & df["_raw_model"].notna()
                if miss_model.any():
                    for idx in df.index[miss_model]:
                        tok = str(df.loc[idx, "_raw_model"]).strip().lower()
                        if len(tok) < self.fuzzy_min_token_len:
                            continue

                        # choices
                        if self.fuzzy_require_brand_for_model and "brand" in df.columns and pd.notna(df.loc[idx, "brand"]):
                            b = str(df.loc[idx, "brand"]).strip().lower()
                            choices = self.brand_to_models_.get(b, [])
                            if not choices:
                                choices = getattr(self, "model_static_vocab_", [])
                        else:
                            choices = getattr(self, "model_static_vocab_", [])

                        # 1) unique-prefix rescue
                        m = self._unique_prefix_match(tok, choices, self.model_prefix_min_len)

                        # 2) difflib rescue
                        if m is None:
                            m = self._fuzzy_one(tok, choices, cutoff=self.fuzzy_cutoff_model)

                        if m is not None and m != tok:
                            df.loc[idx, "model"] = m
                            self.fuzzy_matches_.append({"column": "model", "raw": tok, "match": m})

        df = df.drop(columns=["_raw_model"], errors="ignore")

        # fill missing brand from model where possible
        if "brand" in df.columns and "model" in df.columns:
            mask = df["brand"].isna() & df["model"].notna()
            df.loc[mask, "brand"] = df.loc[mask, "model"].map(MODEL_TO_BRAND)

        # sklearn compatibility
        for col in df.columns:
            if str(df[col].dtype) == "string":
                df[col] = df[col].replace({pd.NA: np.nan}).astype("object")

        if self.verbose:
            _print_section("CarDataCleaner report")

            if self.clean_report_["numeric_new_nans"]:
                num_rep = (
                    pd.DataFrame([{"column": k, "new_NaNs_created": v} for k, v in self.clean_report_["numeric_new_nans"].items()])
                    .sort_values("new_NaNs_created", ascending=False)
                )
                print("Numeric sanity checks (values set to missing because they were implausible):")
                _maybe_display(num_rep, max_rows=self.verbose_top_n)
            else:
                print("Numeric sanity checks: no new missing values were created.")

            if self.clean_report_["categorical_changes"]:
                cat_rep = (
                    pd.DataFrame([{"field": k, "n_changed": v} for k, v in self.clean_report_["categorical_changes"].items()])
                    .sort_values("n_changed", ascending=False)
                )
                print("\nCategorical corrections (typos/variants collapsed to stable labels):")
                _maybe_display(cat_rep, max_rows=self.verbose_top_n)
            else:
                print("\nCategorical corrections: no changes were made.")

            if self.clean_report_["notes"]:
                print("\nSpecial rules applied:")
                for n in self.clean_report_["notes"]:
                    print(f"- {n}")

            if self.use_fuzzy:
                if self.fuzzy_matches_:
                    fm = pd.DataFrame(self.fuzzy_matches_)
                    fm["pair"] = fm["raw"].astype(str) + "  ->  " + fm["match"].astype(str)
                    summary = (
                        fm.groupby(["column", "pair"])
                        .size()
                        .reset_index(name="count")
                        .sort_values(["column", "count"], ascending=[True, False])
                    )

                    print("\nFuzzy matches performed (raw token -> chosen match):")
                    _maybe_display(summary, max_rows=self.verbose_top_n)

                    if self.verbose_plot:
                        for col in summary["column"].unique():
                            sub = summary[summary["column"] == col].head(self.verbose_top_n)
                            plt.figure()
                            plt.title(f"Top fuzzy mappings for {col}")
                            plt.barh(sub["pair"][::-1], sub["count"][::-1])
                            plt.xlabel("Count")
                            plt.tight_layout()
                            plt.show()
                else:
                    print("\nFuzzy matching: no replacements made.")

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
        > set them to NaN
        > clip them to bounds (winsorize)

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
        exclude_id_cols=True,
        skip_discrete=True,
        discrete_unique_thresh=20,
    ):
        self.cols = cols
        self.methods = methods
        self.min_votes = min_votes
        self.iqr_k = iqr_k
        self.z_thresh = z_thresh
        self.action = action
        self.verbose = verbose
        self.exclude_id_cols = exclude_id_cols
        self.skip_discrete = skip_discrete
        self.discrete_unique_thresh = discrete_unique_thresh

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        self.feature_names_in_ = df.columns.to_list()

        # Decide which columns to handle
        if self.cols is None:
            # Only numeric columns by default
            cols_ = df.select_dtypes(include="number").columns.to_list()
            if self.exclude_id_cols:
                cols_ = [c for c in cols_ if not (str(c).lower().endswith("id") or str(c).lower() == "id")]
            self.cols_ = cols_
        else:
            self.cols_ = [c for c in self.cols if c in df.columns]

        self.stats_ = {}

        for col in self.cols_:
            s = pd.to_numeric(df[col], errors="coerce").astype(float)
            s = s.dropna()
            if s.empty:
                self.stats_[col] = {}
                continue

            # Skip very low-cardinality numeric columns (often discrete schedules like road-tax bands)
            if self.skip_discrete:
                nunq = int(s.nunique(dropna=True))
                if nunq <= int(self.discrete_unique_thresh):
                    self.stats_[col] = {}
                    continue

            col_stats = {}

            # IQR fences
            if "iqr" in self.methods:
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - self.iqr_k * iqr
                upper = q3 + self.iqr_k * iqr
                col_stats["iqr"] = {"lower": lower, "upper": upper, "q1": q1, "q3": q3, "iqr": iqr}

            # Modified Z-score (median + MAD)
            if "mod_z" in self.methods:
                med = float(s.median())
                mad = float(np.median(np.abs(s - med)))
                # Avoid division-by-zero
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

            # skipped columns have empty stats
            if not self.stats_.get(col, {}):
                self.n_outliers_by_col_[col] = 0
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

        self.report_ = (
            pd.DataFrame({"column": list(self.n_outliers_by_col_.keys()), "cells_flagged": list(self.n_outliers_by_col_.values())})
              .sort_values("cells_flagged", ascending=False)
              .reset_index(drop=True)
        )

        if self.verbose:
            _print_section("OutlierHandler report")
            print(f"Action: {self.action} (we never drop rows)")
            print(f"Total cells flagged: {self.n_outliers_total_}")
            _maybe_display(self.report_, max_rows=15)

            # Plot
            top = self.report_.head(15)
            if len(top) > 0:
                plt.figure()
                plt.title("Outliers flagged (top columns)")
                plt.barh(top["column"][::-1], top["cells_flagged"][::-1])
                plt.xlabel("Cells flagged")
                plt.tight_layout()
                plt.show()

        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)


################################################################################
# Missing Values (Imputer)
################################################################################

class IndividualHierarchyImputer(BaseEstimator, TransformerMixin):
    """
    Simple hierarchical imputer with specific rules for each column.
    
    Uses fit() to learn group statistics from training data and transform()
    to apply them, preventing data leakage in cross-validation.
    
    Imputation Hierarchy Overview:
    
    brand (mode):
        model -> (fuelType, transmission) -> fuelType -> global
    
    model (mode):
        (brand, engineSize, fuelType, transmission) -> (brand, fuelType, transmission) ->
        (brand, engineSize, transmission) -> (brand, engineSize, fuelType) -> brand -> global
    
    fuelType (mode):
        (model, tax) -> model -> brand -> global
    
    transmission (mode):
        model -> brand -> global
    
    engineSize (median):
        (model, fuelType) -> model -> fuelType -> global
    
    mpg (median):
        (model, engineSize) -> model -> engineSize -> global
    
    tax (median):
        (model, engineSize) -> model -> global
    
    previousOwners (median):
        year -> global
    
    mileage (median):
        year -> global
    
    year (mode):
        mileage_bins -> global
    
    hasDamage (mode):
        global
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def _safe_mode(series):
        """Get mode of series, return np.nan if empty."""
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else np.nan
    
    @staticmethod
    def _safe_median(series):
        """Get median of series, return np.nan if empty."""
        return series.median() if series.notna().any() else np.nan
    
    def fit(self, X, y=None):
        """Learn imputation statistics from training data."""
        df = pd.DataFrame(X).copy()
        
        # Store lookup tables
        self.brand_by_model_ = {}
        self.brand_by_fuelType_transmission_ = {}
        self.brand_by_fuelType_ = {}
        self.model_by_brand_engineSize_fuelType_transmission_ = {}
        self.model_by_brand_fuelType_transmission_ = {}
        self.model_by_brand_engineSize_transmission_ = {}
        self.model_by_brand_engineSize_ = {}
        self.model_by_brand_ = {}
        self.fuel_by_model_tax_ = {}
        self.fuel_by_model_ = {}
        self.fuel_by_brand_ = {}
        self.transmission_by_model_ = {}
        self.transmission_by_brand_ = {}
        self.mpg_by_model_engineSize_ = {}
        self.mpg_by_model_ = {}
        self.mpg_by_engineSize_ = {}
        self.tax_by_model_engineSize_ = {}
        self.tax_by_model_ = {}
        self.previousOwners_by_year_ = {}
        self.engineSize_by_model_fuelType_ = {}
        self.engineSize_by_model_ = {}
        self.engineSize_by_fuelType_ = {}
        self.mileage_by_year_ = {}
        self.year_by_mileage_bin_ = {}
        self.mileage_bins_ = None
        
        ### Learn group statistics (only from non-null values)
        # Learn mode of model for brand
        valid = df[["model", "brand"]].dropna()
        if not valid.empty:
            self.brand_by_model_ = valid.groupby("model")["brand"].agg(self._safe_mode).to_dict()
        
        # Learn mode of (fuelType, transmission) for brand (fallback)
        valid = df[["fuelType", "transmission", "brand"]].dropna()
        if not valid.empty:
            self.brand_by_fuelType_transmission_ = valid.groupby(["fuelType", "transmission"])["brand"].agg(self._safe_mode).to_dict()
        
        # Learn mode of fuelType for brand (fallback)
        valid = df[["fuelType", "brand"]].dropna()
        if not valid.empty:
            self.brand_by_fuelType_ = valid.groupby("fuelType")["brand"].agg(self._safe_mode).to_dict()
        

                    
        # Learn mode of model using different combinations (stored in separate dicts)
        # 1. (brand, engineSize, fuelType, transmission)
        valid = df[["brand", "engineSize", "fuelType", "transmission", "model"]].dropna()
        if not valid.empty:
            self.model_by_brand_engineSize_fuelType_transmission_ = valid.groupby(["brand", "engineSize", "fuelType", "transmission"])["model"].agg(self._safe_mode).to_dict()
        
        # 2. (brand, fuelType, transmission) when engineSize missing
        valid = df[["brand", "fuelType", "transmission", "model"]].dropna()
        if not valid.empty:
            self.model_by_brand_fuelType_transmission_ = valid.groupby(["brand", "fuelType", "transmission"])["model"].agg(self._safe_mode).to_dict()
        
        # 3. (brand, engineSize, transmission): when fuelType missing
        valid = df[["brand", "engineSize", "transmission", "model"]].dropna()
        if not valid.empty:
            self.model_by_brand_engineSize_transmission_ = valid.groupby(["brand", "engineSize", "transmission"])["model"].agg(self._safe_mode).to_dict()
        
        # 4. (brand, engineSize, fuelType) when transmission missing
        valid = df[["brand", "engineSize", "fuelType", "model"]].dropna()
        if not valid.empty:
            self.model_by_brand_engineSize_ = valid.groupby(["brand", "engineSize", "fuelType"])["model"].agg(self._safe_mode).to_dict()
        
        # 5. (brand) fallback to brand only
        valid = df[["brand", "model"]].dropna()
        if not valid.empty:
            self.model_by_brand_ = valid.groupby("brand")["model"].agg(self._safe_mode).to_dict()

        # Learn mode of (model, tax) for fuelType
        valid = df[["model", "tax", "fuelType"]].dropna()
        if not valid.empty:
            self.fuel_by_model_tax_ = valid.groupby(["model", "tax"])["fuelType"].agg(self._safe_mode).to_dict()
        
        # Learn mode of model for fuelType (fallback)
        valid = df[["model", "fuelType"]].dropna()
        if not valid.empty:
            self.fuel_by_model_ = valid.groupby("model")["fuelType"].agg(self._safe_mode).to_dict()
        
        # Learn mode of brand for fuelType (fallback)
        valid = df[["brand", "fuelType"]].dropna()
        if not valid.empty:
            self.fuel_by_brand_ = valid.groupby("brand")["fuelType"].agg(self._safe_mode).to_dict()
        
        # Learn mode of model for transmission
        valid = df[["model", "transmission"]].dropna()
        if not valid.empty:
            self.transmission_by_model_ = valid.groupby("model")["transmission"].agg(self._safe_mode).to_dict()
        
        # Learn mode of brand for transmission (fallback)
        valid = df[["brand", "transmission"]].dropna()
        if not valid.empty:
            self.transmission_by_brand_ = valid.groupby("brand")["transmission"].agg(self._safe_mode).to_dict()
        
        # Learn median of (model, engineSize) for mpg
        valid = df[["model", "engineSize", "mpg"]].dropna()
        if not valid.empty:
            self.mpg_by_model_engineSize_ = valid.groupby(["model", "engineSize"])["mpg"].agg(self._safe_median).to_dict()
        
        # Learn median of model for mpg (fallback)
        valid = df[["model", "mpg"]].dropna()
        if not valid.empty:
            self.mpg_by_model_ = valid.groupby("model")["mpg"].agg(self._safe_median).to_dict()
        
        # Learn median of engineSize for mpg (fallback)
        valid = df[["engineSize", "mpg"]].dropna()
        if not valid.empty:
            self.mpg_by_engineSize_ = valid.groupby("engineSize")["mpg"].agg(self._safe_median).to_dict()
        
        # Learn median of (model, engineSize) for tax
        valid = df[["model", "engineSize", "tax"]].dropna()
        if not valid.empty:
            self.tax_by_model_engineSize_ = valid.groupby(["model", "engineSize"])["tax"].agg(self._safe_median).to_dict()
        
        # Learn median of model for tax (fallback)
        valid = df[["model", "tax"]].dropna()
        if not valid.empty:
            self.tax_by_model_ = valid.groupby("model")["tax"].agg(self._safe_median).to_dict()
        
        # Learn median of year for previousOwners
        valid = df[["year", "previousOwners"]].dropna()
        if not valid.empty:
            self.previousOwners_by_year_ = valid.groupby("year")["previousOwners"].agg(self._safe_median).to_dict()
        
        # Learn median of (model, fuelType) for engineSize
        valid = df[["model", "fuelType", "engineSize"]].dropna()
        if not valid.empty:
            self.engineSize_by_model_fuelType_ = valid.groupby(["model", "fuelType"])["engineSize"].agg(self._safe_median).to_dict()
        
        # Learn median of model for engineSize (fallback)
        valid = df[["model", "engineSize"]].dropna()
        if not valid.empty:
            self.engineSize_by_model_ = valid.groupby("model")["engineSize"].agg(self._safe_median).to_dict()
        
        # Learn median of fuelType for engineSize (fallback)
        valid = df[["fuelType", "engineSize"]].dropna()
        if not valid.empty:
            self.engineSize_by_fuelType_ = valid.groupby("fuelType")["engineSize"].agg(self._safe_median).to_dict()
        
        # Learn median of year for mileage
        valid = df[["year", "mileage"]].dropna()
        if not valid.empty:
            self.mileage_by_year_ = valid.groupby("year")["mileage"].agg(self._safe_median).to_dict()
        
        # Learn median year for mileage bins (since year and mileage are highly correlated)
        valid = df[["mileage", "year"]].dropna()
        if not valid.empty and len(valid) >= 20:
            # Create mileage bins using quantiles
            self.mileage_bins_ = pd.qcut(valid["mileage"], q=10, duplicates='drop', retbins=True)[1]
            # Assign each mileage to a bin
            valid["mileage_bin"] = pd.cut(valid["mileage"], bins=self.mileage_bins_, include_lowest=True)
            # Learn median year for each bin
            self.year_by_mileage_bin_ = valid.groupby("mileage_bin", observed=False)["year"].agg(self._safe_median).to_dict()
        
        # Learn global fallbacks
        self.global_brand_ = self._safe_mode(df["brand"])
        self.global_model_ = self._safe_mode(df["model"])
        self.global_fuel_ = self._safe_mode(df["fuelType"])
        self.global_transmission_ = self._safe_mode(df["transmission"])
        self.global_mpg_ = self._safe_median(df["mpg"])
        self.global_tax_ = self._safe_median(df["tax"])
        self.global_previousOwners_ = self._safe_median(df["previousOwners"])
        self.global_engineSize_ = self._safe_median(df["engineSize"])
        self.global_mileage_ = self._safe_median(df["mileage"])
        self.global_year_ = self._safe_median(df["year"])
        self.global_hasDamage_ = self._safe_mode(df["hasDamage"])
        
        return self
    
    def transform(self, X):
        """Apply learned imputation rules."""
        df = pd.DataFrame(X).copy()
        
        # Impute brand with hierarchical lookup: model -> (fuelType, transmission) -> fuelType -> global
        mask = df["brand"].isna()
        if mask.any():
            df.loc[mask, "brand"] = df.loc[mask, "model"].map(self.brand_by_model_)
            mask = df["brand"].isna()
            
            # Fallback to (fuelType, transmission)
            if mask.any():
                keys = df.loc[mask, ["fuelType", "transmission"]].apply(tuple, axis=1)
                df.loc[mask, "brand"] = df.loc[mask, "brand"].fillna(keys.map(self.brand_by_fuelType_transmission_))
                mask = df["brand"].isna()
            
            # Fallback to fuelType only
            if mask.any():
                df.loc[mask, "brand"] = df.loc[mask, "fuelType"].map(self.brand_by_fuelType_)
        
        df["brand"] = df["brand"].fillna(self.global_brand_)
        
        # Impute model with hierarchical combination lookup, otherwise global mode
        mask = df["model"].isna()
        if mask.any():
            # Try all combinations in order of specificity (matching fit() logic)
            # 1. Try (brand, engineSize, fuelType, transmission) - 4-tuple
            keys = df.loc[mask, ["brand", "engineSize", "fuelType", "transmission"]].apply(tuple, axis=1)
            df.loc[mask, "model"] = df.loc[mask, "model"].fillna(keys.map(self.model_by_brand_engineSize_fuelType_transmission_))
            mask = df["model"].isna()
            
            # 2. Try (brand, fuelType, transmission) - 3-tuple
            if mask.any():
                keys = df.loc[mask, ["brand", "fuelType", "transmission"]].apply(tuple, axis=1)
                df.loc[mask, "model"] = df.loc[mask, "model"].fillna(keys.map(self.model_by_brand_fuelType_transmission_))
                mask = df["model"].isna()
            
            # 3. Try (brand, engineSize, transmission) - 3-tuple
            if mask.any():
                keys = df.loc[mask, ["brand", "engineSize", "transmission"]].apply(tuple, axis=1)
                df.loc[mask, "model"] = df.loc[mask, "model"].fillna(keys.map(self.model_by_brand_engineSize_transmission_))
                mask = df["model"].isna()
            
            # 4. Try (brand, engineSize, fuelType) - 3-tuple
            if mask.any():
                keys = df.loc[mask, ["brand", "engineSize", "fuelType"]].apply(tuple, axis=1)
                df.loc[mask, "model"] = df.loc[mask, "model"].fillna(keys.map(self.model_by_brand_engineSize_))
                mask = df["model"].isna()
            
            # 5. Try (brand) - single value, not tuple
            if mask.any():
                df.loc[mask, "model"] = df.loc[mask, "brand"].map(self.model_by_brand_)
        
        # Final fallback to global mode
        df["model"] = df["model"].fillna(self.global_model_)
        
        # Impute fuelType with hierarchical lookup: (model, tax) -> model -> brand -> global
        mask = df["fuelType"].isna()
        if mask.any():
            # Try (model, tax) combination first
            keys = df.loc[mask, ["model", "tax"]].apply(tuple, axis=1)
            df.loc[mask, "fuelType"] = df.loc[mask, "fuelType"].fillna(keys.map(self.fuel_by_model_tax_))
            mask = df["fuelType"].isna()
            
            # Fallback to model only
            if mask.any():
                df.loc[mask, "fuelType"] = df.loc[mask, "model"].map(self.fuel_by_model_)
                mask = df["fuelType"].isna()
            
            # Fallback to brand only
            if mask.any():
                df.loc[mask, "fuelType"] = df.loc[mask, "brand"].map(self.fuel_by_brand_)
        
        df["fuelType"] = df["fuelType"].fillna(self.global_fuel_)
        
        # Impute transmission with mode of model, then brand, otherwise global mode
        mask = df["transmission"].isna()
        if mask.any():
            df.loc[mask, "transmission"] = df.loc[mask, "model"].map(self.transmission_by_model_)
            mask = df["transmission"].isna()
            
            # Fallback to brand
            if mask.any():
                df.loc[mask, "transmission"] = df.loc[mask, "brand"].map(self.transmission_by_brand_)
        
        df["transmission"] = df["transmission"].fillna(self.global_transmission_)
        
        # Impute mpg with hierarchical lookup: (model, engineSize) -> model -> engineSize -> global
        mask = df["mpg"].isna()
        if mask.any():
            # Try (model, engineSize) combination first
            keys = df.loc[mask, ["model", "engineSize"]].apply(tuple, axis=1)
            df.loc[mask, "mpg"] = df.loc[mask, "mpg"].fillna(keys.map(self.mpg_by_model_engineSize_))
            mask = df["mpg"].isna()
            
            # Fallback to model only
            if mask.any():
                df.loc[mask, "mpg"] = df.loc[mask, "model"].map(self.mpg_by_model_)
                mask = df["mpg"].isna()
            
            # Fallback to engineSize only
            if mask.any():
                df.loc[mask, "mpg"] = df.loc[mask, "engineSize"].map(self.mpg_by_engineSize_)
        
        df["mpg"] = df["mpg"].fillna(self.global_mpg_)
        
        # Impute tax with hierarchical lookup: (model, engineSize) -> model -> global
        mask = df["tax"].isna()
        if mask.any():
            # Try (model, engineSize) combination first
            keys = df.loc[mask, ["model", "engineSize"]].apply(tuple, axis=1)
            df.loc[mask, "tax"] = df.loc[mask, "tax"].fillna(keys.map(self.tax_by_model_engineSize_))
            mask = df["tax"].isna()
            
            # Fallback to model
            if mask.any():
                df.loc[mask, "tax"] = df.loc[mask, "model"].map(self.tax_by_model_)
        
        df["tax"] = df["tax"].fillna(self.global_tax_)
        
        # Impute previousOwners with median of year, otherwise global median
        mask = df["previousOwners"].isna()
        if mask.any():
            df.loc[mask, "previousOwners"] = df.loc[mask, "year"].map(self.previousOwners_by_year_)
        df["previousOwners"] = df["previousOwners"].fillna(self.global_previousOwners_)
        
        # Impute engineSize with hierarchical lookup: (model, fuelType) -> model → fuelType → global
        mask = df["engineSize"].isna()
        if mask.any():
            # Try (model, fuelType) combination first
            keys = df.loc[mask, ["model", "fuelType"]].apply(tuple, axis=1)
            df.loc[mask, "engineSize"] = df.loc[mask, "engineSize"].fillna(keys.map(self.engineSize_by_model_fuelType_))
            mask = df["engineSize"].isna()
            
            # Fallback to model only
            if mask.any():
                df.loc[mask, "engineSize"] = df.loc[mask, "model"].map(self.engineSize_by_model_)
                mask = df["engineSize"].isna()
            
            # Fallback to fuelType only
            if mask.any():
                df.loc[mask, "engineSize"] = df.loc[mask, "fuelType"].map(self.engineSize_by_fuelType_)
        
        df["engineSize"] = df["engineSize"].fillna(self.global_engineSize_)
        
        # Impute mileage with median of year, otherwise global median
        mask = df["mileage"].isna()
        if mask.any():
            df.loc[mask, "mileage"] = df.loc[mask, "year"].map(self.mileage_by_year_)
        df["mileage"] = df["mileage"].fillna(self.global_mileage_)
        
        # Impute year using mileage bins (since they're highly correlated), otherwise global median
        mask = df["year"].isna()
        if mask.any() and self.mileage_bins_ is not None:
            # Assign each mileage to the learned bins
            mileage_bins = pd.cut(df.loc[mask, "mileage"], bins=self.mileage_bins_, include_lowest=True)
            df.loc[mask, "year"] = mileage_bins.map(self.year_by_mileage_bin_)
        df["year"] = df["year"].fillna(self.global_year_)
        
        # Impute hasDamage with global mode
        df["hasDamage"] = df["hasDamage"].fillna(self.global_hasDamage_)
        
        return df


################################################################################
# Feature Engineering (CarFeatureEngineer)
################################################################################

class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates domain-informed numeric features inside the pipeline.

    Why fit() exists:
    - We learn fold-specific reference statistics (e.g., average age per brand)
      using only the training fold. This prevents leakage in cross-validation.

    Note about `y` in fit():
    - sklearn pipelines call fit(X, y) on every step. Even if we don't use y,
      the signature must accept it.
    """

    def __init__(self, ref_year=None, verbose=False, verbose_n_rows=5):
        self.ref_year = ref_year
        self.verbose = verbose
        self.verbose_n_rows = verbose_n_rows

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
        before_cols = set(X.columns)

        # 1. Base Feature Creation:
        #       - Car Age - Newer cars usually have higher prices, models prefer linear features
        age = self.ref_year_ - X["year"]
        X["age"] = age

        X["mpg_x_engine"] = X["mpg"] * X["engineSize"]

        # Add 1 to age because if age is 0 (this year) the value would be lost otherwise
        X["engine_x_age"] = X["engineSize"] * (X["age"] + 1)  # Highlight the aspect of old cars with big engines for that time which were very valuable and might therefore still be valuable

        X["mpg_x_age"] = X["mpg"] * (X["age"] + 1)  # Older cars tend to have higher MPG -> amplify effect
        X["tax_x_age"] = X["tax"] * (X["age"] + 1)

        ###### Division: The Normalizer (create ratios, rates, or efficiency metrics: "How much of A do we have per unit of B?") -> removes the influence of the divisor

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

        if self.verbose:
            _print_section("CarFeatureEngineer report")
            new_cols = sorted(list(set(X.columns) - before_cols))
            print(f"New features created: {len(new_cols)}")
            print(", ".join(new_cols))
            print("\nExample rows (only the new features):")
            _maybe_display(X[new_cols].head(self.verbose_n_rows), max_rows=self.verbose_n_rows)
            print("\nNote: the returned dataframe contains ALL original columns plus the new features above.")

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

    def __init__(self, selectors=None, min_votes=2, verbose=False):
        """
        args:
            selectors: list of sklearn feature selector objects.
            min_votes: int, minimum number of selectors that must agree to keep a feature.
        """
        self.selectors = selectors
        self.min_votes = min_votes
        self.verbose = verbose
        self.fitted_selectors_ = []
        self.support_mask_ = None
        self.feature_names_in_ = None
        self.votes_ = None

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
            votes += sel.get_support().astype(int)

        # Create the final mask: True if votes >= threshold
        self.votes_ = votes.copy()
        self.support_mask_ = votes >= self.min_votes

        if self.verbose:
            _print_section("MajorityVoteSelectorTransformer report")
            n_total = X.shape[1]
            n_keep = int(self.support_mask_.sum())
            print(f"Features kept: {n_keep}/{n_total} (min_votes={self.min_votes})")

            # simple visualization: how many features got 0/1/2/.. votes
            vc = pd.Series(votes).value_counts().sort_index()
            rep = pd.DataFrame({"votes": vc.index, "n_features": vc.values})
            _maybe_display(rep, max_rows=20)

            plt.figure()
            plt.title("Vote distribution (how many selectors agreed)")
            plt.bar(rep["votes"].astype(str), rep["n_features"])
            plt.xlabel("Votes")
            plt.ylabel("Number of features")
            plt.tight_layout()
            plt.show()

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

    def __init__(
        self,
        relevance_threshold=0.1,
        redundancy_threshold=0.85,
        verbose=False,
        verbose_top_n=15,
        verbose_plot=True,
    ):
        self.relevance_threshold = relevance_threshold
        self.redundancy_threshold = redundancy_threshold
        self.verbose = verbose
        self.verbose_top_n = verbose_top_n
        self.verbose_plot = verbose_plot

        self.selected_indices_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        # 1. Input Validation and Feature Name Capture
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array([str(c) for c in X.columns])

        # Convert to Numpy for speed, ensure y is correct shape
        X_arr, y_arr = check_X_y(X, y, dtype=None)
        n_features = X_arr.shape[1]

        # 1) Relevance Filtering (Filter out weak features first)
        relevance_scores = []
        for i in range(n_features):
            # Calculate spearman with target
            corr, _ = spearmanr(X_arr[:, i], y_arr)
            relevance_scores.append(abs(corr))

        self.relevance_scores_ = np.array(relevance_scores)
        relevance_indices = np.where(self.relevance_scores_ > self.relevance_threshold)[0]
        self.relevance_pass_indices_ = relevance_indices

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

        # Verbose report
        if self.verbose:
            _print_section("SpearmanRelevancyRedundancySelector report")

            total = n_features
            passed = len(self.relevance_pass_indices_)
            kept = len(self.selected_indices_)

            print(f"Thresholds: relevance >= {self.relevance_threshold} | redundancy < {self.redundancy_threshold}")
            print(f"Total features: {total}")
            print(f"Passed relevance filter: {passed}")
            print(f"Kept after redundancy pruning: {kept}")

            # Build a readable table for top features by relevance
            names = (
                self.feature_names_in_
                if self.feature_names_in_ is not None
                else np.array([f"x{i}" for i in range(n_features)])
            )

            mask = self._get_support_mask()

            rep = pd.DataFrame(
                {
                    "feature": names,
                    "abs_spearman_with_target": self.relevance_scores_,
                    "selected": mask,
                }
            ).sort_values("abs_spearman_with_target", ascending=False)

            print("\nTop features by relevance (highest |Spearman| first):")
            _maybe_display(rep.head(self.verbose_top_n), max_rows=self.verbose_top_n)

            if self.verbose_plot:
                top = rep.head(self.verbose_top_n)
                if len(top) > 0:
                    plt.figure()
                    plt.title("Top feature relevance (|Spearman with target|)")
                    plt.barh(top["feature"][::-1], top["abs_spearman_with_target"][::-1])
                    plt.xlabel("|Spearman correlation|")
                    plt.tight_layout()
                    plt.show()

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
    def __init__(
        self,
        threshold=0.01,
        n_neighbors=10,
        random_state=42,
        verbose=False,
        verbose_top_n=15,
        verbose_plot=True,
    ):
        """
        threshold: Minimum MI score required to keep a feature.
        n_neighbors: Parameter for the internal MI calculation.
        """
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_top_n = verbose_top_n
        self.verbose_plot = verbose_plot

        self.mask_ = None
        self.feature_names_in_ = None

    def _encode_if_needed(self, X):
        """
        MI needs numeric input. If X contains non-numeric columns, we one-hot encode (pandas get_dummies)
        and remember the resulting columns for consistent transform().

        This makes your quick notebook snippet work directly on X_fe (which still has strings).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array([str(c) for c in X.columns])

            # if already all numeric, keep as-is
            if all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
                X_num = X.copy()
                self.enc_columns_ = list(X_num.columns)
                self.discrete_mask_ = np.array([False] * X_num.shape[1])
                return X_num

            # else: encode categoricals
            cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=True)

            self.enc_columns_ = list(X_enc.columns)

            # discrete mask: columns created from categoricals are discrete; numeric originals treated as continuous
            discrete_mask = np.array([False] * X_enc.shape[1])
            for c in cat_cols:
                prefix = f"{c}_"
                for j, colname in enumerate(self.enc_columns_):
                    if colname.startswith(prefix):
                        discrete_mask[j] = True
            self.discrete_mask_ = discrete_mask
            return X_enc

        # numpy array input (assume already numeric)
        self.feature_names_in_ = None
        self.enc_columns_ = None
        self.discrete_mask_ = None
        return X

    def fit(self, X, y):
        # Encode if needed (so your direct notebook snippet works)
        X_enc = self._encode_if_needed(X)

        # Calculate Mutual Information Scores
        if isinstance(X_enc, pd.DataFrame):
            X_mat = X_enc.values
            feature_names = np.array(self.enc_columns_, dtype=object)
        else:
            X_mat = X_enc
            feature_names = None

        # discrete_features only if we computed it (mixed data)
        if getattr(self, "discrete_mask_", None) is not None:
            self.scores_ = mutual_info_regression(
                X_mat,
                y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                discrete_features=self.discrete_mask_,
            )
        else:
            self.scores_ = mutual_info_regression(
                X_mat,
                y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )

        # 3. Create Mask based on Threshold
        self.mask_ = self.scores_ > self.threshold

        # store names for downstream / verbose
        self.feature_names_enc_ = feature_names

        # Verbose report
        if self.verbose:
            _print_section("MutualInfoThresholdSelector report")

            n_total = len(self.scores_)
            n_kept = int(np.sum(self.mask_))

            print(f"Threshold: MI >= {self.threshold}")
            print(f"Total features: {n_total}")
            print(f"Kept features : {n_kept}")

            names = (
                self.feature_names_enc_
                if self.feature_names_enc_ is not None
                else np.array([f"x{i}" for i in range(n_total)])
            )

            rep = pd.DataFrame(
                {
                    "feature": names,
                    "mutual_information": self.scores_,
                    "selected": self.mask_,
                }
            ).sort_values("mutual_information", ascending=False)

            print("\nTop features by mutual information:")
            _maybe_display(rep.head(self.verbose_top_n), max_rows=self.verbose_top_n)

            if self.verbose_plot:
                top = rep.head(self.verbose_top_n)
                if len(top) > 0:
                    plt.figure()
                    plt.title("Top feature importance (Mutual Information)")
                    plt.barh(top["feature"][::-1], top["mutual_information"][::-1])
                    plt.xlabel("Mutual information")
                    plt.tight_layout()
                    plt.show()

        return self

    def transform(self, X):
        if self.mask_ is None:
            raise NotFittedError("Selector not fitted.")

        # Transform must mirror fit encoding
        if isinstance(X, pd.DataFrame):
            # if fit used encoding (enc_columns_ exists), apply it; else assume numeric DF
            if getattr(self, "enc_columns_", None) is not None:
                # encode using same logic as fit (same cat columns inference)
                cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
                X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=True)

                # align to training columns
                X_enc = X_enc.reindex(columns=self.enc_columns_, fill_value=0)

                return X_enc.loc[:, self.mask_]

            # numeric-only DF path
            return X.loc[:, self.mask_]

        # numpy path
        return X[:, self.mask_]

    def get_support(self):
        return self.mask_

    def get_feature_names_out(self, input_features=None):
        if getattr(self, "feature_names_enc_", None) is not None:
            return np.array(self.feature_names_enc_, dtype=object)[self.mask_]
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.mask_))], dtype=object)[self.mask_]
        return np.array(input_features, dtype=object)[self.mask_]



################################################################################
# Model Setup and Evaluation
################################################################################


def create_model_pipe(prepro_pipe, model):
    # Clone preprocessor and model just to ensure that no state is shared between different model pipelines
    prepro_pipe_clone = clone(prepro_pipe)
    model_clone = clone(model)
    model_pipe = Pipeline([
        ("preprocess", prepro_pipe_clone),
        ("model", model_clone),
    ])
    return model_pipe


def run_quick_randomsearch(models_dict, X_train, y_train, cv, n_iter, random_state, n_jobs):
    """
    Run RandomizedSearchCV on multiple models using the model_hyperparameter_tuning function.
    
    Parameters:
    models_dict : dict
        Dictionary with model names as keys and (pipeline, param_grid) tuples as values
    X_train, y_train : arrays
        Training data
    cv : int
        Number of cross-validation folds
    n_iter : int
        Number of random iterations per model
    random_state : int
        Random state for reproducibility
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    results_df : DataFrame
        Sorted results with best scores and parameters
    best_estimators : dict
        Dictionary of best fitted estimators for each model
    final_table : DataFrame
        Formatted table
    """
    
    results = []
    best_estimators = {}
    
    for name, (pipeline, param_dist) in models_dict.items():
        print(f"Running RandomizedSearchCV for: {name}")
        
        start_time = time.time()
        
        # Skip randomsearch if no parameters to tune (e.g. Median and Linear Regression)
        if not param_dist:
            print(f"No hyperparameters to tune: fit with default parameters...")
            
            # Use cross_validate to get multiple metrics
            cv_results = cross_validate(
                pipeline, X_train, y_train, 
                cv=cv,
                scoring={
                    'mae': 'neg_mean_absolute_error',
                    'rmse': 'neg_root_mean_squared_error',
                    'r2': 'r2'
                },
                n_jobs=n_jobs,
                return_train_score=True
            )
            
            # Get the metrics
            train_mae = -cv_results['train_mae'].mean()
            train_rmse = -cv_results['train_rmse'].mean()
            train_r2 = cv_results['train_r2'].mean()
            val_mae = -cv_results['test_mae'].mean()
            val_rmse = -cv_results['test_rmse'].mean()
            val_r2 = cv_results['test_r2'].mean()
            
            best_params = "No hyperparameters"
            best_estimators[name] = pipeline
            
            scores_dict = {
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            }
            
        else:
            print(f"Running hyperparameter tuning with {n_iter} iterations...")
            
            tuned_pipe, random_search_object, scores_dict = model_hyperparameter_tuning(
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                param_dist=param_dist,
                n_iter=n_iter,
                cv=cv,
                n_jobs=-1,
                random_state=random_state,
                verbose=1
            )
            
            best_params = random_search_object.best_params_
            best_estimators[name] = tuned_pipe
        
        duration = time.time() - start_time
        
        results.append({
            'Model': name,
            'train_mae': scores_dict['train_mae'],
            'train_rmse': scores_dict['train_rmse'],
            'train_r2': scores_dict['train_r2'],
            'val_mae': scores_dict['val_mae'],
            'val_rmse': scores_dict['val_rmse'],
            'val_r2': scores_dict['val_r2'],
            'Best_Params': str(best_params),
            'Duration_mins': duration / 60
        })
        
        print(f"Duration: {duration/60:.2f} minutes\n")
        print("-"*70)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_mae', ascending=True).reset_index(drop=True)
    final_table = create_final_results_table(results_df)
    
    return results_df, best_estimators, final_table


def create_final_results_table(results_df):
    """
    Create a clean results table from raw results. Separates optimized and original preprocessing.
    """
    # Separate optimized and original preprocessing
    optimized_models = results_df[results_df['Model'].str.contains('adjusted')].copy()
    original_models = results_df[results_df['Model'].str.contains('orig')].copy()
    
    # Clean up model names
    # if not optimized_models.empty: # TODO remove
    optimized_models['model'] = optimized_models['Model'].str.replace('_adjusted', '')
    optimized_models['preprocessing'] = 'optimized'
    
    # if not original_models.empty: # TODO remove
    original_models['model'] = original_models['Model'].str.replace('_orig', '')
    original_models['preprocessing'] = 'original'
    
    # Combine tables
    tables_to_concat = []
    tables_to_concat.append(optimized_models)
    tables_to_concat.append(original_models)
    
    final_table = pd.concat([
        df[['model', 'preprocessing', 'val_mae', 'val_rmse', 'val_r2', 
            'train_mae', 'train_rmse', 'train_r2']]
        for df in tables_to_concat
    ], ignore_index=True)
    
    # Rename columns for better readability of metrics
    final_table.columns = [
        'model', 'preprocessing',
        'val_MAE', 'val_RMSE', 'val_R2',
        'train_MAE', 'train_RMSE', 'train_R2'
    ]
    
    # Round values
    numeric_cols = ['val_MAE', 'val_RMSE', 'val_R2', 'train_MAE', 'train_RMSE', 'train_R2']
    final_table[numeric_cols] = final_table[numeric_cols].round(4)
    
    # Sort by val_MAE (optimized first, then original)
    optimized_sorted = final_table[final_table['preprocessing'] == 'optimized'].sort_values('val_MAE')
    original_sorted = final_table[final_table['preprocessing'] == 'original'].sort_values('val_MAE')
    
    final_table = pd.concat([optimized_sorted, original_sorted], ignore_index=True)
    
    return final_table


################################################################################
# Hyperparameter tuning
################################################################################

def model_hyperparameter_tuning(
    X_train,
    y_train,
    cv,
    pipeline,
    param_dist,
    n_iter=100,
    verbose=3,
    verbose_features=None,         
    verbose_metric="mae",          
    verbose_plot=True,
    verbose_top_n=20,
    n_jobs=-1,                
    random_state=42,
):
    """
    RandomizedSearchCV helper + optional verbose plots (one plot per requested feature spec).

    Plot behavior:
    - If a spec has 1 param: x = param, y = validation metric (mean over duplicates).
    - If a spec has 2 params: x = first param, separate line per second param value.
    """

    if verbose_features is None:
        verbose_features = []

    model_random = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring={"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error", "r2": "r2"},
        refit="mae",
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch="2*n_jobs", # Default
        random_state=random_state,
        verbose=verbose,
        return_train_score=True,
        error_score="raise",
    )
    model_random.fit(X_train, y_train)

    # summary metrics
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

    # verbose plots
    if verbose_plot and verbose_features:
        results_df = pd.DataFrame(model_random.cv_results_)
        params_df = pd.json_normalize(results_df["params"])
        results_df = pd.concat([results_df.drop(columns=["params"]), params_df], axis=1)

        if verbose_metric == "mae":
            results_df["_val_"] = -results_df["mean_test_mae"]
        elif verbose_metric == "mse":
            results_df["_val_"] = -results_df["mean_test_mse"]
        elif verbose_metric == "r2":
            results_df["_val_"] = results_df["mean_test_r2"]
        else:
            raise ValueError("verbose_metric must be one of {'mae','mse','r2'}")

        def resolve_param(name: str) -> str:
            if name in results_df.columns:
                return name
            matches = [c for c in results_df.columns if c.endswith(f"__{name}")]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous param '{name}'. Use the full name. Matches: {matches}")
            raise ValueError(f"Param '{name}' not found in cv_results_ columns.")

        # normalize verbose_features into a list of "specs", where each spec is [p1] or [p1,p2]
        #   - if user gives ["p1","p2"] (flat list), treat it as one 2D spec
        #   - if user gives [["p1"],["p3"],["p4","p5"]] -> multiple plots
        if len(verbose_features) > 0 and isinstance(verbose_features[0], (list, tuple)):
            specs = [list(s) for s in verbose_features]
        else:
            specs = [list(verbose_features)]  # one spec

        # top trials table (always show params used in the FIRST spec)
        first_spec = specs[0]
        cols_for_table = ["rank_test_mae", "mean_test_mae", "mean_test_mse", "mean_test_r2"]
        cols_for_table = [c for c in cols_for_table if c in results_df.columns]
        for p in first_spec[:2]:
            cols_for_table.append(resolve_param(p))
        cols_for_table = list(dict.fromkeys(cols_for_table))

        _print_section(f"RandomizedSearchCV verbose (top {verbose_top_n} by MAE)")
        _maybe_display(
            results_df.sort_values("rank_test_mae")[cols_for_table].head(verbose_top_n),
            max_rows=verbose_top_n,
        )

        # plotting loop: one plot per spec
        for spec in specs:
            if not (1 <= len(spec) <= 2):
                raise ValueError(f"Each verbose feature spec must have 1 or 2 params. Got: {spec}")

            p1 = resolve_param(spec[0])
            p2 = resolve_param(spec[1]) if len(spec) == 2 else None

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            if p2 is None:
                plot_df = results_df[[p1, "_val_"]].dropna()
                plot_df = plot_df.groupby(p1, as_index=False)["_val_"].mean().sort_values(p1)
                ax.plot(plot_df[p1], plot_df["_val_"], marker="o", linewidth=2, markersize=6)
                ax.set_xlabel(p1)
            else:
                plot_df = results_df[[p1, p2, "_val_"]].dropna()
                for v, g in plot_df.groupby(p2):
                    gg = g.groupby(p1, as_index=False)["_val_"].mean().sort_values(p1)
                    ax.plot(gg[p1], gg["_val_"], marker="o", linewidth=2, markersize=6, label=f"{p2}={v}")
                ax.set_xlabel(p1)
                ax.legend(fontsize=10)

            ax.set_ylabel(f"Validation {verbose_metric.upper()}")
            ax.set_title(f"Randomized Search: {p1}" + (f" vs {p2}" if p2 else "") + f" on {verbose_metric.upper()}",
                         fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    return model_random.best_estimator_, model_random, model_scores


################################################################################
# Debug Transformer
################################################################################

class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    Prints shape/type and optionally shows a small preview of the data flowing through the pipeline
    -> make the pipeline "transparent" during development

    Safe defaults:
    - show_data=False
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
                display(X.info())
                if self.y_data_profiling:
                    if ProfileReport is None:
                        print("\nProfileReport not available (ydata_profiling not installed).")
                    else:
                        print("\nGenerating data profiling report...")
                        profile = ProfileReport(
                            X,
                            title="Car Data Profiling Report",
                            correlations={
                                "pearson": {"calculate": False},
                                "spearman": {"calculate": True},
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
# Output compatible wrapper
################################################################################
class SetOutputCompatibleWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper that adds set_output compatibility to transformers that don't support it.
    
    This class was created for the category_encoders transformers NestedCVWrapper
    which doesn't implement sklearn's set_output API introduced in version 1.2+.

    This class was necessary to get the feature names for the DebugTransformer.
    
    Parameters
    ----------
    transformer : estimator
        The transformer to wrap. Must implement fit(), transform(), and get_params().
    """
    
    def __init__(self, transformer):
        self.transformer = transformer
        self._sklearn_output_config = {}
    
    def fit(self, X, y=None, **fit_params):
        """
        Fit the wrapped transformer.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Training data.
        y : array-like, optional
            Target values.
        **fit_params : dict
            Additional fit parameters.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.transformer.fit(X, y, **fit_params)
        return self
    
    def transform(self, X):
        """
        Transform the data using the wrapped transformer.
        
        Automatically converts output to pandas DataFrame if set_output
        was configured with transform="pandas".
        
        Parameters
        ----------
        X : array-like or DataFrame
            Data to transform.
            
        Returns
        -------
        X_transformed : array-like or DataFrame
            Transformed data. Type depends on set_output configuration.
        """
        result = self.transformer.transform(X)
        
        # If pandas output is requested and result is numpy, convert it
        if self._sklearn_output_config.get("transform") == "pandas" and not isinstance(result, pd.DataFrame):
            if isinstance(X, pd.DataFrame):
                # Preserve index and use original column names if possible
                result = pd.DataFrame(result, index=X.index, columns=X.columns)
        
        return result
    
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit and transform in one step.
        """
        return self.fit(X, y, **fit_params).transform(X)
    
    def set_output(self, *, transform=None):
        """
        Set output container format.
        
        Parameters
        ----------
        transform : {"default", "pandas"}, default=None
            Configure output of transform and fit_transform.
            - "default": Default output format of the transformer
            - "pandas": DataFrame output
            - None: Transform configuration is unchanged
            
        Returns
        -------
        self : object
            Returns self.
        """
        self._sklearn_output_config["transform"] = transform
        return self
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        if deep:
            return {"transformer": self.transformer, **self.transformer.get_params(deep=True)}
        return {"transformer": self.transformer}
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Returns self.
        """
        transformer_params = {k: v for k, v in params.items() if k != "transformer"}
        if transformer_params:
            self.transformer.set_params(**transformer_params)
        if "transformer" in params:
            self.transformer = params["transformer"]
        return self