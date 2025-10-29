import pandas as pd
import numpy as np

def clean_car_dataframe(df):

    ############################################################################################################## 
    # Numerical columns:
    ##############################################################################################################

    # column carID (set as index, has no duplicates)
    df = df.set_index('carID')


    # column year: 1970 to 2024 
    # (outlier handling: set between 1970 and 2025, otherwise null; float values not needed, so convert to Int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df.loc[~df['year'].between(1970, 2025), 'year'] = np.nan
    df['year'] = np.floor(df['year']).astype('Int64')


    # column mileage: -58.000 to 323.000 
    # (outlier handling: set negative mileage null; max mileage 323.000 is realistic; float values not needed, so convert to Int)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df.loc[df['mileage'] < 0, 'mileage'] = np.nan
    df['mileage'] = np.floor(df['mileage']).astype('Int64')


    # column tax: -91 to 580 
    # (outlier handling: set negative tax null; max tax 580 is maybe realistic; float values not needed, so convert to Int)
    df['tax'] = pd.to_numeric(df['tax'], errors='coerce')
    df.loc[df['tax'] < 0, 'tax'] = np.nan
    df['tax'] = np.floor(df['tax']).astype('Int64')


    # column mpg: -43 to 470 
    # (outlier handling: mpg is usually 10–120, so set <5 and >150 null; float values not needed, so convert to Int)
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')
    df.loc[~df['mpg'].between(5, 150), 'mpg'] = np.nan
    df['mpg'] = np.floor(df['mpg']).astype('Int64')


    # column engineSize: -0.1 to 6.6 
    # (outlier handling: engineSize is usually 1.4–6.3 but 6.6 possible; set <0.6 null, max to 9)
    df['engineSize'] = pd.to_numeric(df['engineSize'], errors='coerce')
    df.loc[~df['engineSize'].between(0.6, 9.0), 'engineSize'] = np.nan
    df['engineSize'] = df['engineSize'].round(1)


    # column paintQuality%: 70–100, few outliers 1.6 or 125 
    # (outlier handling: set <70 null, max 100; float values not needed, so convert to Int)
    df = df.rename(columns={'paintQuality%': 'paintQuality'})
    df['paintQuality'] = pd.to_numeric(df['paintQuality'], errors='coerce')
    df.loc[~df['paintQuality'].between(70, 100), 'paintQuality'] = np.nan
    df['paintQuality'] = np.floor(df['paintQuality']).astype('Int64')


    # column previousOwners: -2.3 to 6.2 
    # (outlier handling: set negative null; float values not needed, so convert to Int)
    df['previousOwners'] = pd.to_numeric(df['previousOwners'], errors='coerce')
    df.loc[df['previousOwners'] < 0, 'previousOwners'] = np.nan
    df['previousOwners'] = np.floor(df['previousOwners']).astype('Int64')


    # column hasDamage (0/nan, not sure if nan means damaged, convert to Int)
    df['hasDamage'] = pd.to_numeric(df['hasDamage'], errors='coerce').astype('Int64')


    ############################################################################################################## 
    # Categorical columns:
    ##############################################################################################################

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


    # build model -> brand mapping: there are rows, where column model is filled, but brand is not. a back mapping is possible
    model_to_brand = {}
    for brand, models in {
        'VW': ['golf', 'passat', 'polo', 'tiguan', 'touran', 'up', 'sharan', 'scirocco', 'amarok', 'arteon', 'beetle'],
        'Toyota': ['yaris', 'corolla', 'aygo', 'rav4', 'auris', 'avensis', 'c-hr', 'verso', 'hilux', 'land cruiser'],
        'Audi': ['a1', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'q2', 'q3', 'q5', 'q7', 'q8', 'tt', 'r8'],
        'Ford': ['fiesta', 'focus', 'mondeo', 'kuga', 'ecosport', 'puma', 'edge', 's-max', 'c-max', 'b-max', 'ka+'],
        'BMW': ['1 series', '2 series', '3 series', '4 series', '5 series', '6 series', '7 series', '8 series',
                'x1','x2','x3','x4','x5','x6','x7','z3','z4','m3','m4','m5','m6'],
        'Skoda': ['fabia','octavia','superb','scala','karoq','kodiaq','kamiq','yeti'],
        'Opel': ['astra','corsa','insignia','mokka','zafira','meriva','adam','vectra','antara','combo life','grandland x','crossland x'],
        'Mercedes': ['a class','b class','c class','e class','s class','glc class','gle class','gla class',
                    'cls class','glb class','gls class','m class','sl class','cl class','v class','x-class','g class'],
        'Hyundai': ['i10','i20','i30','i40','ioniq','ix20','ix35','kona','tucson','santa fe']
    }.items():
        for m in models:
            model_to_brand[m] = brand

    # fill missing brand from model
    df.loc[df['Brand'].isna() & df['model'].notna(), 'Brand'] = \
        df.loc[df['Brand'].isna() & df['model'].notna(), 'model'].map(model_to_brand)

    return df
