import streamlit as st
import pandas as pd
import joblib
import ydata_profiling # import required for deployment because it's included in the pipeline

# BRAND → MODEL MAPPING

BRAND_MODELS = {
    "Audi": ["a1", "a3", "a4", "a5", "a6", "a7", "a8", "q2", "q3", "q5", "q7", "q8", "tt", "r8", "Other"],
    "BMW": [
        "1 series", "2 series", "3 series", "4 series", "5 series", "6 series", "7 series", "8 series",
        "x1", "x2", "x3", "x4", "x5", "x6", "x7", "z3", "z4", "m3", "m4", "m5", "m6", "Other"
    ],
    "Ford": ["fiesta", "focus", "mondeo", "kuga", "ecosport", "puma", "edge", "s-max", "c-max", "b-max", "ka+", "Other"],
    "Hyundai": ["i10", "i20", "i30", "i40", "ioniq", "ix20", "ix35", "kona", "tucson", "santa fe", "Other"],
    "Mercedes": [
        "a class", "b class", "c class", "e class", "s class", "glc class", "gle class", "gla class",
        "cls class", "glb class", "gls class", "m class", "sl class", "cl class", "v class", "x-class", "g class", "Other"
    ],
    "Opel": [
        "astra", "corsa", "insignia", "mokka", "zafira", "meriva", "adam", "vectra",
        "antara", "combo life", "grandland x", "crossland x", "Other"
    ],
    "Skoda": ["fabia", "octavia", "superb", "scala", "karoq", "kodiaq", "kamiq", "yeti", "Other"],
    "Toyota": ["yaris", "corolla", "aygo", "rav4", "auris", "avensis", "c-hr", "verso", "hilux", "land cruiser", "Other"],
    "VW": ["golf", "passat", "polo", "tiguan", "touran", "up", "sharan", "scirocco", "amarok", "arteon", "beetle", "Other"],
    "Other":["Other"]
}

# 1) PAGE CONFIG
st.set_page_config(
    page_title="Cars 4 You - Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2) MODEL LOADING
@st.cache_resource
def load_model():
    """Load the trained model pipeline from disk."""
    model = 'rf_tuned_pipe'
    try:
        loaded_pipe = joblib.load(f"{model}.pkl")
        
        # # Extract pipeline
        pipeline = loaded_pipe
        
        if not hasattr(pipeline, 'predict'):
            raise ValueError(f"Loaded object doesn't have predict() method. Type: {type(pipeline)}")
        
        return pipeline
        
    except FileNotFoundError:
        st.error("Model file not found")
        st.info(f"Make sure '{model}.pkl' is in the same directory as 'app.py'")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# 3) HEADER

st.title("Cars 4 You - Price Prediction")
st.markdown("""
Enter the car details below to get an AI-powered price estimate based on our machine learning model 
trained on 75,969 vehicles.
""")

st.divider()

# 4) INPUT SECTION

st.subheader("Car Details")

# Row 1: Brand & Model (DYNAMIC DROPDOWN)
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        options=sorted(BRAND_MODELS.keys()),
        index=sorted(BRAND_MODELS.keys()).index("VW"),
        help="Select the car manufacturer",
        key="brand_select"
    )

with col2:
    # Get available models for selected brand
    available_models = BRAND_MODELS[brand]
    
    # Create display names (Title Case)
    model_options = [m.title() for m in available_models]
    
    # Default to first model or "golf" for VW
    default_model_index = 0
    if brand == "VW" and "golf" in available_models:
        default_model_index = available_models.index("golf")
    
    model_display = st.selectbox(
        f"Model ({len(available_models)} {brand} models available)",
        options=model_options,
        index=default_model_index,
        help=f"Select a {brand} model",
        key="model_select"
    )
    
    # Convert back to lowercase
    model_name = model_display.lower()

# Row 2: Year & Mileage
col3, col4 = st.columns(2)
with col3:
    year = st.number_input(
        "Year",
        min_value=1970,
        max_value=2020,
        value=2017,
        step=1,
        help="Manufacturing year (1970-2020)"
    )
with col4:
    mileage = st.number_input(
        "Mileage (miles)",
        min_value=0,
        max_value=323_000,
        value=25_000,
        step=1_000,
        help="Total miles driven"
    )

# Row 3: Engine & MPG
col5, col6 = st.columns(2)
with col5:
    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=0.6,
        max_value=9.0,
        value=1.6,
        step=0.1,
        help="Engine displacement in liters"
    )
with col6:
    mpg = st.number_input(
        "MPG (Miles per Gallon)",
        min_value=5.0,
        max_value=150.0,
        value=50.0,
        step=0.5,
        help="Fuel efficiency"
    )

# Row 4: Tax & Previous Owners
col7, col8 = st.columns(2)
with col7:
    tax = st.number_input(
        "Road Tax (£/year)",
        min_value=0,
        max_value=580,
        value=145,
        step=5,
        help="Annual road tax in GBP"
    )
with col8:
    previous_owners = st.number_input(
        "Previous Owners",
        min_value=0,
        max_value=6,
        value=1,
        step=1,
        help="Number of previous owners"
    )

# Row 5: Transmission & Fuel Type
col9, col10 = st.columns(2)
with col9:
    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic", "Semi-Auto", "Other"],
        help="Type of transmission"
    )
with col10:
    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "Hybrid", "Electric", "Other"],
        help="Type of fuel"
    )

st.divider()

# 5) PREDICTION BUTTON & LOGIC

# Centered prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("Predict Price", type="primary", use_container_width=True)

if predict_button:
    # Build input DataFrame
    input_df = pd.DataFrame([{
        "Brand": brand,
        "model": model_name,
        "year": int(year),
        "mileage": int(mileage),
        "tax": int(tax),
        "mpg": float(mpg),
        "engineSize": float(engine_size),
        "transmission": transmission,
        "fuelType": fuel_type,
        "previousOwners": int(previous_owners),
        "hasDamage": 0,
    }])
    
    # Make prediction
    with st.spinner("Calculating price..."):
        try:
            pred_price = model.predict(input_df)[0]
            
            # Display result
            st.success("Prediction Complete")
            
            # Main price display
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px; margin: 1rem 0;'>
                <h1 style='color: #1f77b4; margin: 0;'>£{pred_price:,.0f}</h1>
                <p style='color: #666; margin-top: 0.5rem;'>Estimated Selling Price</p>
                <p style='color: #999; font-size: 0.9rem; margin-top: 0.5rem;'>{brand} {model_display} ({year})</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Car Metrics (Age & Usage only)
            st.subheader("Vehicle Metrics")
            col_a, col_b = st.columns(2)
            
            with col_a:
                car_age = 2020 - year
                st.metric(
                    "Age",
                    f"{car_age} years",
                    help="Age of the vehicle (2020 - manufacturing year)"
                )
            
            with col_b:
                miles_per_year = mileage / max(car_age, 1)
                st.metric(
                    "Usage",
                    f"{miles_per_year:,.0f} mi/yr",
                    help="Average miles driven per year"
                )
            
            # Confidence interval
            mae = 1300
            lower_bound = max(0, pred_price - mae)
            upper_bound = pred_price + mae
            
            st.info(f"""
            Price Range: £{lower_bound:,.0f} - £{upper_bound:,.0f}  
            Based on model's average error of ±£{mae:,.0f}
            """)
            
        except Exception as e:
            st.error("Prediction Failed")
            
            with st.expander("Debug Information"):
                st.write("**Error:**")
                st.code(str(e))
                
                st.write("**Input Data:**")
                st.dataframe(input_df)
                
                st.write("**Model Type:**")
                st.code(f"{type(model)}")
                
                st.exception(e)

# 6) SIDEBAR

with st.sidebar:
    st.header("About")
    
    st.markdown("""
    This tool predicts used car prices using a Random Forest machine learning model, trained on 75,969 vehicles from 1970 to 2020 with prices ranging from £450 to £159,999.
    """)
    
    st.divider()
    
    # Dataset Coverage
    st.header("Dataset Coverage")
    
    st.markdown(
    f"""
    The model was trained on a total of **9 brands** and **114 models**.  
    Peak performance is achieved on the following brands, which are best represented in the training data: \n  
    {", ".join(sorted(BRAND_MODELS.keys()))}

    Prices for all other brands and models can still be estimated by leveraging their remaining vehicle attributes.
    """
)
   
    # Detailed view on request
    if st.checkbox("Show best performing brands & models", key="show_models"):
        for brand_name in sorted(BRAND_MODELS.keys()):
            if brand_name == "Other":
                continue
            models = BRAND_MODELS[brand_name]
            with st.expander(f"{brand_name} ({len(models)})"):
                cols = st.columns(4)
                for i, model_item in enumerate(sorted(models)):
                    cols[i % 4].write(f"• {model_item.title()}")
    
    st.divider()
    
    # Model Performance
    st.header("Model Performance")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Average Error", "±£1,300")
    with col4:
        st.metric("R² Score", "0.944")
    
    st.divider()

    st.caption("Built by Group 5 - ML Project 2025")

# 7) FOOTER

st.divider()
st.caption("""
Disclaimer: This is an educational project. Predictions are estimates only 
and should not be used as the sole basis for pricing decisions.
""")