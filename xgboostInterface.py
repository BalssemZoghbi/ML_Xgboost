import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Load the trained XGBoost model and label encoders
@st.cache_data
def load_model_and_encoders():
    model = joblib.load('xgboost_model.pkl')  # Load the model saved above
    encoders = joblib.load('label_encoders.pkl')  # Load the label encoders saved above
    return model, encoders

# Function to preprocess user input
def preprocess_input(data, label_encoders):
    for column in data.columns:
        if column in label_encoders:
            le = label_encoders[column]
            try:
                data[column] = le.transform(data[column])
            except ValueError:
                st.error(f"Invalid value in '{column}'. Allowed values: {list(le.classes_)}")
                return None
    
    # Convert categorical columns to 'category' dtype
    categorical_columns = ['Style', 'Price', 'Size', 'Season', 'NeckLine', 'SleeveLength', 'Material', 'FabricType', 'Pattern Type']
    for column in categorical_columns:
        if column in data.columns:
            data[column] = data[column].astype('category')
    
    return data

# Function to map and validate inputs
def validate_and_map_inputs(style, price, size, season, neckline, sleeve_length, material, fabric_type, pattern_type, decoration):
    # Define allowed values for each input
    style_mapping = {
        'casual': 'Casual',
        'sexy': 'Sexy',
        'vintage': 'Vintage',
        'cute': 'Cute',
        'brief': 'Brief',
        'bohemian': 'Bohemian',
        'fashion': 'Fashion',
        'party': 'Party',
        'sexy': 'Sexy',
        'work': 'Work'
    }

    # Normalize and map user inputs
    style = style_mapping.get(style.lower(), None)
    price = price.capitalize()  # Ensure the first letter is capitalized (e.g., 'low' -> 'Low')
    season = season.capitalize()  # Capitalize season names
    neckline = neckline.lower()  # Normalize the case
    sleeve_length = sleeve_length.lower()  # Normalize the case
    material = material.lower()  # Normalize the case
    fabric_type = fabric_type.lower()  # Normalize the case
    pattern_type = pattern_type.lower()  # Normalize the case
    decoration = decoration.lower()  # Normalize the case

    # Check if required fields are valid
    if not style:
        st.error("Invalid value for 'Style'. Allowed values: ['Casual', 'Sexy', 'Vintage', 'Cute', 'Brief', 'Bohemian', 'Fashion', 'Party', 'Sexy', 'Work']")
        return None
    
    return style, price, size, season, neckline, sleeve_length, material, fabric_type, pattern_type, decoration

# Streamlit app layout
def main():
    st.title("Dress Recommendation System")
    st.write("Predict whether a dress is **recommended** or **not recommended** using an advanced XGBoost model.")

    # Load model and encoders
    model, label_encoders = load_model_and_encoders()

    # Sidebar inputs for user data
    st.sidebar.header("Input Features")
    style = st.sidebar.selectbox("Style", ["Casual", "Sexy", "vintage", "cute", "Brief", "bohemian", "fashion", "party", "sexy", "vintage", "work"])
    price = st.sidebar.selectbox("Price", ["Low", "Medium", "High"])
    rating = st.sidebar.slider("Rating", 0.0, 5.0, step=0.1, value=4.0)
    size = st.sidebar.selectbox("Size", ["S", "M", "L", "XL", "free"])
    season = st.sidebar.selectbox("Season", ["Summer", "Spring", "Autumn", "Winter"])
    neckline = st.sidebar.selectbox("NeckLine", ["round-neck", "v-neck", "boat-neck"])
    sleeve_length = st.sidebar.selectbox("SleeveLength", ["sleeveless", "short", "full", "half-sleeve"])
    material = st.sidebar.selectbox("Material", ["cotton", "silk", "polyester", "microfiber", "Unknown"])
    fabric_type = st.sidebar.selectbox("FabricType", ["chiffon", "broadcloth", "corduroy", "Unknown"])
    pattern_type = st.sidebar.selectbox("Pattern Type", ["solid", "print", "striped", "animal", "dot"])
    decoration = st.sidebar.selectbox("Decoration", ["Tiered", "applique", "beading", "bow", "button", "cascading", "crystal", "draped", "embroidary", "feathers", "flowers", "hollowout", "lace", "nan", "none", "pearls", "plain", "pleat", "pockets", "rivet", "ruched", "ruffles", "sashes", "sequined", "tassel"])

    # Validate and map the inputs
    inputs = validate_and_map_inputs(style, price, size, season, neckline, sleeve_length, material, fabric_type, pattern_type, decoration)
    if inputs is None:
        return  # Exit if invalid input

    # Create input data
    style, price, size, season, neckline, sleeve_length, material, fabric_type, pattern_type, decoration = inputs
    input_data = pd.DataFrame({
        'Style': [style],
        'Price': [price],
        'Rating': [rating],
        'Size': [size],
        'Season': [season],
        'NeckLine': [neckline],
        'SleeveLength': [sleeve_length],
        'Material': [material],
        'FabricType': [fabric_type],
        'Pattern Type': [pattern_type],
        'Decoration': [decoration],
    })

    st.write("### User Input:")
    st.table(input_data)

    # Preprocess input and predict
    if st.button("Predict Recommendation"):
        processed_data = preprocess_input(input_data.copy(), label_encoders)
        if processed_data is not None:
            # Predict using the model
            prediction = model.predict(processed_data)
            
            if prediction[0] == 1:
                st.success("🌟 This dress is highly **recommended** for your collection!")
            else:
                st.warning("⚠️ This dress is **not recommended** based on the current features.")

if __name__ == "__main__":
    main()
