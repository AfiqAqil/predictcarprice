import streamlit as st
import pandas as pd
from sklearn.externals import joblib  # For scikit-learn version < 0.23
# If using scikit-learn version >= 0.23, use:
# from sklearn import joblib

# Load the trained model
model = joblib.load('random_forest_model.joblib')

<<<<<<< HEAD
# Function for data preprocessing
def preprocess_data(registered_year, engine_capacity, insurance, transmission_type, kms_driven, owner_type, fuel_type, max_power, seats, mileage, body_type, city, brand):
    data = {
        'registered_year': [registered_year],
        'engine_capacity': [engine_capacity],
        'insurance': [insurance],
        'transmission_type': [transmission_type],
        'kms_driven': [kms_driven],
        'owner_type': [owner_type],
        'fuel_type': [fuel_type],
        'max_power': [max_power],
        'seats': [seats],
        'mileage': [mileage],
        'body_type': [body_type],
        'city': [city],
        'brand': [brand]
    }
=======
# List of features used during model training
trained_features = ['registered_year', 'engine_capacity', 'resale_priceINR', 'kms_driven', 'max_power', 'mileage'] + list(all_possible_categories)

# Function to preprocess input data
def preprocess_input(input_data, trained_features):
    # Encode 'owner_type'
    input_data["owner_type"] = input_data["owner_type"].replace("First Owner", "1").replace("Second Owner", "2").replace("Third Owner", "3").replace("Fourth Owner", "4").replace("Fifth Owner", "5")
    input_data["owner_type"] = int(input_data["owner_type"])

    # Get dummies for categorical columns
    input_data = pd.get_dummies(input_data, columns=['insurance', 'transmission_type', 'owner_type', 'fuel_type', 'body_type', 'city', 'brand'])

    # Ensure "resale_priceINR" is present in the input data
    if "resale_priceINR" not in input_data.columns:
        input_data["resale_priceINR"] = 0  # You can replace 0 with a default value or handle it based on your use case

    # Handle new categories during prediction
    for feature in trained_features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # You can replace 0 with a default value or handle it based on your use case

    # Reorder columns to match the order during training
    input_data = input_data[trained_features]
>>>>>>> 2d43bdc7da2897f2cc1aef02383b775374f8d6ae

    df = pd.DataFrame(data)
    # Perform any necessary data preprocessing steps based on your model training data preprocessing
    # For example, one-hot encode categorical variables, handle missing values, etc.

<<<<<<< HEAD
    # Return the preprocessed DataFrame
    return df
=======
    return input_data


# Function to predict price
def predict_price(input_data, trained_features):
    # Preprocess input and make predictions
    input_data = preprocess_input(pd.DataFrame([input_data]), trained_features)  # Wrap input_data in DataFrame

    # Ensure columns match
    X = input_data[trained_features]

    prediction = model.predict(X)
    return prediction[0]
>>>>>>> 2d43bdc7da2897f2cc1aef02383b775374f8d6ae

# Streamlit app
st.title('Car Price Prediction App')

# Input form
registered_year = st.number_input('Registered Year', min_value=1900, max_value=2024, value=2020)
engine_capacity = st.number_input('Engine Capacity (cc)', min_value=300, max_value=10000, value=1300)
insurance = st.selectbox('Insurance Type', ['Comprehensive', 'Not Available', 'Third Party', 'Zero Dep'])
transmission_type = st.selectbox('Transmission Type', ['Automatic', 'Manual'])
kms_driven = st.number_input('Kilometers Driven', min_value=0, value=100000)
owner_type = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner','Fifth Owner'])
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG' , 'Electric', 'LPG'])
<<<<<<< HEAD
max_power = st.number_input('Max Power (bhp)', min_value=0, value=0)
=======
max_power = st.number_input('Max Power (bhp)', min_value=00, value=0)
>>>>>>> 2d43bdc7da2897f2cc1aef02383b775374f8d6ae
seats = st.number_input('Number of Seats', min_value=1, value=1)
mileage = st.number_input('Mileage (kmpl)', min_value=0, value=0)
body_type = st.selectbox('Body Type', ['Convertibles', 'Hatchback', 'Sedan', 'SUV', 'Minivans', 'Coupe', 'Pickup', 'Wagon'])
city = st.selectbox('City', ['Agra', 'Gurgaon', 'Lucknow', 'Delhi', 'Chandigarh', 'Bangalore', 'Jaipur', 'Kolkata', 'Ahmedabad', 'Chennai', 'Pune', 'Mumbai', 'Hyderabad'])
<<<<<<< HEAD
brand = st.selectbox('Brand', ['Audi', 'BMW', 'Chevrolet', 'Citroen', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'] )
=======
brand = st.selectbox('Brand', ['Audi', 'BMW', 'Chevrolet', 'Citroen', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'] )  # Replace with a dropdown if
>>>>>>> 2d43bdc7da2897f2cc1aef02383b775374f8d6ae

# Prediction button
if st.button('Predict'):
    input_data = preprocess_data(registered_year, engine_capacity, insurance, transmission_type, kms_driven, owner_type, fuel_type, max_power, seats, mileage, body_type, city, brand)

<<<<<<< HEAD
    # Make predictions using the trained model
    prediction = model.predict(input_data)[0]

    st.success(f'The predicted car resale price is {prediction} INR')
=======
    prediction = predict_price(input_data, trained_features)

    st.success(f'The predicted resale price is: {prediction} INR')
>>>>>>> 2d43bdc7da2897f2cc1aef02383b775374f8d6ae
