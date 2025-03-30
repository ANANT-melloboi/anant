import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import hashlib
import requests
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import re

# ---- Session State Management ----
if "user_db" not in st.session_state:
    st.session_state.user_db = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "model" not in st.session_state:
    st.session_state.model = None
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {}
if "selected_hotel" not in st.session_state:
    st.session_state.selected_hotel = None


# ---- Utility Functions ----
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def sign_up(username, password):
    if username in st.session_state.user_db:
        st.error("Username already exists!")
        return False
    else:
        st.session_state.user_db[username] = hash_password(password)
        st.success("Sign-up successful! Please log in.")
        return True


def login(username, password):
    if username in st.session_state.user_db and st.session_state.user_db[username] == hash_password(password):
        st.success(f"Welcome, {username}!")
        st.session_state.logged_in = True
        st.session_state.username = username
        # Initialize user preferences upon login
        st.session_state.user_preferences[username] = {
            'rating': 3.0,
            'facilities': [],
            'customer_review_score': 7.0,
            'distance': 5.0  # Default distance
        }
        return True
    else:
        st.error("Invalid username or password.")
        return False


def get_coordinates(city, area):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={area}, {city}, India"
    try:
        response = requests.get(url, headers={"User-Agent": "HotelFinderApp"})
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        else:
            st.warning(f"No results found for {area}, {city}, India. Try a more specific area.")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch coordinates: {e}")
        return None, None


def fetch_hotels(lat, lon, radius):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (node["tourism"="hotel"](around:{radius},{lat},{lon});
     way["tourism"="hotel"](around:{radius},{lat},{lon});
     relation["tourism"="hotel"](around:{radius},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """

    try:
        response = requests.get(overpass_url, params={"data": query})
        response.raise_for_status()
        data = response.json()
        hotels = []
        for element in data.get("elements", []):
            if element.get("type") in ["node", "way", "relation"]:
                hotel = {
                    "Hotel Name": element.get("tags", {}).get("name", "N/A"),
                    "Latitude": element.get("lat") if "lat" in element else (
                        element.get("center", {}).get("lat") if element.get("type") == "relation" and "center" in element else None),
                    "Longitude": element.get("lon") if "lon" in element else (
                        element.get("center", {}).get("lon") if element.get("type") == "relation" and "center" in element else None),

                }
                # Only add the hotel if both latitude and longitude are available
                if hotel["Latitude"] is not None and hotel["Longitude"] is not None:
                    hotels.append(hotel)

        if not hotels:
            st.warning("No hotels found in the specified area.")
        return pd.DataFrame(hotels)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch hotels: {e}")
        return pd.DataFrame()


# ---- Hotel Bestness Prediction Functions ----
def create_sample_data():
    return pd.DataFrame({
        'Price': [1000, 2000, 1500, 800, 1200, 1800, 900, 1100],  # More data
        'Facilities': [5, 3, 5, 2, 4, 4, 3, 5],
        'Rating': [4.5, 3.5, 5.0, 3.0, 4.0, 4.2, 3.8, 4.7],
        'Distance': [1.5, 5.0, 0.5, 10.0, 2.5, 3.0, 7.0, 1.0],
        'Customer_Review_Score': [8.5, 7.0, 9.2, 6.5, 7.8, 8.0, 7.5, 9.0],
        'Best_Hotel': [1, 0, 1, 0, 1, 1, 0, 1]  # More balanced examples
    })


def train_hotel_model():
    df = create_sample_data()
    X = df[['Price', 'Facilities', 'Rating', 'Distance', 'Customer_Review_Score']]
    y = df['Best_Hotel']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    return scaler, model


def predict_best_hotel(price, facility_available, rating, distance, customer_review_score, scaler, model):
    if scaler is None or model is None:
        return None

    input_data = [[price, facility_available, rating, distance, customer_review_score]]
    try:
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


# ---- Streamlit UI ----
st.title("LuxeNumo")

# Authentication Sidebar
with st.sidebar:
    st.header("User Authentication")
    choice = st.selectbox("Choose an option", ["Login", "Sign Up"])
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    if choice == "Sign Up":
        if st.button("Sign Up"):
            if sign_up(username, password):
                st.info("Please log in.")

    elif choice == "Login":
        if st.button("Login"):
            if login(username, password):
                st.success(f"Logged in as {st.session_state.username}")

                # Train the model here, after login.  Avoids training unless logged in.
                st.session_state.scaler, st.session_state.model = train_hotel_model()

# Main App Interface (Conditional Rendering)
if st.session_state.logged_in:
    with st.sidebar:
        st.success(f"Logged in as {st.session_state.username}")

        # Filters
        st.header("Filter Hotels")
        city = st.text_input("Enter City", value="Bengaluru")
        famous_places = {
            "Bengaluru": ["Majestic", "MG Road", "Whitefield", "Indiranagar"],
            "Mumbai": ["Bandra", "Colaba", "Andheri", "Juhu"],
            "Delhi": ["Connaught Place", "Karol Bagh", "Saket", "Paharganj"]
        }
        if city in famous_places:
            area = st.selectbox("Select Area", famous_places[city])
        else:
            area = st.text_input("Enter Area in the City")

        radius = st.slider("Search Radius (meters)", 100, 3000, 1000, 100)

        # User Preference Sliders (in sidebar)
        st.subheader("Your Preferences")
        user_rating = st.slider("Preferred Star Rating (1-5)", min_value=1.0, max_value=5.0,
                                 value=st.session_state.user_preferences[st.session_state.username]['rating'],
                                 step=0.1)
        user_facilities = st.multiselect("Must-Have Facilities", ["Gym", "Pool", "Wi-Fi", "Parking"],
                                         default=st.session_state.user_preferences[st.session_state.username][
                                             'facilities'])
        user_customer_review_score = st.slider("Minimum Customer Review Score (1-10)", min_value=1.0, max_value=10.0,
                                                value=st.session_state.user_preferences[st.session_state.username][
                                                    'customer_review_score'], step=0.1)
        user_distance = st.slider("Maximum Distance from Area Center (km)", min_value=0.5, max_value=20.0, value=st.session_state.user_preferences[st.session_state.username]['distance'], step=0.1)

        # Store the user's preferences in session state
        st.session_state.user_preferences[st.session_state.username]['rating'] = user_rating
        st.session_state.user_preferences[st.session_state.username]['facilities'] = user_facilities
        st.session_state.user_preferences[st.session_state.username][
            'customer_review_score'] = user_customer_review_score
        st.session_state.user_preferences[st.session_state.username]['distance'] = user_distance #save distance



    if city and area:
        area_lat, area_lon = get_coordinates(city, area)

        if area_lat and area_lon:
            df = fetch_hotels(area_lat, area_lon, radius)

            if not df.empty:
                # Ensure Latitude and Longitude are numeric before distance calculation
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

                # Drop rows where Latitude or Longitude is NaN after coercion
                df = df.dropna(subset=['Latitude', 'Longitude'])

                if not df.empty:
                    df['Distance'] = df.apply(lambda row: geodesic((area_lat, area_lon), (row['Latitude'], row['Longitude'])).meters,
                                               axis=1)
                    df_sorted = df.sort_values(by='Distance')

                    # Filter by the user's preferred distance
                    df_sorted = df_sorted[df_sorted['Distance'] <= (user_distance * 1000)] #Distance KM conversion to meters

                    st.write(f"### Available Hotels in {area}, {city} (Sorted by Distance)")
                    st.dataframe(df_sorted)

                    map_center = [area_lat, area_lon]
                    hotel_map = folium.Map(location=map_center, zoom_start=12)
                    for _, row in df_sorted.iterrows():
                        folium.Marker([row["Latitude"], row["Longitude"]],
                                      popup=f"{row['Hotel Name']} ({row['Distance']:.0f}m)").add_to(hotel_map)
                    folium_static(hotel_map)

                    #  "Best Hotel" Prediction Section
                    st.header("Select Best Hotel")
                    hotel_options = []  # Store hotel names for radio buttons
                    hotel_data = {}  # Store hotel information

                    for index, row in df_sorted.iterrows():
                        hotel_name = row['Hotel Name']
                        hotel_options.append(hotel_name)
                        hotel_data[hotel_name] = {  # Example information
                            "Distance": row['Distance'] / 1000,  # Distance in km
                            "Latitude": row['Latitude'],
                            "Longitude": row['Longitude']
                        }

                    # Radio buttons to select the best hotel
                    st.session_state.selected_hotel = st.radio("Select the best hotel:", hotel_options)

                    if st.button("Predict"):
                        if st.session_state.selected_hotel:
                            selected_hotel_data = hotel_data[st.session_state.selected_hotel]
                            distance_km = selected_hotel_data['Distance']  # in KM

                            if st.session_state.scaler is None or st.session_state.model is None:
                                st.warning("Model not trained yet. Please log in again.")
                            else:
                                # Use the user's preferred distance value in the prediction
                                prediction = predict_best_hotel(1000, len(user_facilities), user_rating, distance_km, user_customer_review_score, st.session_state.scaler, st.session_state.model)
                                if prediction == 1:
                                    st.success(f"According to your preferences, {st.session_state.selected_hotel} is the best hotel!")
                                else:
                                    st.warning(f"According to your preferences, {st.session_state.selected_hotel} is not the best hotel.")
                        else:
                            st.warning("Please select a hotel to predict.")

                else:
                    st.write("No hotels found after cleaning invalid coordinates.")
            else:
                st.write("No hotels found in the specified area.")
        else:
            st.write("Failed to fetch location coordinates.")
    elif st.session_state.logged_in:
        st.write("Enter a city and area to search for hotels.")

else:
    st.warning("Please log in to access the hotel booking system.")