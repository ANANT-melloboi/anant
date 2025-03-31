import google.generativeai as genai
import streamlit as st
import os

# Set API key from environment or secrets
API_key = os.getenv("AIzaSyDRSxn96kdk3IHxJUPNJb3BdkFzuKkAc2U") or st.secrets.get("AIzaSyDRSxn96kdk3IHxJUPNJb3BdkFzuKkAc2U")
if not API_key:
    st.error("API key issue.")
    st.stop()

# Configure the API
genai.configure(api_key="AIzaSyDRSxn96kdk3IHxJUPNJb3BdkFzuKkAc2U")

def get_recipe(ingredients):
    try:
        # Create a prompt for the AI model
        prompt = f"Suggest a simple recipe using these ingredients: {', '.join(ingredients)}."
        
        # Use the Gemini model to generate the content
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        
        return response.text if response else "No response from the AI."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit interface
st.title("AI Recipe Generator")
user_input = st.text_input("Enter your ingredients (separate by commas):")

if st.button("Get Recipe"):
    ingredients = [i.strip() for i in user_input.split(',') if i.strip()]
    if ingredients:
        with st.spinner("Generating recipe..."):
            recipe = get_recipe(ingredients)
        st.write("### Suggested Recipe:")
        st.write(recipe)
    else:
        st.warning("Please enter at least one ingredient.")
