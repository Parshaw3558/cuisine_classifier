import streamlit as st
import joblib

# Load files
model = joblib.load("cuisine_model.pkl")
cv = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🍽️ Cuisine Classification App")
st.write("Enter restaurant details and predict the cuisine type")

# Inputs
name = st.text_input("Restaurant Name")
city = st.text_input("City")
cost = st.text_input("Average Cost for Two")

if st.button("Predict Cuisine"):
    if name == "" or city == "" or cost == "":
        st.error("Please fill all fields!")
    else:
        text = name + " " + city + " " + cost
        vector = cv.transform([text])
        pred = model.predict(vector)[0]
        cuisine = le.inverse_transform([pred])[0]
        st.success(f"Predicted Cuisine: **{cuisine}**")
