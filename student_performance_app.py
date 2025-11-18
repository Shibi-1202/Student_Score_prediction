import streamlit as st
import numpy as np
import joblib

# -------------------- LOAD ASSETS --------------------
@st.cache_resource
def load_assets():
    model = joblib.load("Student_ScorePredict.pkl")
    scaler = joblib.load("Scaling_Score.pkl")
    selector = joblib.load("Selector_Score.pkl")
    return model, scaler, selector

model, scaler, selector = load_assets()

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Student Score Prediction", layout="centered")

st.title("🎓 Student Score Prediction App")
st.write("Enter the student's details below to get their predicted score.")

st.markdown("---")

# -------------------- USER INPUTS --------------------
st.sidebar.header("Input Parameters")

study_hours = st.sidebar.number_input("Study Hours per Day", 0.0, 12.0, 2.0)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
sleep_hours = st.sidebar.number_input("Sleep Hours per Day", 0.0, 12.0, 7.0)
previous_score = st.sidebar.slider("Previous Exam Score", 0, 100, 70)

# Put the inputs into a structured format
input_data = {
    "Study Hours": study_hours,
    "Attendance": attendance,
    "Sleep Hours": sleep_hours,
    "Previous Score": previous_score
}

# Display structured input
st.subheader("📌 Inputs Provided")
st.json(input_data)

# Convert to array
input_array = np.array([[study_hours, attendance, sleep_hours, previous_score]])

# -------------------- PREDICTION --------------------
if st.button("Predict Score"):
    try:
        # Scale
        scaled = scaler.transform(input_array)

        # Feature selection
        transformed = selector.transform(scaled)

        # Predict
        prediction = model.predict(transformed)[0]

        # Output UI
        st.success(f"🎯 Predicted Score: **{round(prediction,2)}** / 100")

        # Terminal Logging
        print("\n--- NEW PREDICTION RECEIVED ---")
        print("Inputs:", input_data)
        print("Predicted Score:", prediction)
        print("------------------------------\n")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
