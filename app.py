import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset (from Seaborn or CSV if needed)
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    df.dropna(inplace=True)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    return df

df = load_data()

# Split data
X = df.drop("Survived", axis=1)
y = df["Survived"]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Streamlit UI
st.title("🚢 Titanic Survival Predictor")

st.markdown("Enter passenger details to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of siblings/spouses aboard", 0, 10, 0)
parch = st.number_input("Number of parents/children aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert inputs to model-ready format
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger would have **SURVIVED**.")
    else:
        st.error("💀 The passenger would have **NOT survived**.")

