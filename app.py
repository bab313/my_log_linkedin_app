import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st

#jupyter notebook code
s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1,0)
    return x

ss = pd.DataFrame(
    {'sm_li': clean_sm(s["web1h"]),
     'income': np.where(s['income'] > 9, np.nan,s['income']),
     'education': np.where(s['educ2'] >8, np.nan, s['educ2']),
     'parent': np.where(s['par'] == 1, 1, 0),
     'married': np.where(s['marital'] == 1, 1, 0),
     'female': np.where(s['gender'] == 2, 1, 0),
     'age': np.where(s['age'] > 98, np.nan, s['age'])
    })
ss = ss.dropna()

ss = ss.astype({'income':'int', 'education':'int', 'age':'int'})

Y = ss['sm_li']

X = ss[['income', 'education', 'parent', 'married', 'female','age']]

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    stratify=Y,       
                                                    test_size=0.2,    
                                                    random_state=123)

lr = LogisticRegression(class_weight = 'balanced')

lr.fit(X_train, Y_train)

#streamlit code

#title
st.markdown("LinkedIn User Prediction")

#variables for user inputs
income_var = st.number_input("Income (low=1 to high=9)", 1, 9)
education_var = st.selectbox("Education level", 
             options = ["Less than high school",
                        "High School incomplete",
                        "High School Diploma",
                        "Some College, no degree",
                        "Two Year Associate",
                        "Bachelors Degree",
                        "Some postgraduate",
                        "Graduate Degree"])
parent_var = st.number_input("Parent? (0=no, 1=yes)", 0, 1)
married_var = st.number_input("Married (0=no, 1=yes)", 0, 1)
female_var = st.number_input("Female? (0=no, 1=yes)", 0, 1)
age_var = st.slider(label="Enter your age",
          min_value=1,
          max_value=99,
          value=50)


#convert education selection to number
if education_var == "Less than high school":
    education_var = 1
elif education_var == "High School incomplete":
    education_var = 2
elif education_var == "High School Diploma":
    education_var = 3
elif education_var == "Some College, no degree":
    education_var = 4
elif education_var == "Two Year Associate":
    education_var = 5
elif education_var == "Bachelors Degree":
    education_var = 6
elif education_var == "Some postgraduate":
    education_var = 7
else:
    education_var = 8


user_data = pd.DataFrame({
    'income': [],
    'education': [],
    'parent':[],
    'married':[],
    'female':[],
    'age':[]
})

user_data = user_data.append({
    'income': income_var,
    'education': education_var,
    'parent':parent_var,
    'married':married_var,
    'female':female_var,
    'age':age_var
}, ignore_index=True)

predicted_class = lr.predict(user_data)
probs = lr.predict_proba(user_data)

if predicted_class[0] == 1:
    label = "Yes"
else:
    label = "No"

st.markdown(f"Am I a LinkedIn user?: {label}") 
st.markdown(f"Probability that I am a LinkedIn user: {probs[0][1]:.2f}")