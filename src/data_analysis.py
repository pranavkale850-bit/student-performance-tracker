import pandas as pd
import numpy as np



df = pd.read_csv(r"C:\Users\prana\Downloads\student_project\student_data.csv")


df["performance"] = np.where(
    df["G3"] >= 15, "Good",
    np.where(df["G3"] >= 10, "Average", "Poor"))

df = df[[
    "school","sex","age","traveltime","studytime","failures",
    "schoolsup","famsup","paid","activities","nursery","higher",
    "internet","romantic","famrel","freetime","goout",
    "Dalc","Walc","health","absences","G3","performance"
]]

binary_cols = [
    "schoolsup","famsup","paid","activities",
    "nursery","higher","internet","romantic"
]

for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df = pd.get_dummies(df, columns=["school", "sex"], dtype=int)

df = df.drop(columns=["G3"])

X = df.drop("performance", axis=1)
y = df["performance"]

print(X.head().to_string())
print("\nTarget:\n", y.head())

df.to_csv("clean_data.csv", index=False)

