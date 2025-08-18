#ChatGPT was used to adapt previous GUI to work with this model 
import tkinter as tk
import pandas as pd
from joblib import load

# Load model and scaler
model = load("Logistic Regression/logistic_regression.joblib")
sc = load("Logistic Regression/scaler.joblib")

# Features
numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']
gender_feature = 'gender'
country_features = ['country_Germany', 'country_Spain']
labels = ["No", "Yes"]

# Creating GUI window
window = tk.Tk()
window.title("Customer Churn Prediction")
window.option_add("*font", "lucida 20 bold")

entries = {}

# Numeric feature entries
for feat in numeric_features:
    label = tk.Label(window, text=feat.replace('_', ' ').title())
    label.pack()
    entry = tk.Entry(window)
    entry.pack(pady=5)
    entries[feat] = entry

# Gender toggle
gender_var = tk.IntVar(value=0)
tk.Label(window, text="Gender").pack()
tk.Radiobutton(window, text="Female", variable=gender_var, value=0).pack()
tk.Radiobutton(window, text="Male", variable=gender_var, value=1).pack()

# Country toggles (Yes=1 / No=0)
country_vars = {}
for feat in country_features:
    var = tk.IntVar(value=0)
    tk.Label(window, text=feat.replace('_', ' ').title()).pack()
    tk.Radiobutton(window, text="No", variable=var, value=0).pack()
    tk.Radiobutton(window, text="Yes", variable=var, value=1).pack()
    country_vars[feat] = var

# Result display
result_var = tk.StringVar()
tk.Label(window, textvariable=result_var).pack(pady=20)

def set_text_by_button():
    input_data = {}
    try:
        # Collect numeric inputs
        for feat in numeric_features:
            input_data[feat] = float(entries[feat].get())
        # Gender
        input_data[gender_feature] = gender_var.get()
        # Country flags
        for feat in country_features:
            input_data[feat] = country_vars[feat].get()
    except Exception as e:
        result_var.set(f"Error: {e}")
        return

    # Create DataFrame and reorder columns to match scaler
    sample = pd.DataFrame([input_data])
    scaler_columns = ['credit_score', 'gender', 'age', 'tenure', 'balance',
                      'products_number', 'credit_card', 'active_member',
                      'estimated_salary', 'country_Germany', 'country_Spain']
    sample = sample[scaler_columns]

    # Scale the sample
    sample_scaled = sc.transform(sample)

    # Prediction
    prediction = model.predict(sample_scaled)[0]
    prediction_prob = model.predict_proba(sample_scaled)[0]

    # Display result
    result_var.set(
        f"Did this person churn? {labels[prediction]}\n"
        f"Probability No: {prediction_prob[0]:.2f}, Yes: {prediction_prob[1]:.2f}"
    )

# Predict button
tk.Button(window, height=1, width=20, text="Predict Churn", command=set_text_by_button).pack(pady=20)

window.mainloop()
