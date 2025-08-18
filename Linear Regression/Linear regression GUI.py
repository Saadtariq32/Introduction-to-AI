#Class Examples for GUI and Chatgpt was used as reference
import tkinter
import pandas as pd
from joblib import load
model = load("Linear Regression\linearreg1.joblib")

# map all the variables from the user
tester_row = {
    'relative_compactness': 0.8,
    'surface_area': 600,
    'wall_area': 270,
    'roof_area': 220,
    'overall_height': 7.0,
    'orientation': 2,
    'glazing_area': 0.1,
    'glazing_area_distribution': 3
}

# convert to pandas-format
tester_row = pd.DataFrame([tester_row])

# get the output/result/answer from the model
# based on the user's new data
result = model.predict(tester_row)[0]
print(f"Predicted Heating Load: {result[0]:.2f}")
print(f"Predicted Cooling Load: {result[1]:.2f}")

# Creating the GUI window.
window = tkinter.Tk()
window.title("Heating & Cooling Load Prediction")
window.option_add("*font", "lucida 20 bold")

# Creating entries for each feature
entries = {}
features = [
    'relative_compactness', 'surface_area', 'wall_area', 'roof_area',
    'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution'
]

for feat in features:
    label = tkinter.Label(window, text=feat.replace('_', ' ').title())
    label.pack()
    entry = tkinter.Entry(window)
    entry.pack(pady=5)
    entries[feat] = entry

# Result display variable
result_var = tkinter.StringVar()
label = tkinter.Label(window, textvariable=result_var)
label.pack(pady=20)

def set_text_by_button():
    input_data = {}
    try:
        for feat in features:
            val = entries[feat].get()
            # Convert types accordingly
            if feat in ['orientation', 'glazing_area_distribution']:
                input_data[feat] = int(val)
            else:
                input_data[feat] = float(val)

    except Exception as e:
        result_var.set(f"Error: {e}")
        return
    
    sample = pd.DataFrame([input_data])
    prediction = model.predict(sample)[0]
    heating_load, cooling_load = prediction[0], prediction[1]

    result_var.set(f"Predicted Heating Load: {heating_load:.2f}\nPredicted Cooling Load: {cooling_load:.2f}")

set_up_button = tkinter.Button(window, height=1, width=20, text="Predict Loads", command=set_text_by_button)
set_up_button.pack(pady=20)

window.mainloop()
