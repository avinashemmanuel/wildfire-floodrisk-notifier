import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

#load model
model = joblib.load("model.pkl")

# sample data
data = pd.DataFrame([{
    "temperature": 20.8,
    "wind_speed": 4.2,
    "fire_count": 0,
    "brightness": 0,
    "ndvi": 0.022
}])

# explainer
explainer = shap.TreeExplainer(model)

# compute SHAP
shap_values = explainer(data)

# WATERFALL PLOT
shap.plots.waterfall(shap_values[0], max_display=5)

plt.show()