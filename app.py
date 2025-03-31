import numpy as np
import joblib
import gradio as gr
import tensorflow as tf

# Load model and encoders
model = tf.keras.models.load_model("house_price_model.h5")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

def predict_price(neighborhood, overall_qual, gr_liv_area, garage_cars,
                  total_bsmt_sf, year_built, full_bath, bedroom_abvgr):
    neighborhood = le.transform([neighborhood])[0]
    input_data = np.array([[neighborhood, overall_qual, gr_liv_area, garage_cars,
                            total_bsmt_sf, year_built, full_bath, bedroom_abvgr]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    return round(float(pred[0][0]), 2)

iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Textbox(label="Neighborhood"),
        gr.Slider(1, 10, step=1, label="Overall Quality"),
        gr.Number(label="Above Ground Living Area (sqft)"),
        gr.Slider(0, 4, step=1, label="Garage Cars"),
        gr.Number(label="Total Basement SF"),
        gr.Number(label="Year Built"),
        gr.Slider(0, 4, step=1, label="Full Bathrooms"),
        gr.Slider(0, 6, step=1, label="Bedrooms Above Ground")
    ],
    outputs=gr.Number(label="Predicted Sale Price"),
    title="Ames House Price Predictor",
    description="Enter house details to predict sale price using a deep learning model."
)

iface.launch()
