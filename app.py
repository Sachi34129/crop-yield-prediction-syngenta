from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

# Load the saved models
# Use os.path.join to make file paths more portable
pipe = pickle.load(open(os.path.join(os.path.dirname(__file__), 'crop_yield_pipeline.pkl'), 'rb'))
preprocessor = pickle.load(open(os.path.join(os.path.dirname(__file__), 'crop_yield_preprocessor.pkl'), 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Get unique crops and states for dropdown options
    crops = [
        'Arecanut', 'Arhar/Tur', 'Castor seed', 'Coconut ', 'Cotton(lint)',
        'Dry chillies', 'Gram', 'Jute', 'Linseed', 'Maize', 'Mesta',
        'Niger seed', 'Onion', 'Other  Rabi pulses', 'Potato',
        'Rapeseed &Mustard', 'Rice', 'Sesamum', 'Small millets',
        'Sugarcane', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric',
        'Wheat', 'Bajra', 'Black pepper', 'Cardamom', 'Coriander',
        'Garlic', 'Ginger', 'Groundnut', 'Horse-gram', 'Jowar', 'Ragi',
        'Cashewnut', 'Banana', 'Soyabean', 'Barley', 'Khesari', 'Masoor',
        'Moong(Green Gram)', 'Other Kharif pulses', 'Safflower',
        'Sannhamp', 'Sunflower', 'Urad', 'Peas & beans (Pulses)',
        'other oilseeds', 'Other Cereals', 'Cowpea(Lobia)',
        'Oilseeds total', 'Guar seed', 'Other Summer Pulses', 'Moth'
    ]
    states = [
        'Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal',
        'Puducherry', 'Goa', 'Andhra Pradesh', 'Tamil Nadu', 'Odisha',
        'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram',
        'Punjab', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh',
        'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand', 'Jharkhand',
        'Delhi', 'Manipur', 'Jammu and Kashmir'
    ]
    seasons = ['Whole Year', 'Kharif', 'Rabi', 'Autumn', 'Summer', 'Winter']
    year_intervals = ['90s', '2000s', '2010s']
    
    return render_template('index.html', 
                           crops=crops, 
                           states=states, 
                           seasons=seasons,
                           year_intervals=year_intervals)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        crop = request.form['Crop']
        season = request.form['Season']
        state = request.form['State']
        area = float(request.form['Area'])
        production = float(request.form['Production'])
        annual_rainfall = float(request.form['Annual_Rainfall'])
        input_per_unit_area = float(request.form['Input_Per_Unit_Area'])
        year_interval = request.form['Year_Interval']

        # Create a DataFrame with the input
        input_data = pd.DataFrame({
            'Crop': [crop],
            'Season': [season],
            'State': [state],
            'Area': [area],
            'Production': [production],
            'Annual_Rainfall': [annual_rainfall],
            'Input_Per_Unit_Area': [input_per_unit_area],
            'Year_Interval': [year_interval]
        })

        # Make prediction
        prediction = pipe.predict(input_data)

        # Get unique crops and states for dropdown options
        crops = [
            'Arecanut', 'Arhar/Tur', 'Castor seed', 'Coconut ', 'Cotton(lint)',
            'Dry chillies', 'Gram', 'Jute', 'Linseed', 'Maize', 'Mesta',
            'Niger seed', 'Onion', 'Other  Rabi pulses', 'Potato',
            'Rapeseed &Mustard', 'Rice', 'Sesamum', 'Small millets',
            'Sugarcane', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric',
            'Wheat', 'Bajra', 'Black pepper', 'Cardamom', 'Coriander',
            'Garlic', 'Ginger', 'Groundnut', 'Horse-gram', 'Jowar', 'Ragi',
            'Cashewnut', 'Banana', 'Soyabean', 'Barley', 'Khesari', 'Masoor',
            'Moong(Green Gram)', 'Other Kharif pulses', 'Safflower',
            'Sannhamp', 'Sunflower', 'Urad', 'Peas & beans (Pulses)',
            'other oilseeds', 'Other Cereals', 'Cowpea(Lobia)',
            'Oilseeds total', 'Guar seed', 'Other Summer Pulses', 'Moth'
        ]
        states = [
            'Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal',
            'Puducherry', 'Goa', 'Andhra Pradesh', 'Tamil Nadu', 'Odisha',
            'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram',
            'Punjab', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh',
            'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand', 'Jharkhand',
            'Delhi', 'Manipur', 'Jammu and Kashmir'
        ]
        seasons = ['Whole Year', 'Kharif', 'Rabi', 'Autumn', 'Summer', 'Winter']
        year_intervals = ['90s', '2000s', '2010s']

        return render_template('index.html', 
                               prediction=f"{prediction[0]:.4f}", 
                               crops=crops, 
                               states=states, 
                               seasons=seasons,
                               year_intervals=year_intervals,
                               selected_crop=crop,
                               selected_season=season,
                               selected_state=state,
                               selected_area=area,
                               selected_production=production,
                               selected_rainfall=annual_rainfall,
                               selected_input_per_unit_area=input_per_unit_area,
                               selected_year_interval=year_interval)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)