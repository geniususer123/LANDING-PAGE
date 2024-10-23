import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Train a Random Forest model on dummy data (latitude, longitude, hazard)
data = np.array([
    [18.96, 72.82, 1],  # Example: Mumbai (Cyclone)
    [13.08, 80.27, 2],  # Example: Chennai (Flood)
    [28.61, 77.20, 0]   # Example: Delhi (No hazard)
])

X = data[:, :2]  # Latitude and Longitude
y = data[:, 2]   # Hazard labels (0: Safe, 1: Cyclone, 2: Flood)

model = RandomForestClassifier()
model.fit(X, y)

# Define hazard types
hazard_types = {0: "No hazard", 1: "Cyclone", 2: "Flood"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_hazard', methods=['POST'])
def predict_hazard():
    data = request.json
    latitude = float(data['latitude'])
    longitude = float(data['longitude'])

    # Predict hazard using AI model
    prediction = model.predict([[latitude, longitude]])[0]
    hazard_info = hazard_types[prediction]

    # Generate map
    map_url = generate_map(latitude, longitude, hazard_info)

    return jsonify({'hazard': hazard_info, 'map': map_url})

def generate_map(latitude, longitude, hazard_info):
    # Create a GeoDataFrame for the location
    gdf = gpd.GeoDataFrame({'hazard': [hazard_info]}, geometry=[gpd.points_from_xy([longitude], [latitude])])
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    world.boundary.plot(ax=ax, linewidth=1)
    gdf.plot(ax=ax, color='red', markersize=100, label=hazard_info)
    plt.title(f'Hazard Prediction: {hazard_info}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.axis('equal')

    # Save the plot to a BytesIO object and encode as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
    app.run(debug=True)
