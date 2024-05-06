from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load data from CSV
df = pd.read_csv('CARS_1.csv')

# Encode categorical data and prepare label encoders
categorical_features = ['fuel_type', 'transmission_type', 'body_type']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Remove unused columns and scale numerical features
features_to_scale = ['engine_displacement', 'no_cylinder', 'seating_capacity', 'starting_price', 'ending_price']
df = df.drop(['reviews_count', 'fuel_tank_capacity', 'rating', 'max_torque_nm', 'max_torque_rpm', 'max_power_bhp', 'max_power_rp'], axis=1)
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Prepare Nearest Neighbors
nbrs = NearestNeighbors(n_neighbors=1).fit(df.drop('car_name', axis=1))
import joblib

# Save the models to disk
joblib.dump(nbrs, 'model_neighbors.pkl')
joblib.dump(scaler, 'scaler.pkl')
for col, encoder in label_encoders.items():
    joblib.dump(encoder, f'label_encoder_{col}.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form
        user_input = {
            'fuel_type': request.form['fuel_type'],
            'engine_displacement': float(request.form['engine_displacement']),
            'no_cylinder': int(request.form['no_cylinder']),
            'seating_capacity': int(request.form['seating_capacity']),
            'transmission_type': request.form['transmission_type'],
            'body_type': request.form['body_type'],
            'starting_price': float(request.form['price_low']),
            'ending_price': float(request.form['price_high'])
        }

        # Encode and scale the input
        for col in categorical_features:
            user_input[col] = label_encoders[col].transform([user_input[col]])[0]

        user_input_df = pd.DataFrame([user_input])
        user_input_df[features_to_scale] = scaler.transform(user_input_df[features_to_scale])

        # Predict the nearest car
        distance, index = nbrs.kneighbors(user_input_df)
        recommended_car = df.iloc[index[0][0]]['car_name']
        return render_template('result.html', car_name=recommended_car)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
