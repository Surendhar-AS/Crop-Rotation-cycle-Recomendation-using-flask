from flask import Flask, render_template, request, jsonify
import warnings
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__,static_url_path='/static')
crop_data = pd.read_csv('dataset\Crop Data.csv')

label_encoders = {}
categorical_columns = ['Soil Type', 'Season']
for col in categorical_columns:
    le = LabelEncoder()
    crop_data[col] = le.fit_transform(crop_data[col])
    label_encoders[col] = le

scaler = StandardScaler()
numerical_columns = ['pH', 'Nitrogen (N) ppm', 'Potassium (K) ppm', 'Phosphorus (P) ppm', 'Duration (Months)']
crop_data[numerical_columns] = scaler.fit_transform(crop_data[numerical_columns])

X = crop_data.drop('Crop', axis=1)
Y = crop_data['Crop']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, Y_train)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        try:
            soil_type = request.form['st']
            pH = float(request.form['ph'])
            nitrogen = float(request.form['N'])
            potassium = float(request.form['K'])
            phosphorus = float(request.form['P'])
            starting_season = request.form['S']
            number_of_crops = int(request.form['Nc'])
            duration = int(request.form['Mon'])
            nutrient_loss_data = pd.read_csv('dataset\crop nutrients loss.csv')
            previous_crop = None
            user_input = {
                'Soil Type': soil_type,
                'pH': pH,
                'Nitrogen (N) ppm': nitrogen,
                'Potassium (K) ppm': potassium,
                'Phosphorus (P) ppm': phosphorus,
                'Season': starting_season,
                'Duration (Months)': duration
                }
            for col in categorical_columns:
                le = label_encoders[col]
                user_input[col] = le.transform([user_input[col]])[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
            predicted_crops_list = []
            for i in range(1, number_of_crops + 1):
                if i != 1:
                    if previous_crop is not None:
                        matching_rows = nutrient_loss_data[nutrient_loss_data['Crop'] == previous_crop]
                        if not matching_rows.empty:
                            nutrient_losses = matching_rows.iloc[0]
                            pH_loss = nutrient_losses['pH Loss']
                            nitrogen_loss = nutrient_losses['Nitrogen Loss (ppm)']
                            phosphorus_loss = nutrient_losses['Phosphorus Loss (ppm)']
                            potassium_loss = nutrient_losses['Potassium Loss (ppm)']
                            new_pH = user_input['pH'] - pH_loss
                            new_nitrogen = user_input['Nitrogen (N) ppm'] - nitrogen_loss
                            new_phosphorus = user_input['Phosphorus (P) ppm'] - phosphorus_loss
                            new_potassium = user_input['Potassium (K) ppm'] - potassium_loss

                            user_input = pd.DataFrame({
                                'Soil Type': [user_input['Soil Type']],
                                'pH': [new_pH],
                                'Nitrogen (N) ppm': [new_nitrogen],
                                'Potassium (K) ppm': [new_potassium],
                                'Phosphorus (P) ppm': [new_phosphorus],
                                'Season': [user_input['Season']],
                                'Duration (Months)': [user_input['Duration (Months)']]
                                })

                        else:
                            print(f'No matching nutrient loss data found for Crop: {previous_crop}')
                    else:
                        print('No previous crop data available for the first rotation.')

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    user_input_df = pd.DataFrame(user_input, index=[0])
                    user_input_reshaped = user_input_df.values.reshape(1, -1)  
                    predicted_crops = model.predict(user_input_reshaped)
                    probs = model.predict_proba(user_input_reshaped)
                    sorted_indices = (-probs).argsort(axis=1)
                    if predicted_crops[0] == previous_crop:
                        Next_crop = model.classes_[sorted_indices[0][1]]
                    else:
                        Next_crop = predicted_crops[0]
                    previous_crop = Next_crop
                    predicted_crops_list.append(Next_crop)

        
            prediction_result = {
                'PredictedCrops': predicted_crops_list  
                }
            return jsonify(prediction_result)
        except ValueError as e:
            error_message = str(e)
            return jsonify({'error': error_message})
if __name__ == '__main__':
    app.run(debug=True)
