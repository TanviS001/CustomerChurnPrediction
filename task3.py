from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example: list of features in the order used during training
FEATURE_COLUMNS = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain',
    'Gender_Male'
]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    prediction_proba = None
    
    if request.method == 'POST':
        # Retrieve input values from the form
        input_data = {
            'CreditScore': float(request.form['credit_score']),
            'Age': int(request.form['age']),
            'Tenure': int(request.form['tenure']),
            'Balance': float(request.form['balance']),
            'NumOfProducts': int(request.form['num_of_products']),
            'HasCrCard': int(request.form['has_cr_card']),
            'IsActiveMember': int(request.form['is_active_member']),
            'EstimatedSalary': float(request.form['estimated_salary']),
            # One-hot encoding for Geography
            'Geography_Germany': 1 if request.form['geography'] == 'Germany' else 0,
            'Geography_Spain': 1 if request.form['geography'] == 'Spain' else 0,
            # One-hot encoding for Gender
            'Gender_Male': 1 if request.form['gender'] == 'Male' else 0
        }

        # Create DataFrame in correct column order
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        # Scale numerical features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]  # probability of churn

    return render_template('index.html', prediction=prediction, prediction_proba=prediction_proba)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
