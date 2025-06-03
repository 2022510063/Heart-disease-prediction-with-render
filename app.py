from flask import Flask, render_template, request
import pickle
import joblib

app = Flask(__name__)

# Load your model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        Age = int(request.form['Age'])
        Sex = request.form['Sex']
        ChestPainType = request.form['ChestPainType']
        RestingBP = int(request.form['RestingBP'])
        Cholesterol = int(request.form['Cholesterol'])
        FastingBS = int(request.form['FastingBS'])
        RestingECG = request.form['RestingECG']
        MaxHR = int(request.form['MaxHR'])
        ExerciseAngina = request.form['ExerciseAngina']
        Oldpeak = float(request.form['Oldpeak'])
        ST_Slope = request.form['ST_Slope']

        # Mappings
        mapping = {
            "Sex": {"M": 1, "F": 0},
            "ChestPainType": {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3},
            "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
            "ExerciseAngina": {"N": 0, "Y": 1},
            "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2}
        }

        # Apply mapping
        Sex = mapping["Sex"][Sex]
        ChestPainType = mapping["ChestPainType"][ChestPainType]
        RestingECG = mapping["RestingECG"][RestingECG]
        ExerciseAngina = mapping["ExerciseAngina"][ExerciseAngina]
        ST_Slope = mapping["ST_Slope"][ST_Slope]

        # Prepare features for prediction
        features = [Age, Sex, ChestPainType, RestingBP, Cholesterol,
                    FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]

        # Predict
        prediction = model.predict([features])[0]
        prediction_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
