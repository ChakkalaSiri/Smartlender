from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('diabetes_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

    return render_template('innerpage.html', prediction_text=f'Patient is {output}')

if __name__ == "__main__":
    app.run(debug=True)
