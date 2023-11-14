import pandas as pd
from flask import Flask, render_template, request
from prediction.diabetes_predict import DiabetesPredict
app = Flask(__name__)

#model_saved_name = 'PytorchDiabetesModel.pt'
model_saved_name = 'DiabetesModelTrained1.pt'
scaler_saved_filename = 'diabetesScaler.sav'

"""
    Landing page route
"""


@app.route('/DiabetesPrediction')
def load_page():
    return render_template('main.html')


"""
    Route that does an async prediction calculation and appends prediction data to
    an existing data table.
"""


@app.route('/calculatePredictions', methods=['POST'])
def calculate_predictions():
    form_dict = request.form.to_dict()
    # Get name of csv file that contains prediction data
    predict_file_name = form_dict.get('selectedFileName')
    df = pd.read_csv("static/" + predict_file_name)
    prediction_data_json = DiabetesPredict.predict(df, model_saved_name, scaler_saved_filename)
    return prediction_data_json


if __name__ == '__main__':
    app.run(port=5010, debug=True)