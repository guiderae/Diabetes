from models.ANN_model import ANNModel
from models.ANN_configurable import ANNModelConfigurable
import joblib as joblib
import numpy as np
import torch
import json

"""
    This class contains but one static method that calculates predictions using data from a 
    user selected csv file and a saved, trained model
"""


class DiabetesPredict:
    """
        Static method that calculates predictions from a saved, trained model.  The prediction data is scaled by the same
        scaler that was used to train the model.
        :param data_frame  dataframe of prediction data
        :param saved_model_name name of the file that contains the saved model params.
        :param saved_scaler_name name of the file that contains the saved scaler params
        :returns prediction results in json format.  Results is a list of dictionaries.
    """
    @staticmethod
    def predictXXXXX(data_frame, saved_model_name, saved_scaler_name):
        ##ann_model = ANNModel()
        #ann_model.load("static/" + saved_model_name)
        ann_model = ANNModel.load_model('static/' + saved_model_name)
        saved_scaler = joblib.load("static/" + saved_scaler_name)
        predict_data_list = data_frame.values.tolist()  #Convert df to list
        predict_data_np = np.array(predict_data_list)  # Convert data from 1d to 2d for use in transform
        predict_data_np_scaled = saved_scaler.transform(predict_data_np)  # Scale the data using saved scaler
        predict_data_scaled_list = predict_data_np_scaled.tolist()  # Put prediction data into a list
        predict_data_tensor = torch.tensor(predict_data_scaled_list)  # Convert input array to tensor
        prediction_value_tensor = ann_model(predict_data_tensor)  # Do prediction.  This returns a tensor
        # Dict for textual display of prediction
        outcomes = {0: 'No Diabetes', 1: 'Diabetes Predicted'}
        # Get probabilities for prediction_value using Softmax
        softmax_calc = torch.nn.Softmax(dim=1)  # Applies along row so that sum of probabilities = 1
        probs = softmax_calc(prediction_value_tensor)  # Returns a tensor of predictions
        # Extract probabilities values from prediction tensor and display on console
        #for i, row in enumerate(probs):
        #    print("Data to Predict: {}".format(predict_data_list[i]))
        #    print("Prediction: {}".format(outcomes[prediction_value_tensor[i].argmax().item()]))
        #    print("No Diabetes Probability: {}".format(probs[i, 0].item()))
        #    print("Diabetes Probability: {}".format(probs[i, 1].item()))
        #    print("_______________________")

        """
        Put results into list of dict:
        [{"pred" : "p1", "nodprob" : a1, "yesdprob" : b1}, {"pred" : "p2", "nodprob" : a2, "yesdprob" : b2}, ......]
        """
        key_list = ["pred", "nodprob", "yesdprob"]
        dict_list = []
        for i, row in enumerate(probs):
            values = [outcomes[prediction_value_tensor[i].argmax().item()],  #Prediction text
                      "{:.2f}%".format(100*probs[i, 0].item()),     # No diab probability
                      "{:.2f}%".format(100*probs[i, 1].item())]     # Yes diab probability
            temp_dict = dict(zip(key_list, values))
            dict_list.append(temp_dict)
        converted_json = json.dumps(dict_list) # convert list of dict to json
        return converted_json

    @staticmethod
    def predict(data_frame, saved_model_name, saved_scaler_name):
        #ann_model = ANNModelConfigurable.load_model('static/' + saved_model_name)
        # ann_model.load("static/" + saved_model_name)
        #ann_model.load('static/' + saved_model_name)
        #ann_model = torch.load('static/' + saved_model_name)
        #ann_model.eval()
        ann_model = ANNModelConfigurable(hidden_layers=[48,24,0,0])
        ann_model.load_state_dict(torch.load('static/' + saved_model_name))
        ann_model.eval()
        saved_scaler = joblib.load("static/" + saved_scaler_name)

        pred_scaled = saved_scaler.transform(data_frame)
        pred_scaled_tensor = torch.FloatTensor(pred_scaled)
        prediction_value_tensor = ann_model(pred_scaled_tensor)

        #predict_data_list = data_frame.values.tolist()  # Convert df to list
        #predict_data_np = np.array(predict_data_list)  # Convert data from 1d to 2d for use in transform
        #predict_data_np_scaled = saved_scaler.transform(predict_data_np)  # Scale the data using saved scaler
        #predict_data_scaled_list = predict_data_np_scaled.tolist()  # Put prediction data into a list
        #predict_data_tensor = torch.tensor(predict_data_scaled_list)  # Convert input array to tensor
        #prediction_value_tensor = ann_model(predict_data_tensor)  # Do prediction.  This returns a tensor

        # Dict for textual display of prediction
        outcomes = {0: 'No Diabetes', 1: 'Diabetes Predicted'}
        # Get probabilities for prediction_value using Softmax
        softmax_calc = torch.nn.Softmax(dim=1)  # Applies along row so that sum of probabilities = 1
        probs = softmax_calc(prediction_value_tensor)  # Returns a tensor of predictions
        # Extract probabilities values from prediction tensor and display on console
        # for i, row in enumerate(probs):
        #    print("Data to Predict: {}".format(predict_data_list[i]))
        #    print("Prediction: {}".format(outcomes[prediction_value_tensor[i].argmax().item()]))
        #    print("No Diabetes Probability: {}".format(probs[i, 0].item()))
        #    print("Diabetes Probability: {}".format(probs[i, 1].item()))
        #    print("_______________________")

        """
        Put results into list of dict:
        [{"pred" : "p1", "nodprob" : a1, "yesdprob" : b1}, {"pred" : "p2", "nodprob" : a2, "yesdprob" : b2}, ......]
        """
        key_list = ["pred", "nodprob", "yesdprob"]
        dict_list = []
        for i, row in enumerate(probs):
            values = [outcomes[prediction_value_tensor[i].argmax().item()],  # Prediction text
                      "{:.2f}%".format(100 * probs[i, 0].item()),  # No diab probability
                      "{:.2f}%".format(100 * probs[i, 1].item())]  # Yes diab probability
            temp_dict = dict(zip(key_list, values))
            dict_list.append(temp_dict)
        converted_json = json.dumps(dict_list)  # convert list of dict to json
        return converted_json
