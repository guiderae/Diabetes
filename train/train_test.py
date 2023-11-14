import pandas as pd
from sklearn.model_selection import train_test_split
import joblib as joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from model.ANN_configurable import ANNModelConfigurable
"""
Optimal parameters:  
    hidden_layers_list = [48,24,0,0]
    learning_rate = 0.015
    
    Gives loss = 0.300
    Gives Accuracy = 0.797
"""

class TrainTest:
    def __init__(self, data_file_name, hidden_layers_list, epochs, learning_rate):
        self.layers_list = hidden_layers_list
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.data_file_name = data_file_name
        self.df_losses = None

    def train(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor =  \
            self.scale_to_tensors(X_train, X_test, y_train, y_test)
        torch.manual_seed(20)  # Seed with same number each time model is trained for reproducibility

        model = ANNModelConfigurable(self.layers_list, self.epochs, self.learning_rate)
        # Backward Propagation - loss and optimizer
        loss_function = nn.CrossEntropyLoss()  # CrossEntropyLoss also used in Tensorflow
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)  # note Tensorflow also uses Adam
        final_losses = []
        for i in range(self.epochs):
            y_pred = model.forward(X_train_tensor)
            loss = loss_function(y_pred, y_train_tensor)
            final_losses.append({"epoch" : i, "loss" : loss.item()})
            if i % 10 == 0:
                print("Epoch number: {} and the loss : {}".format(i, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.df_losses = pd.DataFrame(final_losses)
        # Accuracy - comparing the results from the test data
        predictions = []
        with torch.no_grad():
            for i, data in enumerate(X_test_tensor):
                y_pred = model(data)
                #print("y_pred: {}  argmax: {}   item: {}".format(y_pred, y_pred.argmax(), y_pred.argmax().item()))
                # predictions.append(y_pred.argmax().item())
                predictions.append(y_pred.argmax())

        score = accuracy_score(y_test_tensor, predictions)  # Simply calculates number of hits / length of y_test
        print("Final Test Accuracy: {}\n\n".format(score))
        model.save('static/DiabetesModelTrained1.pt')
        #torch.save(model, 'static/DiabetesModelTrained1.pt')
        return self.df_losses

    def prepare_data(self ):
        df = pd.read_csv("static/" + self.data_file_name)
        # Remove all rows that contain a zero in any of the columns specified below
        df = df.loc[(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
                         'Age']] != 0).all(axis=1)]
        y = df['Outcome']
        X = df.drop('Outcome', axis=1)  # axis = 1 is column
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def scale_to_tensors(self, X_train, X_test, y_train, y_test):
        scaler = MinMaxScaler()

        # MinMaxScaler.fit() calculates scaling parameters while MinMaxScaler.transform()
        # scales the data using those scaling parameters determined by fit()
        scaler.fit(X_train)  # Calculates scaling params based on training data
        joblib.dump(scaler, 'static/diabetesScaler.sav')  # Save scaling params of training data
        # Transform training and test data.
        X_train_scaled = scaler.transform(X_train)  # Scale training data
        X_test_scaled = scaler.transform(X_test)  # Scale test data based on scaling params of training data
        # Creating Tensors (multidimensional matrix) x-input data  y-output data
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.LongTensor(
            y_train.values)  # y train/test values are not scaled since their values are either 0 or 1.
        y_test_tensor = torch.LongTensor(y_test.values)
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
