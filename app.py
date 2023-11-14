from flask import Flask, render_template, request
from train.train_test import TrainTest
from graph.graph_manager import GraphManager


app = Flask(__name__)

@app.route('/DiabetesTrainConfig')
def landing():
    #train_test_model()
    return render_template('main.html')

@app.route('/TrainAndTest', methods=['POST'])
def train_test_model():
    data_file_name = 'diabetes.csv'
    # Put values of hidden layers into a list.  The values represent how many nodes that are in each
    # hidden layer.  If a hidden layer value is zero, then that layer does not exist. There must be at least
    # 2 hidden layers.
    form_dict = request.form.to_dict()
    hidden_layers_list = [int(form_dict.get("hidden1")), int(form_dict.get("hidden2")), int(form_dict.get("hidden3")),
                         int(form_dict.get("hidden4"))]
    epochs = int(form_dict.get("epochs"))
    learning_rate = float(form_dict.get("learningRate"))
    # TODO:  The following 3 list are for testing purposes.  Delete them for production
    #hidden_layers_list = [24, 12, 0, 0]
    #epochs = 200
    #learning_rate = .0008
    train_test = TrainTest(data_file_name, hidden_layers_list, epochs, learning_rate)
    df_losses = train_test.train()
    return GraphManager.generate_loss_graph(df_losses, hidden_layers_list, learning_rate)


if __name__ == '__main__':
    app.run(port=5020, debug=True)