import base64
from io import BytesIO

#import matplotlib.pyplot as plt
from matplotlib import pyplot


class GraphManager:

    @staticmethod
    def generate_loss_graph(loss_df, hidden_layer_list_nodes, learning_rate):
        epochs_list = loss_df['epoch'].tolist()
        loss_list = loss_df['loss'].tolist()
        fig = pyplot.figure()
        fig.set_figheight(4)
        fig.set_figwidth(6)
        pyplot.xlabel("Epoch")
        pyplot.ylabel("Cross Entropy Loss")
        pyplot.plot(epochs_list, loss_list)
        title = 'Node1: ' + str(hidden_layer_list_nodes[0]) + " Node2: " + str(hidden_layer_list_nodes[1]) + \
                 ' Node3: ' +  str(hidden_layer_list_nodes[2]) + ' Node4: ' + str(hidden_layer_list_nodes[3]) + \
                 ' Learn Rate: ' + str(learning_rate)
        pyplot.title(title)
        encoded_graph = GraphManager._build_encoded_HTML(pyplot)
        return encoded_graph

    """
        Private method
        :returns: The base64 encoded graph image string 
        """
    @staticmethod
    def _build_encoded_HTML(ax):
        buffer = BytesIO()
        ax.savefig(buffer, format="png")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return encoded
