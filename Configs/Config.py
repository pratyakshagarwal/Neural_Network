import pandas as pd
from datetime import datetime

class Config:
    def __init__(self, dense_layers, dropout_rate, perceptrons, normalization_used,
                        activation, kernel_initializer, loss, optimizer, batch_size, epochs):
        self.dense_layer = dense_layers
        self.dropout_rate = dropout_rate
        self.perceptron = perceptrons
        self.normalization_used = normalization_used
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.configurations = []

    def save_parameters(self):
        """
        Save neural network configuration.

        Parameters:
        - dense_layers (int): Number of dense layers.
        - dropout_rate (float): Dropout rate.
        - perceptrons (list): List of perceptrons for each layer.
        - normalization_used (str): Type of normalization used.
        - dropout_rate_used (float): Dropout rate used.
        - activation (str): Activation function used.
        - kernel_initializer (str): Kernel initializer used.
        - loss (str): Loss function used.
        - optimizer (str): Optimizer used.
        - batch_size (int): Batch size used.
        - epochs (int): Number of epochs used.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_data = {'Timestamp': timestamp, 'Dense_Layers': self.dense_layer, 'Dropout_Rate': self.dropout_rate,
                       'Perceptrons': self.perceptron, 'Normalization_Used': self.normalization_used,
                       'Activation': self.activation, 'Kernel_Initializer': self.kernel_initializer, 'Loss': self.loss, 'Optimizer': self.optimizer,
                       'Batch_Size': self.batch_size, 'Epochs': self.epochs}

        # Append the configuration to the list
        self.configurations.append(config_data)

    def get_saved_parameters(self):
        return self.configurations

    def save_to_dataframe_and_csv(self, filename='configurations.csv'):
        """
        Save configurations to a DataFrame and then save the DataFrame to a CSV file.

        Parameters:
        - filename (str): Name of the CSV file (default is 'configurations.csv').
        """
        df = pd.DataFrame(self.configurations)
        df.to_csv(filename, index=False)
        print(f"Configurations saved to {filename}")

    def get_config(self, parameter_value, file_name='configurations.csv'):
        df = pd.read_csv("configurations.csv")
        return df[parameter_value]

# Create an instance of the Config class
        
if __name__ == '__main__':
    config = Config(2, 0.2, [64, 128], 'batch', 'relu', 'glorot_uniform', 'mse', 'adam', 32, 10)

    # Save configurations
    config.save_parameters()

    # Save configurations to a DataFrame and CSV file
    config.save_to_dataframe_and_csv()

    # print(config.get_saved_parameters())
    print(config.get_config('Loss'))