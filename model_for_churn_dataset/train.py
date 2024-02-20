import pandas as pd
from model import SimpleNeuralNetwork
from data import preprocess_data
from Data_preprocessing_utils.Data_preprocessing_util import Spiliting_in_test_nd_train
from Configs.Config import Config

df = pd.read_csv("Churn_Modelling.csv")

preprocessed_data = preprocess_data(df)
X_train, X_test, y_train, y_test = Spiliting_in_test_nd_train(preprocessed_data)

print(X_train.head(5))

config = Config(2, 0.2, [32, 64, 128], 'batch', 'relu', 'glorot_uniform', 'mse', 'adam', 32, 10)
config.save_parameters()
# Save configurations to a DataFrame and CSV file
config.save_to_dataframe_and_csv()

# data_parameter
input_size = 13
output_size = 1

model = SimpleNeuralNetwork(input_size=input_size,
                             hidden_size0=config.get_config('perceptron')[0],
                             hidden_size1=config.get_config('perceptron')[1],
                             hidden_size2=config.get_config('perceptron')[2],
                             output_size=output_size)

print(model.get_summary())