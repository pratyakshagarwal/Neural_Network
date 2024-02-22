import pandas as pd
from Configs.Config import Config
from Data_preprocessing_util.Data_preprocessing_utils import Spiliting_in_test_nd_train
from data import preprocess_data
from model import SimpleNeuralNetwork

df = pd.read_csv('model_for_churn_dataset\Churn_Modelling.csv')

preprocessed_data = preprocess_data(df, ['Balance', 'EstimatedSalary'])
X_train, y_train, X_test, y_test = Spiliting_in_test_nd_train(preprocessed_data)

print(X_train.head(5))

config = Config(2, 0.2, [32, 64, 128], 'BatchNormalization', 'relu', 'glorot_uniform', 'binary_crossentropy', 'adam', 32, 10)
config.save_parameters()
# Save configurations to a DataFrame and CSV file
config.save_to_dataframe_and_csv()

# data_parameter
input_size = 13
output_size = 1


# model paramter
hidden_dim = config.get_config('Perceptrons')
# print(hidden_dim.iloc[0])
model = SimpleNeuralNetwork(input_size=input_size,
                             hidden_size0=hidden_dim.iloc[0][0],
                             hidden_size1=hidden_dim.iloc[0][1],
                             hidden_size2=hidden_dim.iloc[0][2],
                             output_size=output_size)

print(model.get_summary())
history = model.train(X_train, y_train)
model.evaluate(X_test, y_test)
model.save_model('churn_model.h5')