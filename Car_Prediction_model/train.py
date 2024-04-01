import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from Configs.Config import Config
from sklearn.model_selection import train_test_split
from model import SimpleNeuralNetwork
from Data_preprocessing import X_train, y_train, X_test, y_test
from Custom_callbacks.callbacks_utils import CustomCSVLogger, CustomEarlyStopping, CustomReduceLROnPlateau

# model Parameters
dim1 = 128
dim2 = 256
batch_size = 8
epochs = 30

# Create an instance of the Adam optimizer with specified parameters
adam_optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
csvcallback  = CustomCSVLogger.make_callback(filename='Car_Prediction_model\logs.csv')
early_stopping = CustomEarlyStopping.make_callback(patience=5, monitor='val_loss')
lr_schduler = CustomReduceLROnPlateau.make_callback(monitor='val_loss')

callbacks = [csvcallback, early_stopping, lr_schduler]

# configuration file path
filepath = 'Car_Prediction_model\configurations.csv'

config = Config(2, [0.0, 0.0, 0.0], [dim1, dim2], 'BatchNormalization', 'relu', 'he_uniform', 'mean_absolute_error', f'{adam_optimizer}', batch_size=batch_size, epochs=epochs)
config.save_parameters()
# Save configurations to a DataFrame and CSV file
config.save_to_dataframe_and_csv(filename=filepath)

# data_parameter
input_size = 17
output_size = 1

# model paramter
hidden_dim = config.get_config('Perceptrons', file_name=filepath)
# print(hidden_dim.iloc[0])
model = SimpleNeuralNetwork(input_size=input_size,
                             hidden_sizes=[hidden_dim.iloc[0][0], hidden_dim.iloc[0][1]],
                             output_size=output_size,
                             optimizer=adam_optimizer)

print(model.get_summary())
history = model.train(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
model.evaluate(X_test, y_test)
model.save_model('Car_Prediction_model\car_prediction.keras')

