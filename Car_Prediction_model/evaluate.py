import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from Data_preprocessing import X_test, y_test

input_size = 17

modelpath = 'Car_Prediction_model\car_prediction.keras'
model  = load_model(modelpath)

predictions = []
for i in range(len(X_test)):
    x = X_test.iloc[i].values
    x = x.reshape(1, input_size)
    pred  = model.predict(x)
    predictions.append(pred[0])


print(predictions)
plt.plot(predictions, y_test)
plt.show()