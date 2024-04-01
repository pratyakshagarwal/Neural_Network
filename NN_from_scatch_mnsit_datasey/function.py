import numpy as np
import matplotlib.pyplot as plt

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def onehot(Y):
    onehot_y = np.zeros((Y.size, Y.max() + 1))
    onehot_y[np.arange(Y.size), Y] = 1
    onehot_y = onehot_y.T
    return onehot_y

def backward(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    onehot_y = onehot(Y)
    dZ2 = A2 - onehot_y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, B1, W2, B2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * db1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * db2
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iteration):
    W1, B1, W2, B2 = init_params()
    for i in range(iteration):
          Z1, A1, Z2, A2 = forward(W1, B1, W2, B1, X)
          dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, W1, W2, X, Y)
          W1, B1, W2, B2 = update_params(W1, B1, W2, B2, dW1, db1, dW2, db2, alpha)
          prediction = get_predictions(A2)
          accuracy = get_accuracy(prediction, Y)
         
          if i % 10 == 0:
                 print('i : {} accuracy {}'.format(i, accuracy))
    return W1, B1, W2, B2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, X, Y, W1, b1, W2, b2):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

