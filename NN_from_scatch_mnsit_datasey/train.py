from function import gradient_descent, forward, get_predictions, test_prediction, make_predictions, get_accuracy
from data import X_train, Y_train, X_dev, Y_dev


if __name__ == '__main__':
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
    test_prediction(0, X_train, Y_train,W1, b1, W2, b2)
    test_prediction(1, X_train, Y_train,W1, b1, W2, b2)
    test_prediction(2, X_train, Y_train,W1, b1, W2, b2)
    test_prediction(3, X_train, Y_train,W1, b1, W2, b2)

    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    get_accuracy(dev_predictions, Y_dev)
