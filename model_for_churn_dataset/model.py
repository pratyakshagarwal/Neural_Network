import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.initializers import glorot_uniform

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size0, hidden_size1, hidden_size2, output_size, dropout_rate=None):
        # Define the architecture of the neural network
        self.model = Sequential([
            Dense(hidden_size0, input_dim=input_size, activation='relu', kernel_initializer=glorot_uniform()),
            BatchNormalization(),
            Dropout(0.3),

            Dense(hidden_size1, activation='relu', kernel_initializer=glorot_uniform()),
            BatchNormalization(),
            Dropout(0.2),

            Dense(hidden_size2, activation='relu', kernel_initializer=glorot_uniform()),
            BatchNormalization(),
            Dropout(0.1),

            Dense(output_size, activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        # Define callbacks
        checkpoint = ModelCheckpoint("model_checkpoint.h5", save_best_only=True)
        early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

        # Train the model with callbacks
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping, tensorboard]
        )

    def evaluate(self, X_test, y_test):
        # Evaluate the model on test data
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    def get_summary(self):
        self.model.summary()

    def save_model(self, model_name):
        self.model.save(model_name)

# Example usage
if __name__ == "__main__":
    # Assuming you have your data (X_train, y_train, X_val, y_val, X_test, y_test) ready
    input_size = 13
    output_size = 2  # Number of classes in your classification problem

    # parameters
    hidden_size0 = 128
    hidden_size1 = 64
    hidden_size2 = 32

    # Create an instance of the SimpleNeuralNetwork class
    neural_network = SimpleNeuralNetwork(input_size, hidden_size0=hidden_size0, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=output_size)
    print(neural_network.get_summary())
