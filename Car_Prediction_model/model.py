import keras
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.initializers import he_uniform
from keras.metrics import RootMeanSquaredError

class SimpleNeuralNetwork(keras.Model):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rates=None, optimizer=None):
        super(SimpleNeuralNetwork, self).__init__()
        inputs = Input(shape=(input_size,))
        x = inputs

        for i, hidden_size in enumerate(hidden_sizes):
            x = Dense(hidden_size, activation='relu', kernel_initializer=he_uniform())(x)
            # x = BatchNormalization()(x)
            if dropout_rates and i < len(dropout_rates):
                x = Dropout(dropout_rates[i])(x)

        outputs = Dense(output_size, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[RootMeanSquaredError()])

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, callbacks=None):
        # Define callbacks
        checkpoint = ModelCheckpoint("model_checkpoint.h5", save_best_only=True)
        early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

        # Train the model with callbacks
        if X_val is None and y_val is None:
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks
            )
        else:
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks
            )

    def evaluate(self, X_test, y_test):
        # Evaluate the model on test data
        loss, rmse = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, Test RMSE: {rmse:.4f}")

    def get_summary(self):
        # get the summary of the model
        return self.model.summary()

    def save_model(self, model_name):
        # save the model
        self.model.save(model_name)

    def call(self, inputs):
        return self.model(inputs)

