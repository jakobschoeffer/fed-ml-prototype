import os
import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Do not set validation steps as validation data set is already very small
        #steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 24)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy of global model on local test data (client 1)": accuracy}


def main() -> None:

    # Load and compile Keras model
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet",
                                                      classes=2)
    base_model.trainable = True

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    # Load a subset of the 'Quality Control of Packages' data set to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_data()

    # Start Flower client
    client = FederatedClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("localhost:8080", client=client)


def load_data():
    """Load data of client 1 and split in train and test data."""

    batch_size = 10000  # Set batch size to large number so that all images are processed in one batch

    # Load training data
    # 400 images in total, 360 for local training + test, 40 for global test of global model
    # 360 images / 3 clients = 120 images per client
    # 120 * 0.8 = 96 images for training per client
    # 120 * 0.2 = 24 images for local test of global model per client
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='./data/client1', # complete data set can be found here: https://www.kaggle.com/christianvorhemus/industrial-quality-control-of-packages
        validation_split=0.2,
        subset="training",
        image_size=(224, 224),
        shuffle=True,
        seed=123,
        batch_size=batch_size)

    # Load validation data
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='./data/client1', # complete data set can be found here: https://www.kaggle.com/christianvorhemus/industrial-quality-control-of-packages
        validation_split=0.2,
        subset="validation",
        image_size=(224, 224),
        shuffle=True,
        seed=123,
        batch_size=batch_size)

    # Convert BatchDatasets to Numpy arrays
    x_train = None
    y_train = None
    for image, label in tfds.as_numpy(train_ds):
        x_train = image
        y_train = label

    x_test = None
    y_test = None
    for image, label in tfds.as_numpy(test_ds):
        x_test = image
        y_test = label

    # Adjust shape of labels
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    main()
