from typing import Dict, Optional, Tuple

import flwr as fl
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
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

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=3,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        #on_evaluate_config_fn=evaluate_config,
        initial_parameters=model.get_weights(),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8080", config={"num_rounds": 8}, strategy=strategy)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    batch_size = 10000  # Set batch size to large number so that all images are processed in one batch

    # Load validation data
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='data/test_data',
        # complete data set can be found here: https://www.kaggle.com/christianvorhemus/industrial-quality-control-of-packages
        image_size=(224, 224),
        shuffle=True,
        seed=123,
        batch_size=batch_size)

    # Convert BatchDatasets to Numpy arrays
    x_val = None
    y_val = None
    for image, label in tfds.as_numpy(val_ds):
        x_val = image
        y_val = label

    # Adjust shape of labels
    y_val = np.reshape(y_val, (-1, 1))

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy of global model on global test data": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 24, perform six rounds of training with four
    local epochs, increase to six local epochs afterwards.
    """
    config = {
        "batch_size": 24,
        "local_epochs": 4 if rnd < 6 else 6,
    }
    return config


# Do not set validation steps for this example as validation data set is already very small
#def evaluate_config(rnd: int):
#    """Return evaluation configuration dict for each round.
#    Perform five local evaluation steps on each client (i.e., use five
#    batches) during rounds one to three, then increase to ten local
#    evaluation steps.
#    """
#    val_steps = 5 if rnd < 4 else 10
#    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
