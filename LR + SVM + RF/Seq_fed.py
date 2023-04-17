import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from flwr.common import Metrics

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf




NUM_CLIENTS = 4
# accuracy_list = []
# loss_list = []

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    # def get_parameters(self):
    #     """Get parameters of the local model."""
    #     raise Exception("Not implemented (server-side parameter initialization)")
    
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_split=0.1,
        )
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            # "val_loss": history.history["val_loss"][0],
            # "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results 
        # self.model.set_weights(parameters)
        # self.model.fit(self.x_train, self.y_train, epochs=2, verbose=2)
        # return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, precision, recall, auc, f1_score, auprc = self.model.evaluate(self.x_val, self.y_val)
        num_examples_test = len(self.x_val)
        return loss, num_examples_test, len(self.x_val), {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, 'f1_score': f1_score, 'auprc': auprc}
        # loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        # return loss, {"accuracy": acc}
        

# ## Writing results to file
# accuracy_list = np.array(accuracy_list)
# loss_list = np.array(loss_list)
# np.savetxt('accuracy_client_1.txt', accuracy_list)
# np.savetxt('loss_client_1.txt', loss_list)

def client_fn(cid: str) -> fl.client.Client:
    # Load model
    
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        # tfa.metrics.F1Score(name='f1_score', num_classes=5),
        tf.keras.metrics.AUC(name='auprc', curve='PR'), # precision-recall curve
    ]

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax'),
        ]
    )
    #optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    #model.compile(optimizer, "sparse_categorical_crossentropy", metrics=["accuracy"])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)
    
    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    data = np.load('data.npz')
    x = data['x']
    y = data['y']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    partition_size = math.floor(len(x_train) / NUM_CLIENTS)
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    x_train_cid = x_train[idx_from:idx_to]
    y_train_cid = y_train[idx_from:idx_to]

    # Create and return client
    return FlwrClient(model, x_train_cid, y_train_cid)


def main() -> None:
    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 2},
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=1,
            min_evaluate_clients=2,
            min_available_clients=NUM_CLIENTS,
        ),
    )



if __name__ == "__main__":
    main()