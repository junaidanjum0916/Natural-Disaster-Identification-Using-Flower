
import tensorflow as tf
from typing import Dict, Optional, Tuple

import flwr as fl


# Start Flower server
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=1,
    min_evaluate_clients=2,
)


fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
)