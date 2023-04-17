import flwr as fl


if __name__ == "__main__":

    # Define strategy
    strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
        # followings are used when only 1 client
        # min_fit_clients=1,
        # min_eval_clients=1,
        # min_available_clients=1,
    )

    # Start server
    fl.server.start_server(
    server_address="localhost:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=5)
)