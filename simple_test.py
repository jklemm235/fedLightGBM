from logic import FederatedLightGBMClient

num_clients = 1
client = FederatedLightGBMClient(config_file="config.yaml",
                                    num_clients=num_clients)
local_preds = client.predict(data=client.test_data,
                             model=client.local_model)
if local_preds is not None and client.test_y is not None:
    metrics = client.evaluate(local_preds, client.test_y)
if local_preds is not None:
    local_preds.to_csv("predictions_local_model.csv")
