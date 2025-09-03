from logic import FederatedLightGBMClient
import json

num_clients = 1
client = FederatedLightGBMClient(config_file="config.yaml",
                                    num_clients=num_clients)
local_preds = client.predict(data=client.test_data,
                             model=client.local_model)
if local_preds is not None and client.test_y is not None:
    metrics = client.evaluate(local_preds, client.test_y)
    metrics_json = metrics.to_json()
    # write json to file
    with open("metrics.json", "w") as f:
        f.write(metrics_json)
    local_preds.to_csv("predictions_local_model.csv")
    client.local_model.save_model("local_model.txt")

