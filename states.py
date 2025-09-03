from FeatureCloud.app.engine.app import AppState, app_state
from logic import FederatedLightGBMClient
import os

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('terminal')
        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        clients = self.clients
        num_clients = len(clients)
        input_path = os.path.join("mnt", "input")
        config_path = os.path.join(input_path, "config.yml")
        client = FederatedLightGBMClient(config_file=config_path,
                                         num_clients=num_clients,
                                         input_path=input_path)

        # local prediction/model output
        local_preds = None
        metrics = None
        if client.test_data is not None:
            local_preds = client.predict(data=client.test_data,
                                         model=client.local_model)
            assert local_preds is not None and client.test_y is not None
            metrics = client.evaluate(local_preds, client.test_y)

        else:
            local_preds = client.predict(data=client.data,
                                         model=client.local_model)
            assert local_preds is not None
            metrics = client.evaluate(local_preds, client.y)
        assert local_preds is not None and metrics is not None

        # write local pred/metrics/model
        local_preds.to_csv("predictions_local_model.csv")
        metrics_json = metrics.to_json()
        with open("metrics.json", "w") as f:
            f.write(metrics_json)
        client.local_model.save_model("local_model.txt")

        # TODO: implement the ensembling aka the whole fed learning logic
        return 'terminal'
