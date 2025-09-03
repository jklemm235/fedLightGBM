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
        config_path = os.path.join("mnt", "input", "config.yml")
        client = FederatedLightGBMClient(config_file=config_path,
                                         num_clients=num_clients)
        local_preds = None
        if client.test_data is not None:
            local_preds = client.predict(data=client.test_data,
                                         model=client.local_model)
        else:
            local_preds = client.predict(data=client.data,
                                         model=client.local_model)
        if local_preds is not None:
            local_preds.to_csv("predictions_local_model.csv")
        # TODO: implement the ensembling aka the whole fed learning logic
        return 'terminal'
