import math
from typing import Optional, Tuple, Dict, List

import yaml
import pandas as pd
import numpy as np
import hashlib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

from helper import Config, Metrics

class FederatedLightGBMClient:
    def __init__(self, config_file: str, num_clients: int):
        print(f"Initializing lightGBM Client for training with {num_clients} clients")
        self.num_clients = num_clients
        # Load config
        try:
            with open(config_file, 'r') as file:
                config_dict = yaml.safe_load(file)
            self.config = Config(**config_dict['boosting_tree'])
        except Exception as e:
            raise ValueError(f"Error reading config file: {e}") from e

        # load data
        trainfile = self.config.trainfile
        testfile = self.config.testfile
        self.data, self.y = self.read_csv_helper(trainfile)
        if testfile:
            self.test_data, self.test_y = self.read_csv_helper(testfile)
        else:
            self.test_data = None
            self.test_y = None

        # lighgbm specifics
        # Hash/sanitize feature names for LightGBM and keep the mapping.
        X_train_hashed, feature_map, hashed_columns = self._hash_features(self.data)
        self._feature_map = feature_map
        self._hashed_columns = hashed_columns

        self.lgb_data = lgb.Dataset(X_train_hashed, label=self.y)
        params = {
            'objective': self.config.mode,
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_iterations': math.ceil(self.config.num_estimator/self.num_clients),
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }

        self.local_model = lgb.train(
            params=params,
            train_set=self.lgb_data,
        )

    def predict(self, data: Optional[pd.DataFrame], model: lgb.Booster) -> Optional[pd.DataFrame]:
        """
        Predict with the given model.
        """
        if data is not None:
            # rename test frame columns to hashed names used at training
            data_hashed = data.rename(columns=self._feature_map)
            # ensure same column order as training
            try:
                data_hashed = data_hashed[self._hashed_columns]
            except Exception:
                # if columns mismatch, let LightGBM raise a clearer error
                pass
            preds = model.predict(data_hashed)
            # attach predictions to original test frame (not the hashed one)
            data["prediction"] = preds
            return data
        else:
            return None

    def evaluate(self, df: pd.DataFrame, y: pd.Series) -> Metrics:
        """
        Given a dataframe with predictions in the predictions column and
        true labels in the label column, compute and print evaluation metrics.
        """
        if "prediction" not in df.columns:
            raise ValueError(f"Dataframe must contain 'prediction' column: {df.head()}")

        # Create dict with metrics
        y_true = y.values
        y_pred_raw = df["prediction"].values

        # If regression mode, compute RMSE and return just that
        if self.config.mode == "regression":
            try:
                y_true_f = np.asarray(y_true).astype(float)
                y_pred_f = np.asarray(y_pred_raw).astype(float)
            except Exception:
                raise ValueError("Regression mode requires numeric true labels and numeric predictions")
            if len(y_true_f) != len(y_pred_f):
                raise ValueError(f"Predictions and labels have different lengths: {len(y_pred_f)} vs {len(y_true_f)}")
            rmse = float(math.sqrt(mean_squared_error(y_true_f, y_pred_f)))
            print(f"Regression RMSE: {rmse}")
            return Metrics(float('nan'), float('nan'), float('nan'), rmse)

        # Classification: convert predictions to discrete labels
        first = y_pred_raw[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            raise ValueError("Multiclass classification not supported yet")
        else:
            arr = np.asarray(y_pred_raw)
            if self.config.mode == "binary" and np.issubdtype(arr.dtype, np.floating):
                y_pred = (arr >= 0.5).astype(int)
            else:
                try:
                    y_pred = arr.astype(int)
                except Exception:
                    y_pred = arr

        # Ensure same length
        y_pred = np.asarray(y_pred)
        y_pred_raw = np.asarray(y_true)
        if len(y_pred) != len(y_pred_raw):
            raise ValueError(f"Predictions and labels have different lengths: {len(y_pred)} vs {len(y_pred_raw)}")

        accuracy = float(accuracy_score(y_pred_raw, y_pred))
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_pred_raw, y_pred, average='macro', zero_division=0)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}, Accuracy: {accuracy}")
        return Metrics(float(precision), float(recall), float(f1_score), float(accuracy))


    def read_csv_helper(self, csv_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        seperator = self.config.seperator
        label_column = self.config.label_column
        id_column = self.config.id_column
        data = pd.read_csv(
            csv_file,
            sep=seperator,
            index_col=id_column,
        )
        print(f"Read in {len(data)} rows from {csv_file}")
        # remove all rows with nan values in the label column
        data = data.dropna(subset=[label_column])
        print(f"{len(data)} rows remaining after dropping rows with NaN in label column")
        # With NaN values pandas fucks up the typing to Object for any boolean columns
        for col in data.columns:
            if data[col].dtype == object:
                if sorted(data[col].dropna().unique().tolist()) == sorted([True, False]):
                    # translate to float (1 is true)
                    print(f"Column {col} has boolean values, transforming to 0,1 for compatability")
                    data[col] = data[col].astype(float)
                else:
                    raise ValueError(f"Column {col} has object type with non-boolean values: {data[col].unique().tolist()}. Please convert to numeric.")
                    # TODO: support categorical values

        y = data.pop(label_column)
        return data, y

    def _hash_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
        """Return (hashed_df, feature_map, hashed_columns).

        feature_map maps original column -> hashed safe column name.
        """
        feature_map: Dict[str, str] = {}
        for col in df.columns:
            h = hashlib.sha1(col.encode('utf-8')).hexdigest()[:32]
            feature_map[col] = f"f_{h}"
        hashed = df.rename(columns=feature_map)
        return hashed, feature_map, hashed.columns.tolist()
