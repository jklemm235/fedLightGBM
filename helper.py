from typing import Optional

class Config:
    """Configuration for training/evaluation.

    Fields:
      - trainfile: path to training CSV
      - testfile: optional path to test CSV
      - seperator: CSV separator (note: intentionally 'seperator' to match existing callers)
      - label_column: name of the label column
      - id_column: optional id column name
      - num_estimator: number of estimators
      - mode: Mode.CLASSIFICATION or Mode.REGRESSION
    """
    trainfile: str
    testfile: Optional[str] = None
    seperator: str = ','
    label_column: str = 'label'
    id_column: Optional[str | int] = None
    num_estimator: int = 100
    mode: str = "classification"

    def __init__(self,
                 trainfile: str,
                 testfile: str,
                 seperator: str = ',',
                 label_column: str = 'label',
                 id_column: Optional[str] = None,
                 num_estimator: int = 100,
                 mode: str = "classification") -> None:
        self.trainfile = trainfile
        self.testfile = testfile
        self.seperator = seperator
        self.label_column = label_column
        self.id_column = id_column
        self.num_estimator = num_estimator
        if mode not in ["multiclass", "regression", "binary"]:
            raise ValueError("Mode must be either 'multiclass', 'regression', or 'binary'")
        self.mode = mode

class Metrics:
    """
    Metrics automatically calculated by the app
    """
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    def __init__(self, precision: float, recall: float, f1_score: float, accuracy: float) -> None:
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.accuracy = accuracy
