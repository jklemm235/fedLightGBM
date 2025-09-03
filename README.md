# Config
```
boosting_tree:
  trainfile: "example_data/train.csv"
  testfile: "example_data/test.csv"
  seperator: ","
  label_column: "C0920688@Yes_No:boolean" # cancer diagnosis
  id_column: "PatientID"
  num_estimator: 100
  mode: "binary"

```
# Info
Mini app using lighgbm boosting trees.
Right now doesnt implement federated learning TODO