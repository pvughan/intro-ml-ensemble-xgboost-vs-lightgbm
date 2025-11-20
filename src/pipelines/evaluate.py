from sklearn.metrics import accuracy_score
from src.data.loader import load_dataset
from src.models.io import load_model

def evaluate(model_path):
    model = load_model(model_path)
    _, X_test, _, y_test = load_dataset()
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
