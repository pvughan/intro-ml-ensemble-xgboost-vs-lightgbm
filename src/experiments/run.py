from src.pipelines.train import train
from src.pipelines.evaluate import evaluate

if __name__ == '__main__':
    model = train()
    acc = evaluate("model_xgboost.pkl")
    print("Accuracy:", acc)
