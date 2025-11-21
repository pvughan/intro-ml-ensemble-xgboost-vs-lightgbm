from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    return {"accuracy": acc, "f1_score": f1}
