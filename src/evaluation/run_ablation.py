from evaluation.ablation import run_ablation
from data.loader import load_data

def main():
    X_train, X_test, y_train, y_test = load_data()
    df = run_ablation(X_train, X_test, y_train, y_test)
    print(df)

if __name__ == "__main__":
    main()