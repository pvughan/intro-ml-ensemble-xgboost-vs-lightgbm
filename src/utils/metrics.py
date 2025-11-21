def print_metrics(title, metrics):
    print(f"\n--- {title} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
