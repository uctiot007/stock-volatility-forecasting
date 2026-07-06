import numpy as np
from src.data_loader import get_data
from src.features import prepare_dataset
from src.model import gradient_descent, predict, compute_loss
from src.evaluate import evaluate_model, plot_predictions

if __name__ == "__main__":
    df = get_data()

    X, y, df_final = prepare_dataset(df)

    # Normalize features using TRAIN stats only
    split = int(0.8 * len(X))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    X_train = (X_train_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    # ----- Sweep lambda values to see ridge's effect -----
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]

    print("\n=== RIDGE LAMBDA SWEEP ===")
    results = []
    for lam in lambdas:
        w_l, b_l, _ = gradient_descent(X_train, y_train, lr=0.1, epochs=5000, lambda_=lam)
        y_test_pred_l = predict(X_test, w_l, b_l)
        test_mse_l = compute_loss(y_test, y_test_pred_l)
        results.append((lam, test_mse_l, w_l, b_l))
        print(f"lambda={lam:<8} Test MSE: {test_mse_l:.6f}")

    best_lam, best_mse, w, b = min(results, key=lambda r: r[1])
    print(f"\nBest lambda: {best_lam} (Test MSE: {best_mse:.6f})")

    # ----- Final training run with best lambda -----
    w, b, loss_history = gradient_descent(X_train, y_train, lr=0.1, epochs=5000, lambda_=best_lam)

    print("\nFinal weights:", w)
    print("Final bias:", b)

    # Evaluate
    y_test_pred, y_test_pred_sklearn = evaluate_model(
        X_train, y_train, X_test, y_test, w, b, lambda_=best_lam
    )

    # Plot
    plot_predictions(y_test, y_test_pred, title=f"Custom GD (lambda={best_lam}): Predicted vs Actual Volatility")
    plot_predictions(y_test, y_test_pred_sklearn, title="Sklearn: Predicted vs Actual Volatility")

    # ----- Save trained model artifacts for later prediction -----
    import os
    os.makedirs("outputs", exist_ok=True)
    np.savez(
        "outputs/trained_model.npz",
        w=w, b=b, X_mean=X_mean, X_std=X_std, lambda_=best_lam
    )
    print("\nModel artifacts saved to outputs/trained_model.npz")