"""Modeling skeleton for the NFL salary-performance project.

This module provides a light scaffold for experimentation. Fill in the
placeholders below with feature selection, preprocessing, and model training.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score

from clean import load_salary_data, load_game_data, build_modeling_dataset


def load_model_dataset():
	"""Load and return the merged modeling dataset from clean.py."""
	salary = load_salary_data()
	games = load_game_data()
	return build_modeling_dataset(salary, games)


def run_linear_regression(df):
	"""Train and evaluate a linear regression model (Lokesh's section)."""
	# TODO (Lokesh): Implement linear regression training and evaluation.
	pass


def run_logistic_regression(df):
    """Train and evaluate a logistic regression model (Matthew's section)."""
    FEATURE_COLS = ["Offense_P", "Defense_P"]
    TARGET_COL   = "Playoffs"

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred       = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Playoffs", "Made Playoffs"]))

    coefficients = pd.Series(model.coef_[0], index=FEATURE_COLS)
    print("Coefficients:")
    print(coefficients)

    return model, scaler, X, y
	

def plot_decision_boundary(model, scaler, X, y, feature_names):
    x_min, x_max = X.iloc[:, 0].min() * 0.9, X.iloc[:, 0].max() * 1.1
    y_min, y_max = X.iloc[:, 1].min() * 0.9, X.iloc[:, 1].max() * 1.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn")
    colors = y.map({0: "red", 1: "green"})
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors, edgecolors="black",
                linewidths=0.6, s=70, zorder=3)
    plt.legend(handles=[Patch(color="green", label="Made Playoffs"),
                        Patch(color="red",   label="No Playoffs")])
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Logistic Regression — Decision Boundary\n(NFL Playoffs by Salary Spending)")
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150)
    plt.show()


def main():
    df = load_model_dataset()
    if df.empty:
        print("Model dataset is empty. Check data and clean.py.")
        return

    # Call either/both once features and targets are defined.
    run_linear_regression(df)
    run_logistic_regression(df)

    model, scaler, X, y = run_logistic_regression(df)

    plot_decision_boundary(model, scaler, X, y, ["Offense_P", "Defense_P"])


if __name__ == "__main__":
    main()
	