"""Modeling skeleton for the NFL salary-performance project.

This module provides a light scaffold for experimentation. Fill in the
placeholders below with feature selection, preprocessing, and model training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

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
	# TODO (Matthew): Implement logistic regression training and evaluation.
	pass


def main():
	df = load_model_dataset()
	if df.empty:
		print("Model dataset is empty. Check data and clean.py.")
		return

	# Call either/both once features and targets are defined.
	run_linear_regression(df)
	run_logistic_regression(df)


if __name__ == "__main__":
	main()
