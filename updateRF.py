import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load data
def load_data(patient, file_numbers=[1, 3, 4, 7, 8]):
    dfs = [pd.read_csv(f'/home/myhand/Downloads/data/p{patient}/p{patient}_{i}.csv') for i in file_numbers]
    data = pd.concat(dfs, ignore_index=True)
    selected_columns = ['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5','emg6', 'emg7']
    data = data[selected_columns + ['label']]
    return data

# Preprocessing data
def preprocess_data(data, past_time_steps=0):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    imp = IterativeImputer()
    X_imputed = imp.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_final = X_scaled
    if past_time_steps > 0:
        X_padded = np.pad(X_final, ((past_time_steps, 0), (0, 0)), 'constant')
        for i in range(past_time_steps):
            X_final = np.concatenate([X_final, X_padded[i:i-X_final.shape[0]]], axis=1)
    return X_final, y

# Select RF parameters with cross validation
def select_rf_params(X_train, y_train):
    rf_clf = RandomForestClassifier()
    param_grid = {'n_estimators': [10, 50, 100, 200],
                  'max_depth': [5, 10, 20, None]}
    grid_search = GridSearchCV(rf_clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# Train classifiers
def train_classifiers(X_train, y_train, rf_params):
    rf_clf = RandomForestClassifier(**rf_params)
    rf_clf.fit(X_train, y_train)
    return rf_clf

# Evaluate classifiers
def evaluate_classifiers(classifiers, X_test, y_test):
    rf_clf = classifiers
    rf_score = rf_clf.score(X_test, y_test)
    return rf_score

# Main loop
rf_scores = []

for p in test_patients:
    print('Testing on patient', p)
    X_test, y_test = preprocess_data(load_data(p, test_files), past_time_steps=1)
    for q in train_patients:
        if q != p:
            print('Training on patient', q)
            X_train, y_train = preprocess_data(load_data(q, train_files), past_time_steps=1)
            rf_params = select_rf_params(X_train, y_train)
            rf_clf = train_classifiers(X_train, y_train, rf_params)
            score = evaluate_classifiers(rf_clf, X_test, y_test)
            rf_scores.append(score)

print('Random Forest accuracy:', np.mean(rf_scores))
