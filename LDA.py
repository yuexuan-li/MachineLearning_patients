import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from transformers import ElectraTokenizer, TFElectraForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# Load data
def load_data(patient, file_numbers=[1, 3, 4, 7, 8]):
    dfs = [pd.read_csv(f'/home/myhand/Downloads/data/p{patient}/p{patient}_{i}.csv') for i in file_numbers]
    data = pd.concat(dfs, ignore_index=True)
    selected_columns = ['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5','emg6', 'emg7']
    data = data[selected_columns]
    return data

train_patients = [1, 3, 4, 7, 8]
test_patients = [1, 3, 4, 7, 8]

train_files = ['111', '121', '131', '141']
test_files = ['112', '122', '132', '142']

# Preprocessing
def preprocess_data(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    imp = IterativeImputer()
    X_imputed = imp.fit_transform(X)
    return X_imputed, y

# Train classifiers
def train_classifiers(X_imputed, y_train):
    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_imputed, y_train)

    return lda_clf

def evaluate_classifiers(classifiers, X_imputed, y_test):
    lda_clf = classifiers
    lda_score = lda_clf.score(X_imputed, y_test)

    return lda_score

# Main loop
lda_scores = []

for p in test_patients:
    print('Testing on patient', p)
    X_test, y_test = preprocess_data(load_data(p, test_files))
    for q in train_patients:
        if q != p:
            print('Training on patient', q)
            X_train, y_train = preprocess_data(load_data(q, train_files))
            lda_clf = train_classifiers(X_train, y_train)
            score = evaluate_classifiers(lda_clf, X_test, y_test)
            lda_scores.append(score)

print('LDA accuracy:', np.mean(lda_scores))
