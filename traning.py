import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def model_training():
    df = pd.read_csv('cirrhosis.csv')
    df['Drug'][df['Drug'].isnull()] = df['Drug'].dropna().sample(df['Drug'].isnull().sum()).values
    df['Ascites'][df['Ascites'].isnull()] = df['Ascites'].dropna().sample(df['Ascites'].isnull().sum()).values
    df['Hepatomegaly'][df['Hepatomegaly'].isnull()] = df['Hepatomegaly'].dropna().sample(
        df['Hepatomegaly'].isnull().sum()).values
    df['Spiders'][df['Spiders'].isnull()] = df['Spiders'].dropna().sample(df['Spiders'].isnull().sum()).values

    mean_value = df['Cholesterol'].mean()
    median_value = df['Cholesterol'].median()
    missing_indices = df['Cholesterol'].isnull()
    half_count = int(sum(missing_indices) / 2)
    df['Cholesterol'] = np.where(missing_indices, np.where(np.arange(len(df)) < half_count, mean_value, median_value),
                                 df['Cholesterol'])

    mean_value = df['Copper'].mean()
    median_value = df['Copper'].median()
    missing_indices = df['Copper'].isnull()
    half_count = int(sum(missing_indices) / 2)
    df['Copper'] = np.where(missing_indices, np.where(np.arange(len(df)) < half_count, mean_value, median_value),
                            df['Copper'])

    mean_value = df['Alk_Phos'].mean()
    median_value = df['Alk_Phos'].median()
    missing_indices = df['Alk_Phos'].isnull()
    half_count = int(sum(missing_indices) / 2)
    df['Alk_Phos'] = np.where(missing_indices, np.where(np.arange(len(df)) < half_count, mean_value, median_value),
                              df['Alk_Phos'])

    mean_value = df['SGOT'].mean()
    median_value = df['SGOT'].median()
    missing_indices = df['SGOT'].isnull()
    half_count = int(sum(missing_indices) / 2)
    df['SGOT'] = np.where(missing_indices, np.where(np.arange(len(df)) < half_count, mean_value, median_value),
                          df['SGOT'])

    mean_value = df['Tryglicerides'].mean()
    median_value = df['Tryglicerides'].median()
    missing_indices = df['Tryglicerides'].isnull()
    half_count = int(sum(missing_indices) / 2)
    df['Tryglicerides'] = np.where(missing_indices, np.where(np.arange(len(df)) < half_count, mean_value, median_value),
                                   df['Tryglicerides'])

    df.drop(columns='ID', axis=1, inplace=True)
    df.dropna(inplace=True)

    df['Sex'] = df['Sex'].replace({'F': 1, 'M': 0})
    df['Drug'] = df['Drug'].replace({'D-penicillamine': 1, 'Placebo': 0})
    df['Ascites'] = df['Ascites'].replace({'Y': 1, 'N': 0})
    df['Hepatomegaly'] = df['Hepatomegaly'].replace({'Y': 1, 'N': 0})
    df['Spiders'] = df['Spiders'].replace({'Y': 1, 'N': 0})
    df['Edema'] = df['Edema'].replace({'Y': 1, 'N': 0,'S':2})
    df['Status'] = df['Status'].replace({'D': 0, 'C': 1, 'CL': 2})
    columns = ['Sex','Drug']
    x = df.drop(columns='Status',axis=1)
    y = df['Status']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    rf = RandomForestClassifier(n_estimators=20, criterion='entropy')

    rf.fit(X_train, y_train)

    trained_model = rf

    with open('model.pkl', 'wb') as file:
        pickle.dump(rf, file)
    return trained_model
