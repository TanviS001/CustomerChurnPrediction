import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def remove_irrelevant_columns(df):
    """
    Remove columns RowNumber, CustomerId, Surname from the dataframe.
    """
    return df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')

def encode_categorical(df):
    """
    Encode categorical columns (Geography, Gender) using one-hot encoding.
    """
    return pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into features (X) and target (y), then into training and test sets.
    """
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale numerical features in X_train and X_test using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    df = pd.read_csv('Churn_Modelling.csv')
    df = remove_irrelevant_columns(df)
    df = encode_categorical(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("Processed DataFrame (first 5 rows):")
    print(df.head())
    print("\nShape of X_train_scaled:", X_train_scaled.shape)
    print("Shape of X_test_scaled:", X_test_scaled.shape)
