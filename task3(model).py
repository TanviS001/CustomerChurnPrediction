import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into features (X) and target (y), then into training and test sets.
    """
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler and return the scaler for later use.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_model():
    """
    Build and return the MLPClassifier model with the specified configuration.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        alpha=0.001,
        random_state=42,
        verbose=True
    )
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Train the model, evaluate it on the test data, and print metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    return model

def save_object(obj, filename):
    """
    Save a Python object to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("Modified_Churn_Modelling.csv")
    
    # Split, scale, build model
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = build_model()
    
    # Train and evaluate
    trained_model = train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save the scaler and the trained model
    save_object(scaler, "scaler.pkl")
    save_object(trained_model, "model.pkl")
    print("Model and scaler saved successfully.")
