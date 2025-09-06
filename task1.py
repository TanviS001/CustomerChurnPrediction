import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def check_missing_values(df):
    return df.isnull().sum()

def churn_balance(df):
    total = df.shape[0]
    churned = df[df['Exited'] == 1].shape[0]
    non_churned = total - churned
    churn_rate = churned / total
    return {
        "total": total,
        "churned": churned,
        "non_churned": non_churned,
        "churn_rate": churn_rate
    }

def descriptive_statistics(df):
    return df.describe()

if __name__ == "__main__":
    df = load_data("Churn_Modelling.csv")
    
    missing_values = check_missing_values(df)
    print("Missing values in each column:")
    print(missing_values)

    print("\n")
    
    churn_stats = churn_balance(df)
    print(f"Total customers: {churn_stats['total']}")
    print(f"Churned customers: {churn_stats['churned']}")
    print(f"Stayed customers: {churn_stats['non_churned']}")
    print(f"Churn rate: {churn_stats['churn_rate']:.2%}")

    print("\n")
    
    stats = descriptive_statistics(df)
    print("Descriptive statistics for numerical features:")
    print(stats)
