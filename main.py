import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_pipeline(file_path):
    print("--- 1. Loading Dataset ---")
    df = pd.read_csv(file_path)
    
    print("\n--- 2. Handling Missing Values ---")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    print("\n--- 3. Feature Engineering ---")
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d-%m-%Y %H:%M')
    df['Month'] = df['Date_Time'].dt.month
    df['Hour'] = df['Date_Time'].dt.hour
    df['DayOfWeek'] = df['Date_Time'].dt.dayofweek
    df = pd.get_dummies(df, columns=['Load_Type'], drop_first=True)
    
    # Save the processed data
    output_path = 'data/ux data/cleaned_load_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n--- Processed data saved successfully to {output_path} ---")
    
    print("\n--- 4. Machine Learning Model Training ---")
    # Drop Date_Time for modeling since models need numerical inputs
    model_df = df.drop(columns=['Date_Time'])
    
    # Define Features (X) and Target (y) - predicting Usage_kWh
    X = model_df.drop(columns=['Usage_kWh'])
    y = model_df['Usage_kWh']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")

if __name__ == "__main__":
    file_path = 'data/ux data/load_data.csv'
    run_pipeline(file_path)