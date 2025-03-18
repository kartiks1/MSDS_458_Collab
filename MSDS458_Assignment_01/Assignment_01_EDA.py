#!/usr/bin/env python3

# Import required libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def load_data():
    """Load the diamonds dataset."""
    diamonds = sns.load_dataset('diamonds')
    return diamonds

def display_data_info(diamonds):
    """Display information about the dataset."""
    print("\nDataset Info:")
    print(diamonds.info())
    print("\nFirst 5 rows of the dataset:")
    print(diamonds.head())
    
    print("\nBasic Statistics:")
    print(diamonds.describe())

def analyze_categorical_variables(diamonds):
    """Analyze categorical variables in the dataset."""
    categorical_cols = ['cut', 'color', 'clarity']
    
    for col in categorical_cols:
        print(f"\n{col.upper()} Distribution:")
        print(diamonds[col].value_counts())
        
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=diamonds, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{col}_distribution.png')
        plt.close()

def analyze_price_distribution(diamonds):
    """Analyze the distribution of diamond prices."""
    plt.figure(figsize=(12, 6))
    
    # Create a histogram with KDE
    sns.histplot(data=diamonds, x='price', bins=50, kde=True)
    plt.title('Distribution of Diamond Prices')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    plt.close()
    
    # Print price statistics
    print("\nPrice Statistics:")
    print(diamonds['price'].describe())

def analyze_carat_price_relationship(diamonds):
    """Analyze the relationship between carat and price."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=diamonds, x='carat', y='price', alpha=0.5)
    plt.title('Carat vs Price')
    plt.tight_layout()
    plt.savefig('carat_price_relationship.png')
    plt.close()
    
    # Calculate correlation
    correlation = diamonds['carat'].corr(diamonds['price'])
    print(f"\nCorrelation between carat and price: {correlation:.3f}")

def analyze_cut_impact(diamonds):
    """Analyze how cut quality affects price."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=diamonds, x='cut', y='price')
    plt.title('Price Distribution by Cut Quality')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cut_price_relationship.png')
    plt.close()
    
    # Print average price by cut
    print("\nAverage Price by Cut:")
    print(diamonds.groupby('cut')['price'].mean().sort_values(ascending=False))

def analyze_correlations(diamonds):
    """Analyze correlations between numerical variables."""
    numerical_cols = diamonds.select_dtypes(include=[np.number]).columns
    correlation_matrix = diamonds[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def create_model(input_dim):
    """Create and compile the neural network model."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_data(diamonds):
    """Prepare the data for model training."""
    # Separate features and target
    X = diamonds.drop('price', axis=1)
    y = diamonds['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing steps
    numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    categorical_features = ['cut', 'color', 'clarity']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def train_model(X_train, y_train):
    """Train the neural network model."""
    model = create_model(X_train.shape[1])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """Plot the training history of the model."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"\nTest MAE: ${test_mae:.2f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Diamond Prices')
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()

def main():
    # Load the dataset
    diamonds = load_data()
    
    # Display basic information
    display_data_info(diamonds)
    
    # Perform comprehensive EDA
    print("\nPerforming Exploratory Data Analysis...")
    
    # Analyze categorical variables
    analyze_categorical_variables(diamonds)
    
    # Analyze price distribution
    analyze_price_distribution(diamonds)
    
    # Analyze carat-price relationship
    analyze_carat_price_relationship(diamonds)
    
    # Analyze cut impact on price
    analyze_cut_impact(diamonds)
    
    # Analyze correlations
    analyze_correlations(diamonds)
    
    print("\nEDA complete! Check the generated plots for visual insights.")
    
    # Prepare data for model training
    print("\nPreparing data for model training...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(diamonds)
    
    # Train the model
    print("\nTraining the model...")
    model, history = train_model(X_train, y_train)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)
    
    print("\nModel training and evaluation complete! Check the generated plots for results.")

if __name__ == "__main__":
    main() 