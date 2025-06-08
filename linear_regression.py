import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:

    def __init__(self, lr = 0.001, n_iter=1000):
        self.n_iter = n_iter
        self.lr = lr
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias

            #TODO: can have loss calculated here and print it with each iterations

            # Calculating the gradients
            # we need to take transpose here for proper matrix multiplication
            dw = (1/n_samples)*np.dot(X.T, (y_pred - Y))
            db = (1/n_samples)*np.sum(y_pred - Y)

            # updating the parameters
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self,X):
        return np.dot(X, self.weights) + self.bias
    


def main():
    print("=== Testing Linear Regression Implementation ===\n")
    
    # Test 1: Simple 1D dataset
    print("Test 1: Simple 1D Linear Relationship")
    print("-" * 40)
    
    # Create simple data: y = 2x + 3 + noise
    np.random.seed(42)
    X_simple = np.array([[1], [2], [3], [4], [5], [6]])
    y_simple = 2 * X_simple.flatten() + 3 + np.random.normal(0, 0.1, 6)
    
    print(f"Input X: {X_simple.flatten()}")
    print(f"Target y: {y_simple}")
    print(f"True relationship: y = 2x + 3\n")
    
    # Train model
    model1 = LinearRegression(lr=0.01, n_iter=1000)
    model1.fit(X_simple, y_simple)
    
    # Make predictions
    predictions1 = model1.predict(X_simple)
    
    print(f"Learned weights: {model1.weights}")
    print(f"Learned bias: {model1.bias:.4f}")
    print(f"Predictions: {predictions1}")
    print(f"MSE: {mean_squared_error(y_simple, predictions1):.4f}")
    print(f"R² Score: {r2_score(y_simple, predictions1):.4f}\n")
    
    # Test 2: Multi-dimensional dataset
    print("Test 2: Multi-dimensional Dataset")
    print("-" * 34)
    
    # Generate synthetic dataset
    X_multi, y_multi = make_regression(
        n_samples=100, 
        n_features=3, 
        noise=10, 
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples\n")
    
    # Train model
    model2 = LinearRegression(lr=0.001, n_iter=1000)
    model2.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model2.predict(X_train)
    test_pred = model2.predict(X_test)
    
    print(f"Learned weights: {model2.weights}")
    print(f"Learned bias: {model2.bias:.4f}")
    print(f"Training MSE: {mean_squared_error(y_train, train_pred):.4f}")
    print(f"Test MSE: {mean_squared_error(y_test, test_pred):.4f}")
    print(f"Training R²: {r2_score(y_train, train_pred):.4f}")
    print(f"Test R²: {r2_score(y_test, test_pred):.4f}\n")
    
    # Test 3: Perfect linear relationship
    print("Test 3: Perfect Linear Relationship (No Noise)")
    print("-" * 45)
    
    # Perfect relationship: y = 3x₁ + 2x₂ + 1
    X_perfect = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_perfect = 3 * X_perfect[:, 0] + 2 * X_perfect[:, 1] + 1
    
    print(f"True relationship: y = 3x₁ + 2x₂ + 1")
    print(f"Input X:\n{X_perfect}")
    print(f"Target y: {y_perfect}\n")
    
    # Train model
    model3 = LinearRegression(lr=0.01, n_iter=800)
    model3.fit(X_perfect, y_perfect)
    
    # Make predictions
    predictions3 = model3.predict(X_perfect)
    
    print(f"Learned weights: {model3.weights} (should be close to [3, 2])")
    print(f"Learned bias: {model3.bias:.4f} (should be close to 1)")
    print(f"Predictions: {predictions3}")
    print(f"Actual: {y_perfect}")
    print(f"MSE: {mean_squared_error(y_perfect, predictions3):.6f} (should be very small)")
    
    # Test 4: Different learning rates
    print("\nTest 4: Effect of Different Learning Rates")
    print("-" * 42)
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    for lr in learning_rates:
        model_lr = LinearRegression(lr=lr, n_iter=500)
        model_lr.fit(X_simple, y_simple)
        pred_lr = model_lr.predict(X_simple)
        mse_lr = mean_squared_error(y_simple, pred_lr)
        print(f"LR = {lr:6.4f}: Final MSE = {mse_lr:.6f}, Weights = {model_lr.weights}, Bias = {model_lr.bias:.4f}")

if __name__ == "__main__":
    main()