import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class LogisticRegression:

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
            y_pred_linear = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(y_pred_linear)

            #TODO: can have loss calculated here and print it with each iterations

            # Calculating the gradients
            # we need to take transpose here for proper matrix multiplication
            # Also the calculation is same for this as of linear as the dw and db comes out same for 
            # mse and cross_entropy_loss with sigmoid
            dw = (1/n_samples)*np.dot(X.T, (y_pred - Y))
            db = (1/n_samples)*np.sum(y_pred - Y)

            # updating the parameters
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict_probab(self,X):
        # applies sigmoid then gives the output
        y_output_linear = np.dot(X, self.weights) + self.bias
        y_output = sigmoid(y_output_linear)
        return y_output
    
    def predict(self, X, threshold=0.5):
        probab = self.predict_probab(X)
        return (probab >= threshold).astype(int)  # Vectorized instead of list comprehension
    

def main():
    print("=== Testing Your Logistic Regression Implementation ===\n")
    
    # Test 1: Simple dataset
    print("Test 1: Simple 2D Dataset")
    print("-" * 30)
    
    np.random.seed(42)
    X_simple = np.array([
        [1, 2], [2, 1], [2, 3], [3, 2],      # Class 0 (bottom-left)
        [6, 7], [7, 6], [7, 8], [8, 7]       # Class 1 (top-right)
    ])
    y_simple = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    print(f"Training data shape: {X_simple.shape}")
    print(f"Classes: {np.unique(y_simple)}")
    
    # Train model
    model = LogisticRegression(lr=0.1, n_iter=1000)
    model.fit(X_simple, y_simple)
    
    # Test predictions
    probabilities = model.predict_probab(X_simple)
    predictions = model.predict(X_simple)
    
    print(f"\nResults:")
    print(f"Learned weights: {model.weights}")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"Training accuracy: {accuracy_score(y_simple, predictions):.4f}")
    
    print(f"\nPrediction breakdown:")
    print(f"{'Actual':<8} {'Probability':<12} {'Predicted':<10}")
    print("-" * 35)
    for i in range(len(y_simple)):
        print(f"{y_simple[i]:<8} {probabilities[i]:<12.4f} {predictions[i]:<10}")
    
    # Test 2: Larger dataset
    print(f"\n" + "="*50)
    print("Test 2: Larger Random Dataset")
    print("="*50)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Train model
    model2 = LogisticRegression(lr=0.01, n_iter=1200)
    model2.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model2.predict(X_train_scaled)
    test_pred = model2.predict(X_test_scaled)
    test_probab = model2.predict_probab(X_test_scaled)
    
    print(f"\nPerformance:")
    print(f"Training accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}")
    
    # Test 3: Different thresholds
    print(f"\nTest 3: Different Decision Thresholds")
    print("-" * 40)
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Pred Class 1':<15}")
    print("-" * 40)
    
    for thresh in thresholds:
        pred_thresh = model2.predict(X_test_scaled, threshold=thresh)
        acc = accuracy_score(y_test, pred_thresh)
        class_1_count = sum(pred_thresh)
        print(f"{thresh:<10} {acc:<10.4f} {class_1_count:<15}")
    
    # Test 4: Edge cases
    print(f"\nTest 4: Edge Cases")
    print("-" * 20)
    
    # Very confident predictions
    confident_samples = X_test_scaled[:5]  # First 5 test samples
    confident_probab = model2.predict_probab(confident_samples)
    
    print(f"Sample confidence levels:")
    for i, prob in enumerate(confident_probab):
        confidence = max(prob, 1-prob)  # Distance from 0.5
        print(f"Sample {i+1}: P = {prob:.4f}, Confidence = {confidence:.4f}")

if __name__ == "__main__":
    main()


