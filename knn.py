import numpy as np
from collections import Counter
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class Knn:

    def __init__(self, k):
        self.k = k

    def fit(self,X,Y):
        """This is kind of lazy learning so, mostly it's just storing data and calculating the distances"""
        self.X_train = X
        self.Y_train = Y
        

    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self,x):

        # get the distances for each points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the indices of the distances for getting corresponding y_labels
        pred_indices = np.argsort(distances)[:self.k]
        pred_label = [self.Y_train[i] for i in pred_indices]

        # get the most common label
        most_common_tuple = Counter(pred_label).most_common()

        return int(most_common_tuple[0][0])



def main():
    print("=== Testing KNN Implementation ===\n")
    
    # Test 1: Simple 2D dataset
    print("Test 1: Simple 2D Dataset")
    print("-" * 30)
    
    # Create simple training data
    X_train = np.array([
        [1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 7]
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])  # Two classes: 0 and 1
    
    # Test points
    X_test = np.array([
        [2, 2],    # Should be class 0 (close to first group)
        [7, 6],    # Should be class 1 (close to second group)
        [4, 4]     # Border case
    ])
    
    # Create and train KNN
    knn = Knn(k=3)
    knn.fit(X_train, y_train)
    
    # Make predictions
    predictions = knn.predict(X_test)
    
    print(f"Training data: {X_train.tolist()}")
    print(f"Training labels: {y_train.tolist()}")
    print(f"Test points: {X_test.tolist()}")
    print(f"Predictions: {predictions}")
    print()
    
    # Test 2: Iris Dataset
    print("Test 2: Iris Dataset")
    print("-" * 20)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Test different k values
    k_values = [1, 3, 5, 7]
    
    for k in k_values:
        knn = Knn(k=k)
        knn.fit(X_train, y_train)
        
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"k={k}: Accuracy = {accuracy:.3f}")
    
    

if __name__ == "__main__":
    main()