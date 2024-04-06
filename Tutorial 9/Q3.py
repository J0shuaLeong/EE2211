from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

housing = fetch_california_housing()

X = housing.data[:, housing.feature_names.index("MedInc")]
y = housing.target

class Node:
    def __init__(self, depth, max_depth):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None
    
    def fit(self, X, y):
        if self.depth == self.max_depth or len(np.unique(y)) == 1:
            self.value = np.mean(y)
            return
        
        num_features = X.shape[1]
        best_gain = 0
        best_feature_index = None
        best_threshold = None
        
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                y_left = y[X[:, feature_index] <= threshold]
                y_right = y[X[:, feature_index] > threshold]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                gain = self.calculate_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        if best_gain == 0:
            self.value = np.mean(y)
            return
        
        self.feature_index = best_feature_index
        self.threshold = best_threshold
        
        X_left = X[X[:, self.feature_index] <= self.threshold]
        y_left = y[X[:, self.feature_index] <= self.threshold]
        X_right = X[X[:, self.feature_index] > self.threshold]
        y_right = y[X[:, self.feature_index] > self.threshold]
        
        self.left = Node(self.depth + 1, self.max_depth)
        self.right = Node(self.depth + 1, self.max_depth)
        self.left.fit(X_left, y_left)
        self.right.fit(X_right, y_right)
    
    def calculate_gain(self, y, y_left, y_right):
        mse_parent = np.mean((y - np.mean(y))**2)
        mse_left = np.mean((y_left - np.mean(y_left))**2)
        mse_right = np.mean((y_right - np.mean(y_right))**2)
        return mse_parent - (len(y_left) / len(y)) * mse_left - (len(y_right) / len(y)) * mse_right
    
    def predict(self, x):
        if self.value is not None:
            return self.value
        elif x[self.feature_index] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

class RegressionTree:
    def __init__(self, max_depth=2):
        self.root = Node(depth=0, max_depth=max_depth)
    
    def fit(self, X, y):
        self.root.fit(X, y)
    
    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])
    
custom_tree = RegressionTree(max_depth=2)
custom_tree.fit(X.reshape(-1, 1), y)

regr_1 = DecisionTreeRegressor(criterion='squared_error', max_depth=1)
regr_1.fit(X.reshape(-1, 1), y)

y_pred_custom = custom_tree.predict(X.reshape(-1, 1))
y_pred_sklearn = regr_1.predict(X.reshape(-1, 1))

mse_custom = mean_squared_error(y, y_pred_custom)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)

print(f"MSE for Custom Tree: {mse_custom}")
print(f"MSE for DecisionTreeRegressor: {mse_sklearn}")