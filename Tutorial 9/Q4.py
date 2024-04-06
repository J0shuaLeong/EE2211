from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

class_tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
class_tree.fit(X_train, y_train)

y_train_pred = class_tree.predict(X_train)
y_test_pred = class_tree.predict(X_test)

y_train_acc = accuracy_score(y_train, y_train_pred)
y_test_acc = accuracy_score(y_test, y_test_pred)

print(f"Training accuracy: {y_train_acc}")
print(f"Test accuracy: {y_test_acc}")

class_names_list = iris.target_names.tolist()

plt.figure(figsize=(12, 8))
tree.plot_tree(class_tree, feature_names=iris.feature_names, class_names=class_names_list, filled=True)
plt.show()