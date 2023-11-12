import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

# Load the Abalone dataset
abalone = pd.read_csv("abalone.csv")

# Label encode the "Type" column (for the "sex" prediction)
label_encoder = LabelEncoder()
abalone['Type'] = label_encoder.fit_transform(abalone['Type'])
print(abalone.head())

print("This is for categories yourself for abalone")

# Split the dataset into train and test sets (default parameter values)
X = abalone.drop('Type', axis=1)  # Features
y = abalone['Type']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the percentage of instances in each output class (sex)
class_counts = abalone['Type'].value_counts()
class_percentages = (class_counts / len(abalone)) * 100

# Plot the percentages
plt.figure(figsize=(8, 6))
class_percentages.plot(kind='bar', color='skyblue')
plt.xlabel('Sex')
plt.ylabel('Percentage')
plt.title('Percentage of Instances in Each Sex')
plt.xticks(rotation=0)  # Assuming you don't need rotation for just three categories

# Save the plot to a file 
plt.savefig("abalone-sex.png")

# Display the plot (optional)
plt.show()

# Create a Decision Tree model with default parameters
clf = DecisionTreeClassifier(max_depth=3)

# Fit the model to the training data
clf.fit(X_train, y_train)

# Evaluate the best model on the test data
y_pred_c = clf.predict(X_test)

######
print("------------------------------------------------------")

# Compute the confusion matrix
confusion_matrix_c = confusion_matrix(y_test, y_pred_c)
print("Confusion Matrix (Decision Tree):")
print(confusion_matrix_c)

# Compute precision, recall, and F1-measure for each class
#Convert class name to string to prevent errors
class_names_str = y.unique().astype(str)
report_c = classification_report(y_test, y_pred_c, target_names=class_names_str)

print("Classification Report (Decision Tree):")
print(report_c)


# Calculate accuracy
accuracy_c = accuracy_score(y_test, y_pred_c)

# Calculate macro-average F1
f1_macro_c = f1_score(y_test, y_pred_c, average='macro')

# Calculate weighted-average F1
f1_weighted_c = f1_score(y_test, y_pred_c, average='weighted')

print("Accuracy (Decision Tree):", accuracy_c)
print("Macro-Average F1 (Decision Tree):", f1_macro_c)
print("Weighted-Average F1 (Decision Tree):", f1_weighted_c)

with open("penguin-performance.txt", "a") as file:
    file.write("This is for categories yourself for penguins\n")
    file.write("------------------------------------------------------\n")
    file.write("(A)")
    file.write("Confusion Matrix (Decision Tree):\n")
    file.write(str(confusion_matrix_c) + "\n")
    file.write("Classification Report (Decision Tree):\n")
    file.write(report_c + "\n")

    file.write(f"Accuracy (Decision Tree): {accuracy_c:.4f}\n")
    file.write(f"Macro-Average F1 (Decision Tree): {f1_macro_c:.4f}\n")
    file.write(f"Weighted-Average F1 (Decision Tree): {f1_weighted_c:.4f}\n")
    file.write("------------------------------------------------------\n")


####



# Plot the Decision Tree graphically
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=class_names_str, rounded=True, fontsize=10)
plt.title('Decision Tree own for abalone')
plt.show()


# Define the parameter grid for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None,5, 10],  # You can choose different values for max_depth
    'min_samples_split': [2, 5, 10]  # You can choose different values for min_samples_split
}

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Create a GridSearchCV object with cross-validation
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, scoring='accuracy', cv=5)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best Decision Tree model
best_dt = grid_search.best_estimator_
best_dt.set_params(max_depth=3)
print("------------------------------------------------------")

# Print the best hyperparameters
print("Best Hyperparameters for Top decision tree:", grid_search.best_params_)

# Evaluate the best model on the test data
y_pred = best_dt.predict(X_test)

######

# Compute the confusion matrix
confusion_matrix_dt = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Top Decision Tree):")
print(confusion_matrix_dt)

# Compute precision, recall, and F1-measure for each class
report_dt = classification_report(y_test, y_pred,  target_names=class_names_str)
print("Classification Report (Top Decision Tree):")
print(report_dt)


# Calculate accuracy
accuracy_dt = accuracy_score(y_test, y_pred)

# Calculate macro-average F1
f1_macro_dt = f1_score(y_test, y_pred, average='macro')

# Calculate weighted-average F1
f1_weighted_dt = f1_score(y_test, y_pred, average='weighted')

print("Accuracy (Top Decision Tree):", accuracy_dt)
print("Macro-Average F1 (Top Decision Tree):", f1_macro_dt)
print("Weighted-Average F1 (Top Decision Tree):", f1_weighted_dt)

with open("penguin-performance.txt", "a") as file:
    file.write("------------------------------------------------------\n")
    file.write("(B)")
    file.write("Confusion Matrix (Top Decision Tree):\n")
    file.write(str(confusion_matrix_dt) + "\n")
    file.write("Classification Report (Top Decision Tree):\n")
    file.write(report_dt + "\n")
    file.write(f"Accuracy (Top Decision Tree): {accuracy_dt:.4f}\n")
    file.write(f"Macro-Average F1 (Top Decision Tree): {f1_macro_dt:.4f}\n")
    file.write(f"Weighted-Average F1 (Top Decision Tree): {f1_weighted_dt:.4f}\n")
    file.write("\nBest Hyperparameters (Top Decision Tree):\n")
    file.write(str(grid_search.best_params_) + "\n")
    file.write("------------------------------------------------------\n")


####


# Plot the best Decision Tree graphically
plt.figure(figsize=(12, 8))
plot_tree(best_dt, filled=True, feature_names=X.columns, class_names=class_names_str, rounded=True, fontsize=10)
plt.title('Best Decision Tree (GridSearchCV) for Abalone')
plt.show()

# Create a Multi-Layered Perceptron (MLP) with the specified parameters
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', random_state=42)

# Fit the MLP to the training data
mlp.fit(X_train, y_train)

y_pred_m = mlp.predict(X_test)
print("------------------------------------------------------")


##########
# Compute the confusion matrix
confusion_matrix_m = confusion_matrix(y_test, y_pred_m)
print("Confusion Matrix (Multi-Layer Perceptron):")
print(confusion_matrix_m)

# Compute precision, recall, and F1-measure for each class
report_m = classification_report(y_test, y_pred_m,  target_names=class_names_str)
print("Classification Report (Multi-Layer Perceptron):")
print(report_m)


# Calculate accuracy
accuracy_m = accuracy_score(y_test, y_pred_m)

# Calculate macro-average F1
f1_macro_m = f1_score(y_test, y_pred_m, average='macro')

# Calculate weighted-average F1
f1_weighted_m = f1_score(y_test, y_pred_m, average='weighted')

print("Accuracy (Multi-Layer Perceptron):", accuracy_m)
print("Macro-Average F1 (Multi-Layer Perceptron):", f1_macro_m)
print("Weighted-Average F1 (Multi-Layer Perceptron):", f1_weighted_m)

with open("penguin-performance.txt", "a") as file:
    file.write("------------------------------------------------------\n")
    file.write("(C)")
    file.write("Confusion Matrix (Multi-Layer Perceptron):\n")
    file.write(str(confusion_matrix_m) + "\n")
    file.write("Classification Report (Multi-Layer Perceptron):\n")
    file.write(report_m + "\n")
    file.write(f"Accuracy (Multi-Layer Perceptron): {accuracy_m:.4f}\n")
    file.write(f"Macro-Average F1 (Multi-Layer Perceptron): {f1_macro_m:.4f}\n")
    file.write(f"Weighted-Average F1 (Multi-Layer Perceptron): {f1_weighted_m:.4f}\n")
    file.write("------------------------------------------------------\n")


# Evaluate the MLP on the test data
accuracy = mlp.score(X_test, y_test)
print("Test Accuracy:", accuracy)

print("------------------------------------------------------")


# Define the parameter grid for GridSearchCV
param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],  # You can choose different network architectures
    'solver': ['adam', 'sgd']
}

# Create an MLP classifier
mlp_classifier = MLPClassifier(random_state=42)

# Create a GridSearchCV object with cross-validation
grid_search = GridSearchCV(estimator=mlp_classifier, param_grid=param_grid, scoring='accuracy', cv=5)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best MLP model
best_mlp = grid_search.best_estimator_


# Use the trained model to make predictions on the test data
y_pred_mlp = best_mlp.predict(X_test)

#####

# Compute the confusion matrix
confusion_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
print("Confusion Matrix (Top Multi-Layer Perceptron):")
print(confusion_matrix_mlp)

# Compute precision, recall, and F1-measure for each class
report_mlp = classification_report(y_test, y_pred_mlp,  target_names=class_names_str)
print("Classification Report (Top Multi-Layer Perceptron):")
print(report_mlp)

# Calculate accuracy
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

# Calculate macro-average F1
f1_macro_mlp = f1_score(y_test, y_pred_mlp, average='macro')

# Calculate weighted-average F1
f1_weighted_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

print("Accuracy (Top Multi-Layer Perceptron):", accuracy_mlp)
print("Macro-Average F1 (Top Multi-Layer Perceptron):", f1_macro_mlp)
print("Weighted-Average F1 (Top Multi-Layer Perceptron):", f1_weighted_mlp)
print("Best Hyperparameters (Top Multi-Layer Perceptron) :", grid_search.best_params_)

with open("penguin-performance.txt", "a") as file:
    file.write("------------------------------------------------------\n")
    file.write("(D)")
    file.write("Confusion Matrix (Top Multi-Layer Perceptron):\n")
    file.write(str(confusion_matrix_mlp) + "\n")
    file.write("Classification Report (Top Multi-Layer Perceptron):\n")
    file.write(report_mlp + "\n")
    file.write(f"Accuracy (Top Multi-Layer Perceptron): {accuracy_mlp:.4f}\n")
    file.write(f"Macro-Average F1 (Top Multi-Layer Perceptron): {f1_macro_mlp:.4f}\n")
    file.write(f"Weighted-Average F1 (Top Multi-Layer Perceptron): {f1_weighted_mlp:.4f}\n")
    file.write(f"Weighted-Average F1 (Top Multi-Layer Perceptron): {f1_weighted_mlp:.4f}\n")
    file.write("\nBest Hyperparameters (Top Multi-Layer Perceptron):\n")
    file.write(str(grid_search.best_params_) + "\n")
    file.write("------------------------------------------------------\n")

##############

# Evaluate the best model on the test data
accuracy = best_mlp.score(X_test, y_test)
print("Test Accuracy:", accuracy)

print("------------------------------------------------------")