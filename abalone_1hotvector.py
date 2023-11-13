import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
import numpy as np
# Loading the Abalone dataset
abalone = pd.read_csv("abalone.csv")

# Performing one-hot encoding on the "Type" column
abalone = pd.get_dummies(abalone, columns=['Type'])
print(abalone.head())

print("This is for 1-hot vector for abalone")
num_runs = 5

# Calculating the percentage of instances in each output class (sex)
class_counts = abalone[['Type_F', 'Type_I', 'Type_M']].sum()
class_percentages = (class_counts / len(abalone)) * 100

# Plotting the percentages
plt.figure(figsize=(8, 6))
class_percentages.plot(kind='bar', color='skyblue')
plt.xlabel('Sex')
plt.ylabel('Percentage')
plt.title('Percentage of Instances in Each Sex')
plt.xticks(rotation=0)  # Assuming you don't need rotation for just three categories

# Saving the plot to a file 
plt.savefig("abalone-classes.png")

# Displaying the plot 
plt.show()

# Splitting the dataset into train and test sets (default parameter values)
X = abalone.drop(['Type_F', 'Type_I', 'Type_M'], axis=1)  # Features
y = abalone[['Type_F', 'Type_I', 'Type_M']]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

list_accuracy= []
list_f1macro=[]
list_f1weighted=[]
for _ in range(num_runs):
# Creating a Decision Tree model with default parameters
    clf = DecisionTreeClassifier(max_depth=3)

    # Fitting the model to the training data
    clf.fit(X_train, y_train)

    # Evaluating the best model on the test data
    y_pred_c = clf.predict(X_test)

    #Converting multi-label to binary label because confusion matrix not designed for multi-label 
    y_test_binary = y_test.idxmax(axis=1)
    y_pred_binary = pd.DataFrame(y_pred_c, columns=['Type_F', 'Type_I', 'Type_M']).idxmax(axis=1)
    ######
    print("------------------------------------------------------")
    # Computing the confusion matrix
    confusion_matrix_c = confusion_matrix(y_test_binary, y_pred_binary)
    print("Confusion Matrix (Decision Tree):")
    print(confusion_matrix_c)

    # Computing precision, recall, and F1-measure for each class
    report_c = classification_report(y_test_binary, y_pred_binary, target_names=['Type_F', 'Type_I', 'Type_M'])
    print("Classification Report (Decision Tree):")
    print(report_c)


    # Calculating accuracy
    accuracy_c = accuracy_score(y_test_binary, y_pred_binary)

    # Calculating macro-average F1
    f1_macro_c = f1_score(y_test_binary, y_pred_binary, average='macro')

    # Calculating weighted-average F1
    f1_weighted_c = f1_score(y_test_binary, y_pred_binary, average='weighted')

    precision_c= precision_score(y_test, y_pred_c, average=None)
    recall_c =recall_score(y_test, y_pred_c, average=None)

    print("Accuracy (Decision Tree):", accuracy_c)
    print("Macro-Average F1 (Decision Tree):", f1_macro_c)
    print("Weighted-Average F1 (Decision Tree):", f1_weighted_c)
    print("Precision (Decision Tree):")
    for i, precision in enumerate(precision_c):
        print(f"Class {i}: {precision:.4f}")
    print("Recall (Decision Tree):")
    for i, recall in enumerate(recall_c):
        print(f"Class {i}: {recall:.4f}")

    list_accuracy.append(accuracy_c)
    list_f1macro.append(f1_macro_c)
    list_f1weighted.append(f1_weighted_c)

    with open("penguin-performance.txt", "a") as file:
        file.write("This is for 1-hot vector for penguins\n")
        file.write("------------------------------------------------------\n")
        file.write("(A)")
        file.write("Confusion Matrix (Decision Tree):\n")
        file.write(str(confusion_matrix_c) + "\n")
        file.write("Classification Report (Decision Tree):\n")
        file.write(report_c + "\n")

        file.write(f"Accuracy (Decision Tree): {accuracy_c:.4f}\n")
        file.write(f"Macro-Average F1 (Decision Tree): {f1_macro_c:.4f}\n")
        file.write(f"Weighted-Average F1 (Decision Tree): {f1_weighted_c:.4f}\n")
        file.write(f"Precision (Decision Tree): \n")
        for i, precision in enumerate(precision_c):
            file.write(f"Class {i}: {precision:.4f}\n")
        file.write("Recall (Decision Tree):\n")
        for i, recall in enumerate(recall_c):
            file.write(f"Class {i}: {recall:.4f}\n")
        file.write("------------------------------------------------------\n")

####

# Plotting the Decision Tree graphically
plt.figure(figsize=(12, 8))
class_names = ["Type_F", "Type_I", "Type_M"]
plot_tree(clf, filled=True, feature_names=X.columns, class_names=class_names, rounded=True, fontsize=10)
plt.title('Decision Tree 1-hot vector for abalone')
plt.show()

for _ in range(num_runs):
    # Defining the parameter grid for GridSearchCV
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None,5, 10],  
        'min_samples_split': [2, 5, 10]  
    }

    # Creating a Decision Tree classifier
    dt_classifier = DecisionTreeClassifier(max_depth=3,random_state=None)

    # Creating a GridSearchCV object with cross-validation
    grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid)

    # Fitting the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Getting the best Decision Tree model
    best_dt = grid_search.best_estimator_
    print("------------------------------------------------------")
    # Printing the best hyperparameters
    print("Best Hyperparameters for Top decision tree:", grid_search.best_params_)



    # Evaluating the best model on the test data
    y_pred = best_dt.predict(X_test)


    #Converting multi-label to binary label because confusion matrix not designed for multilabel
    y_test_binary2 = y_test.idxmax(axis=1)
    y_pred_binary2 = pd.DataFrame(y_pred, columns=['Type_F', 'Type_I', 'Type_M']).idxmax(axis=1)

    ######

    # Computing the confusion matrix
    confusion_matrix_dt = confusion_matrix(y_test_binary2, y_pred_binary2)
    print("Confusion Matrix (Top Decision Tree):")
    print(confusion_matrix_dt)

    # Computing precision, recall, and F1-measure for each class
    report_dt = classification_report(y_test_binary2, y_pred_binary2, target_names=['Type_F', 'Type_I', 'Type_M'])
    print("Classification Report (Top Decision Tree):")
    print(report_dt)


    # Calculating accuracy
    accuracy_dt = accuracy_score(y_test_binary2, y_pred_binary2)

    # Calculating macro-average F1
    f1_macro_dt = f1_score(y_test_binary2, y_pred_binary2, average='macro')

    # Calculating weighted-average F1
    f1_weighted_dt = f1_score(y_test_binary2, y_pred_binary2, average='weighted')

    precision_dt= precision_score(y_test, y_pred,average=None)
    recall_dt =recall_score(y_test, y_pred, average=None)
    print("Accuracy (Top Decision Tree):", accuracy_dt)
    print("Macro-Average F1 (Top Decision Tree):", f1_macro_dt)
    print("Weighted-Average F1 (Top Decision Tree):", f1_weighted_dt)
    print("Precision (Top Decision Tree):")
    for i, precision in enumerate(precision_dt):
        print(f"Class {i}: {precision:.4f}")
    print("Recall(Top Decision Tree):")
    for i, recall in enumerate(recall_dt):
        print(f"Class {i}: {recall:.4f}")
    
    list_accuracy.append(accuracy_dt)
    list_f1macro.append(f1_macro_dt)
    list_f1weighted.append(f1_weighted_dt)
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
        file.write(f"Precision (Top Decision Tree): \n")
        for i, precision in enumerate(precision_dt):
            file.write(f"Class {i}: {precision:.4f}\n")
        file.write(f"Recall (Top Decision Tree): \n")
        for i, recall in enumerate(recall_dt):
            file.write(f"Class {i}: {recall:.4f}\n")
        file.write("------------------------------------------------------\n")

####

# Plotting the best Decision Tree graphically
plt.figure(figsize=(12, 8))
class_names = ["Type_F", "Type_I", "Type_M"]
plot_tree(best_dt, filled=True, feature_names=X.columns, class_names=class_names, rounded=True, fontsize=10)
plt.title('Best Decision Tree (GridSearchCV) for Abalone')
plt.show()

for _ in range(num_runs):
    # Creating a Multi-Layered Perceptron (MLP) with the specified parameters
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', random_state=None)

    # Fitting the MLP to the training data
    mlp.fit(X_train, y_train)

    y_pred_m = mlp.predict(X_test)

    print("------------------------------------------------------")

    #Converting multi-label to binary label because confusion matrix not designed for the prior
    y_test_binary3 = y_test.idxmax(axis=1)
    y_pred_binary3 = pd.DataFrame(y_pred_m, columns=['Type_F', 'Type_I', 'Type_M']).idxmax(axis=1)

    ##########
    # Computing the confusion matrix
    confusion_matrix_m = confusion_matrix(y_test_binary3, y_pred_binary3)
    print("Confusion Matrix (Multi-Layer Perceptron):")
    print(confusion_matrix_m)

    # Computing precision, recall, and F1-measure for each class
    report_m = classification_report(y_test_binary3, y_pred_binary3, target_names=['Type_F', 'Type_I', 'Type_M'])
    print("Classification Report (Multi-Layer Perceptron):")
    print(report_m)


    # Calculating accuracy
    accuracy_m = accuracy_score(y_test_binary3, y_pred_binary3)

    # Calculating macro-average F1
    f1_macro_m = f1_score(y_test_binary3, y_pred_binary3, average='macro')

    # Calculating weighted-average F1
    f1_weighted_m = f1_score(y_test_binary3, y_pred_binary3, average='weighted')
    precision_m= precision_score(y_test, y_pred,average=None)
    recall_m =recall_score(y_test, y_pred, average=None)
    print("Accuracy (Multi-Layer Perceptron):", accuracy_m)
    print("Macro-Average F1 (Multi-Layer Perceptron):", f1_macro_m)
    print("Weighted-Average F1 (Multi-Layer Perceptron):", f1_weighted_m)
    print("Precision (Multi-Layer Perceptron):")
    for i, precision in enumerate(precision_m):
        print(f"Class {i}: {precision:.4f}")

    list_accuracy.append(accuracy_m)
    list_f1macro.append(f1_macro_m)
    list_f1weighted.append(f1_weighted_m)
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
        file.write(f"Precision (Multi-Layer Perceptron): \n")
        for i, precision in enumerate(precision_m):
            file.write(f"Class {i}: {precision:.4f}\n")
        file.write(f"Recall (Multi-Layer Perceptron): \n")
        for i,recall in enumerate(recall_m):
            print(f"Class {i}: {recall:.4f}")
        file.write("------------------------------------------------------\n")


    # Evaluating the MLP on the test data
    accuracy = mlp.score(X_test, y_test)
    print("Test Accuracy:", accuracy)

print("------------------------------------------------------")

for _ in range(num_runs):
    # Defining the parameter grid for GridSearchCV
    param_grid = {
        'activation': ['sigmoid', 'tanh', 'relu'],
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],  
        'solver': ['adam', 'sgd']
    }

    # Creating an MLP classifier
    mlp_classifier = MLPClassifier(random_state=None)

    # Creating a GridSearchCV object with cross-validation
    grid_search = GridSearchCV(estimator=mlp_classifier, param_grid=param_grid, scoring='accuracy', cv=5)

    # Fitting the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Getting the best MLP model
    best_mlp = grid_search.best_estimator_


    # Using the trained model to make predictions on the test data
    y_pred_mlp = best_mlp.predict(X_test)

    #converting multi-label to binary label because confusion matrix not designed for the prior
    y_test_binary4 = y_test.idxmax(axis=1)
    y_pred_binary4 = pd.DataFrame(y_pred_mlp, columns=['Type_F', 'Type_I', 'Type_M']).idxmax(axis=1)

    #####

    # Computing the confusion matrix
    confusion_matrix_mlp = confusion_matrix(y_test_binary4, y_pred_binary4)
    print("Confusion Matrix (Top Multi-Layer Perceptron):")
    print(confusion_matrix_mlp)

    # Computing precision, recall, and F1-measure for each class
    report_mlp = classification_report(y_test_binary4, y_pred_binary4, target_names=['Type_F', 'Type_I', 'Type_M'])
    print("Classification Report (Top Multi-Layer Perceptron):")
    print(report_mlp)

    # Calculating accuracy
    accuracy_mlp = accuracy_score(y_test_binary4, y_pred_binary4)

    # Calculating macro-average F1
    f1_macro_mlp = f1_score(y_test_binary4, y_pred_binary4, average='macro')

    # Calculating weighted-average F1
    f1_weighted_mlp = f1_score(y_test_binary4, y_pred_binary4, average='weighted')
    precision_mlp= precision_score(y_test, y_pred,average=None)
    recall_mlp =recall_score(y_test, y_pred, average=None)

    print("Accuracy (Top Multi-Layer Perceptron):", accuracy_mlp)
    print("Macro-Average F1 (Top Multi-Layer Perceptron):", f1_macro_mlp)
    print("Weighted-Average F1 (Top Multi-Layer Perceptron):", f1_weighted_mlp)
    print("Precision (Top Multi-Layer Perceptron):")
    for i, precision in enumerate(precision_mlp):
        print(f"Class {i}: {precision:.4f}")
    print("Recall(Top Multi-Layer Perceptron):")
    for i, recall in enumerate(recall_mlp):
        print(f"Class {i}: {recall:.4f}")
    print("Best Hyperparameters (Top Multi-Layer Perceptron) :", grid_search.best_params_)
    
    list_accuracy.append(accuracy_mlp)
    list_f1macro.append(f1_macro_mlp)
    list_f1weighted.append(f1_weighted_mlp)
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
        file.write(f"Precision (Top Multi-Layer Perceptron): \n")
        for i, precision in enumerate(precision_mlp):
            file.write(f"Class {i}: {precision:.4f}\n")
        file.write(f"Recall (Multi-Layer Perceptron): \n")
        for i, recall in enumerate(recall_mlp):
            print(f"Class {i}: {recall:.4f}")
        file.write(str(grid_search.best_params_) + "\n")
        file.write("------------------------------------------------------\n")


##############



    # Evaluating the best model on the test data
    accuracy = best_mlp.score(X_test, y_test)
    print("Test Accuracy:", accuracy)


average_accuracy = np.mean(list_accuracy)
variance_accuracy = np.var(list_accuracy)

average_f1_macro = np.mean(list_f1macro)
variance_f1_macro = np.var(list_f1macro)

average_f1_weighted = np.mean(list_f1weighted)
variance_f1_weighted = np.var(list_f1weighted)

with open("abalone-performance.txt", "a") as file:
    file.write("Summary: \n")
    file.write("Average Accuracy: {:.4f}, Variance: {:.4f}\n".format(average_accuracy, variance_accuracy))
    file.write(" Average Macro-Average F1: {:.4f}, Variance: {:.4f}\n".format(average_f1_macro, variance_f1_macro))
    file.write(" Average Weighted-Average F1: {:.4f}, Variance: {:.4f}\n".format(average_f1_weighted, variance_f1_weighted))
    file.write("------------------------------------------------------\n")


print("Average Accuracy: {:.4f}, Variance: {:.4f}".format(average_accuracy, variance_accuracy))
print("Average Macro-Average F1: {:.4f}, Variance: {:.4f}".format(average_f1_macro, variance_f1_macro))
print("Average Weighted-Average F1: {:.4f}, Variance: {:.4f}".format(average_f1_weighted, variance_f1_weighted))



print("------------------------------------------------------")