# Data Processing
import pandas as pd
import numpy as np

# Modelling
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Visualisation
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from IPython.display import Image
import graphviz

# From data_cleaning.py
from Data_cleaning import clean_raw, clean_data_zscore, split_data, preprocessor

target = "body_style"

# Get data
df = pd.read_csv("data.csv")
pre = preprocessor(target)

X_train, X_test, y_train, y_test = split_data(df, target)

X_train_processed = pre.fit_transform(X_train, y_train)
X_test_processed = pre.transform(X_test)

""" print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape) """

tree_clf = DecisionTreeClassifier(random_state=42) # initialise classifier
tree_clf.fit(X_train_processed, y_train) # train the classifier

y_pred = tree_clf.predict(X_test_processed) # predict on the previously unseen test set

print("Train accuracy:", accuracy_score(y_train, tree_clf.predict(X_train_processed)))
print("Test accuracy: ", accuracy_score(y_test, y_pred))

plot_cm = False
if(plot_cm):
    cm = confusion_matrix(y_test, y_pred, labels=tree_clf.classes_) # compute a confusion matrix between the predicted labels and true labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree_clf.classes_) # create a figure for the confusion matrix
    disp.plot(cmap='Blues') # display the confusion matrix
    plt.tight_layout()
    plt.show()

show_tree = True
if show_tree:
    plt.figure(figsize=(24, 14))
    plot_tree(tree_clf, feature_names=pre.get_feature_names_out(), class_names=tree_clf.classes_, max_depth=4, filled=True, rounded=True, impurity=False, fontsize=9)
    plt.savefig("temp_figures/body_style_tree_depth4.png", dpi=300, bbox_inches="tight")
    plt.show()

print("Maximum depth:", tree_clf.get_depth())
print("Layers on deepest path:", tree_clf.get_depth() + 1)  # includes the root
print("Number of leaves:", tree_clf.get_n_leaves())
print("Total nodes:", tree_clf.tree_.node_count)
print("Classes:", tree_clf.classes_)
print(export_text(
    tree_clf,
    feature_names=list(pre.get_feature_names_out()),
    max_depth=3,  # limit output; remove this argument to print the full tree
))