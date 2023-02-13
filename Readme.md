# Supervised machine learning for blight ticket compliance
This is a Python code for a supervised machine learning model for predicting compliance with blight ticket compliance. The code imports required libraries, reads the train and test datasets and performs data preprocessing. Then, it uses Gradient Boosting Classifier to fit the model on the training data and predict the probability of compliance for the test data. The code also visualizes the performance of the model using ROC curve and AUC. The final output is the prediction of compliance for the test data with a 'ticket_id' index and a 'probability' column.

The code is a Python script that implements a supervised machine learning algorithm for blight ticket compliance prediction. The script does the following:

* Imports necessary libraries: pandas, numpy, scikit-learn, and matplotlib.

* Defines the blight_model() function to perform the machine learning task.

* Loads the training data into a pandas data frame using the pd.read_csv() function and saves it in a variable df_train.

* Loads the test data into a pandas data frame using the pd.read_csv() function and saves it in a variable df_test.

* Selects relevant columns from the training data frame and removes any missing values using the df.dropna() function. The selected columns are ticket_id, compliance, judgment_amount, payment_amount, and balance_due.

* Splits the data into training and test sets using the train_test_split() function from the scikit-learn library. The training set is saved in variables X_train and y_train, and the test set is saved in variables X_test and y_test.

* Instantiates a gradient boosting classifier with learning rate 0.09 and maximum depth 3 using the GradientBoostingClassifier class from scikit-learn.

* Fits the gradient boosting classifier to the training data using the fit() method.

* Predicts the probabilities of the test set and the final test set using the predict_proba() method. The final test set probabilities are saved in a data frame probability_final.

* Calculates the ROC curve and AUC score using the roc_curve() and auc() functions from scikit-learn. The false positive rate and true positive rate are saved in variables fpr_lr and tpr_lr, respectively, and the AUC score is saved in a variable roc_auc_lr.

* Plots the ROC curve using the plot() function from matplotlib.

* Returns the final test set probabilities data frame.

* The code also includes a plot of the ROC curve, but it will only be shown if the function is run in an interactive environment (e.g., Jupyter Notebook).
