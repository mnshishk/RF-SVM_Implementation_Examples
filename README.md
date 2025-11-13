These are two files which implement the preprocessesing, classification, training, and testing of RF and SVM algorithms on CICIDS2017 and UNSW-NB15 datasets.
Libraries used are:
1. pandas
2. numpy
3. sklearn

*First two are standard libraries for python and sklearn includes RF and SVM among other tools and algorithms for training and testing. 

The datasets must be downloaded from their respective research pages:
1. CICIDS2017 - https://www.unb.ca/cic/datasets/ids-2017.html (Monday and Tuesday traffic files were used for the example outputs). 
2. UNSW-NB15 - https://research.unsw.edu.au/projects/unsw-nb15-dataset (the testing and training sets were used for example outputs).

After running the code the following outputs will be displayed:

CICIDS2017:

Dataset distribution: 
Label
0    98581
1     1385
Name: count, dtype: int64

--- Random Forest ---
Accuracy: 0.9997999399819946
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19717
           1       1.00      0.99      0.99       277

    accuracy                           1.00     19994
   macro avg       1.00      0.99      1.00     19994
weighted avg       1.00      1.00      1.00     19994

Confusion Matrix:
 [[19717     0]
 [    4   273]]

--- Support Vector Machine ---
Accuracy: 0.9954486345903771
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19717
           1       0.94      0.72      0.81       277

    accuracy                           1.00     19994
   macro avg       0.97      0.86      0.91     19994
weighted avg       1.00      1.00      1.00     19994

Confusion Matrix:
 [[19704    13]
 [   78   199]]

Summary:

Random Forest Accuracy: 99.98%
Detection Rate (Recall): 98.56%
False Positive Rate: 0.00%

SVM Accuracy: 99.54%
Detection Rate (Recall): 71.84%
False Positive Rate: 0.07%

UNSW-NB15
Dataset distribution: 
label
1    63917
0    36083
Name: count, dtype: int64


--- Random Forest ---
Accuracy: 0.9791
              precision    recall  f1-score   support

           0       0.97      0.97      0.97      7217
           1       0.98      0.98      0.98     12783

    accuracy                           0.98     20000
   macro avg       0.98      0.98      0.98     20000
weighted avg       0.98      0.98      0.98     20000

Confusion Matrix:
 [[ 6993   224]
 [  194 12589]]

--- Support Vector Machine ---
Accuracy: 0.92875
              precision    recall  f1-score   support

           0       0.94      0.86      0.90      7217
           1       0.92      0.97      0.95     12783

    accuracy                           0.93     20000
   macro avg       0.93      0.91      0.92     20000
weighted avg       0.93      0.93      0.93     20000

Confusion Matrix:
 [[ 6190  1027]
 [  398 12385]]

Summary:

Random Forest Accuracy: 97.91%
Detection Rate (Recall): 98.48%
False Positive Rate: 3.10%

SVM Accuracy: 92.88%
Detection Rate (Recall): 96.89%
False Positive Rate: 14.23%