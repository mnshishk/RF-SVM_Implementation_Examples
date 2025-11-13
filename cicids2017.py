import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepareData():
    # This can be adapted to your self-test runs by substituting the filenames/paths 
    # and add more datasets.
    monday = pd.read_csv('./MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv')
    tuesday = pd.read_csv('./MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv')

    data = pd.concat([monday, tuesday])
    data = data.dropna()
    data.columns = data.columns.str.strip()

    # Encode label: 0 = Benign, 1 = Attack
    data['Label'] = data['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

    # Limit samples where n is the sample size
    data = data.sample(n=100000, random_state=42)
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    print("\nDataset distribution: ")
    print(data['Label'].value_counts())
    print("\n")
    return data

def splitData(data):
    x = data.drop(columns=['Label'])
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)
    
    # normalization vector for svm
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

def trainRF(X_train, X_test, y_train, y_test):
    # n_estimators = 100, 200, 500, 1000 (>500 diminishing returns)
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    print("--- Random Forest ---")
    print("Accuracy:", accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

    return rf, rf_pred

def trainSVM(X_train_scaled, X_test_scaled, y_train, y_test):    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)

    svm_pred = svm.predict(X_test_scaled)

    print("\n--- Support Vector Machine ---")
    print("Accuracy:", accuracy_score(y_test, svm_pred))
    print(classification_report(y_test, svm_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
    
    return svm, svm_pred

def confusionCalculation(y_test, mod_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, mod_pred).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    return acc, tpr, fpr

def printOverallStats(y_test, rf_pred, svm_pred):
    rf_acc, rf_tpr, rf_fpr = confusionCalculation(y_test, rf_pred)

    print("\nSummary:\n")

    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(f"Detection Rate (Recall): {rf_tpr*100:.2f}%")
    print(f"False Positive Rate: {rf_fpr*100:.2f}%\n")
    
    svm_acc, svm_tpr, svm_fpr = confusionCalculation(y_test, svm_pred)

    print(f"SVM Accuracy: {svm_acc*100:.2f}%")
    print(f"Detection Rate (Recall): {svm_tpr*100:.2f}%")
    print(f"False Positive Rate: {svm_fpr*100:.2f}%")    

def main ():
    data = prepareData()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = splitData(data)

    rf_model, rf_pred = trainRF(X_train, X_test, y_train, y_test)
    svm_model, svm_pred = trainSVM(X_train_scaled, X_test_scaled, y_train, y_test)

    printOverallStats(y_test, rf_pred, svm_pred)

if __name__ == "__main__":
    main()
