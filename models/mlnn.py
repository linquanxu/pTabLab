# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from pandas import read_excel

def knn(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=6)  
    knn.fit(x_train, y_train) 
    # joblib.dump(knn,'a_knn.m')
    y_pred = knn.predict(x_test)
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))
    return y_pred

def LR(x_train, y_train, x_test, y_test):
    LR = LogisticRegression(C=0.01, solver='liblinear',verbose=2).fit(x_train,y_train)
    # joblib.dump(LR,'a_LR.m')
    y_pred = LR.predict(x_test)
    print(confusion_matrix(y_test, y_pred))  
    print (classification_report(y_test, y_pred))
    return y_pred

def DecisionTree(x_train, y_train, x_test, y_test):
    classifier = DecisionTreeClassifier()  
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
    return y_pred

def randfroest(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(n_estimators=100,verbose=2)
    model.fit(x_train, y_train)
    # joblib.dump(model,'a_rf.m')
    # model = joblib.load(pred_model)
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
    return y_pred
    
def svm(x_train, y_train, x_test, y_test):
    model = SVC(kernel = 'rbf', C = 1000,gamma=0.001, probability=True)
    model.fit(x_train, y_train)
    # joblib.dump(model,'a_svm.m')
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
    return y_pred

def bayes(x_train, y_train, x_test, y_test):
    GNB = GaussianNB().fit(x_train,y_train)
    # joblib.dump(GNB,'a_beyes.m')
    y_pred = GNB.predict(x_test)
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))
    return y_pred

def make_mlp_data(path):
    df = read_excel(path, header=None)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    X = X.astype('float32')
    y = y.astype('float32')
    return X, y

if __name__ == "__main__":
    train_path = 'train.xls'
    test_path = 'test.xls'
    x_train, y_train = make_mlp_data(train_path)
    x_test, y_test = make_mlp_data(test_path)
    results = svm(x_train, y_train, x_test, y_test )