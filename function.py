import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# this function return the list of numerical and categorical feature name
def cat_features(df,feature_category):
  cat_df = df.select_dtypes(include = ['object', 'bool']).copy()
  num_df = df.select_dtypes(exclude = ['object', 'bool']).copy()
  if feature_category=='numerical':
    return num_df.columns
  else:
    return cat_df.columns

# plot for categorical feature
def categorical_plot(df,feature_name,target):
  if df[target].dtype == "bool":
    df = df.astype({target: 'object'})
  # print("cat plot", df[feature_name].dtype, pd.api.types.is_numeric_dtype(df[feature_name]))
  # print("cat plot", df[target].dtype, pd.api.types.is_numeric_dtype(df[target]))
  if pd.api.types.is_numeric_dtype(df[feature_name]) or pd.api.types.is_numeric_dtype(df[target]):
    f,axes = plt.subplots(3,1,figsize = (15,25))
    sns.countplot(data = df,x= feature_name,ax = axes[0])
    sns.boxplot(data = df,x=feature_name, y=target,ax = axes[1],palette='rainbow')
    sns.violinplot(data = df,x=feature_name, y=target,ax = axes[2],palette='rainbow')
  else:
    f,axes = plt.subplots(1,1,figsize = (15,10))
    sns.countplot(data = df,x= feature_name)
  return plt

from seaborn.relational import scatterplot
def numerical_plot(df,feature_name,target):
  if pd.api.types.is_numeric_dtype(df[target]):
    f,axes = plt.subplots(3,1,figsize=(14,14))
    sns.set(palette="rainbow", font_scale = 1.1)
    sns.distplot(df[target],norm_hist=False, bins=20, hist_kws={'alpha':1},ax = axes[0]).set(xlabel=target,ylabel = 'count')
    sns.histplot(data = df,x = feature_name,ax = axes[1],bins = 15)
    sns.scatterplot(data = df,x = feature_name,y = target,ax = axes[2])
  else:
    f,axes = plt.subplots(1,1,figsize=(14,5))
    sns.set(palette="rainbow", font_scale = 1.1)
    sns.histplot(data = df,x = feature_name,bins = 15)
  return plt

# label encoding step here function made by abhishek 

# feature scaling
from sklearn.preprocessing import MinMaxScaler
def feature_scaling(df,target):
  x = df.drop(df[target],axis = 1)
  col_name = x.columns
  scaler = MinMaxScaler()
  X = scaler.fit_transform(x)
  X = pd.DataFrame(X,columns = [col_name])
  return X

# dataset split
from sklearn.model_selection import train_test_split
def split_dataset(df,target):
    x = df.drop(target, axis = 1)
    y = df[target]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 11)
    return x_train,x_test,y_train,y_test

# feature selection 
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel 
def feature_selection(x,y):
  feature_sel_model = SelectFromModel(Lasso(alpha=0.005,random_state = 0))
  feature_sel_model.fit(x,y)
  selected_feature = x.columns[(feature_sel_model.get_support())]
  x = x[selected_feature].reset_index(drop=True)
  return x

# x_train = feature_selection(x_train,y_train)
# x_test = feature_selection(x_test,y_test)

# model training
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def linear_regression(x_train,y_train,x_test,y_test):
  try:
    lm = LinearRegression()
    lm.fit(x_train,y_train)
    lm_predict = lm.predict(x_test)

    # plt.scatter(y_test,lm_predict)
    # plt.show()

    ret = []
    ret.append("### Linear Regression Results")
    ret.append("MAE : " + str(metrics.mean_absolute_error(y_test,lm_predict)))
    ret.append("MSE : " + str(metrics.mean_squared_error(y_test,lm_predict)))
    ret.append('RMSE : ' + str(np.sqrt(metrics.mean_squared_error(y_test,lm_predict))))

    ret.append('\nAccuracy : ' + str(format(lm.score(x_test,y_test)*100)))
    return ret
  except:
    return ["Can't apply the selected model on data"]

from sklearn.svm import LinearSVR
def linear_SVR(x_train,y_train,x_test,y_test):
  try:
    svr = LinearSVR()
    svr.fit(x_train,y_train)
    svr_predict = svr.predict(x_test)

    # plt.scatter(y_test,svr_predict)
    # plt.show()

    ret = []
    ret.append("### Linear SVR Results")
    ret.append("MAE : " + str(metrics.mean_absolute_error(y_test,svr_predict)))
    ret.append("MSE : " + str(metrics.mean_squared_error(y_test,svr_predict)))
    ret.append('RMSE : ' + str(np.sqrt(metrics.mean_squared_error(y_test,svr_predict))))

    ret.append('\nAccuracy : ' + str(format(svr.score(x_test,y_test)*100)))
    return ret
  except:
    return ["Can't apply the selected model on data"]

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
def decisiontreeregressor(x_train,y_train,x_test,y_test):
  try:
    dtr = DecisionTreeRegressor()
    dtr.fit(x_train,y_train)
    dtr_predict = dtr.predict(x_test)

    # plt.scatter(y_test,dtr_predict)
    # plt.show()

    ret = []
    ret.append("### Decision Tree Regressor Results")
    ret.append("MAE : " + str(metrics.mean_absolute_error(y_test,dtr_predict)))
    ret.append("MSE : " + str(metrics.mean_squared_error(y_test,dtr_predict)))
    ret.append('RMSE : ' + str(np.sqrt(metrics.mean_squared_error(y_test,dtr_predict))))

    ret.append('\nAccuracy : ' + str(format(dtr.score(x_test,y_test)*100)))
    return ret
  except:
    return ["Can't apply the selected model on data"]

from sklearn.neighbors import KNeighborsRegressor
def kneighborsregressor(x_train,y_train,x_test,y_test):
  try:
    knr = KNeighborsRegressor()
    knr.fit(x_train,y_train)
    knr_predict = knr.predict(x_test)

    # plt.scatter(y_test,knr_predict)
    # plt.show()

    ret = []
    ret.append("### K Neighbors Regressor Results")
    ret.append("MAE : " + str(metrics.mean_absolute_error(y_test,knr_predict)))
    ret.append("MSE : " + str(metrics.mean_squared_error(y_test,knr_predict)))
    ret.append('RMSE : ' + str(np.sqrt(metrics.mean_squared_error(y_test,knr_predict))))

    ret.append('\nAccuracy : ' + str(format(knr.score(x_test,y_test)*100)))
    return ret
  except:
    return ["Can't apply the selected model on data"]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
def logisticregression(x_train,y_train,x_test,y_test):
  try:
    lr = LogisticRegression()
    lr.fit(x_train,y_train)

    lr_predict = lr.predict(x_test)

    # cm = confusion_matrix(y_test,lr_predict)
    # sns.heatmap(cm,cmap='Blues')
    # sns.show()
    # print()
    # print("classication report : ")
    # print()
    # print(classification_report(y_test,lr_predict))
    # print()
    # print('\nAccuracy : {}',fostr(rmat(accuracy_score(y_test,lr_predict)*100))
    report = classification_report(y_test,lr_predict, output_dict=True)
    return ["### Logistic Regressor Results", pd.DataFrame(report).transpose()]
  except:
    return ["Can't apply the selected model on data"]



from sklearn.naive_bayes import GaussianNB
def naivebayes(x_train,y_train,x_test,y_test):
  try:
    nb = GaussianNB()
    nb.fit(x_train,y_train)

    nb_predict = nb.predict(x_test)

    # cm = confusion_matrix(y_test,nb_predict)
    # sns.heatmap(cm,cmap='Blues')
    # sns.show()
    # print()
    # print("classication report : ")
    # print()
    # print(classification_report(y_test,nb_predict))
    # print()
    # print('\nAccuracy : {}',fostr(rmat(accuracy_score(y_test,nb_predict)*100))
    report = classification_report(y_test,nb_predict, output_dict=True)
    return ["### Naive Bayes Results", pd.DataFrame(report).transpose()]
  except:
    return ["Can't apply the selected model on data"]



from sklearn.neighbors import KNeighborsClassifier
def kneighborsclassifier(x_train,y_train,x_test,y_test):
  try:
    knc = KNeighborsClassifier()
    knc.fit(x_train,y_train)

    knc_predict = knc.predict(x_test)

    # cm = confusion_matrix(y_test,knc_predict)
    # sns.heatmap(cm,cmap='Blues')
    # sns.show()
    # print()
    # print("classication report : ")
    # print()
    # print(classification_report(y_test,knc_predict))
    # print()
    # print('\nAccuracy : {}',fostr(rmat(accuracy_score(y_test,knc_predict)*100))
    report = classification_report(y_test,knc_predict, output_dict=True)
    return ["### K Neighbors Classifier Results", pd.DataFrame(report).transpose()]
  except:
    return ["Can't apply the selected model on data"]



from sklearn.tree import DecisionTreeClassifier
def decisiontreeclassifier(x_train,y_train,x_test,y_test):
  try:
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train,y_train)

    dtc_predict = dtc.predict(x_test)

    # cm = confusion_matrix(y_test,dtc_predict)
    # sns.heatmap(cm,cmap='Blues')
    # sns.show()
    # print()
    # print("classication report : ")
    # print()
    # print(classification_report(y_test,dtc_predict))
    # print()
    # print('\nAccuracy : {}',fostr(rmat(accuracy_score(y_test,dtc_predict)*100))
    report = classification_report(y_test,dtc_predict, output_dict=True)
    return ["### Decision Tree Classifier Results", pd.DataFrame(report).transpose()]
  except:
    return ["Can't apply the selected model on data"]



from sklearn.svm import LinearSVC
def linear_SVC(x_train,y_train,x_test,y_test):
  svc = LinearSVC()
  try:
    svc.fit(x_train,y_train)

    lr_predict = svc.predict(x_test)

    # cm = confusion_matrix(y_test,lr_predict)
    # sns.heatmap(cm,cmap='Blues')
    # sns.show()
    
    # print()
    # print("classication report : ")
    # print()
    # return ('\nAccuracy : {}',fostr(rmat(accuracy_score(y_test,lr_predict)*100))
    report = classification_report(y_test,lr_predict, output_dict=True)
    return ["### Linear SVC Results", pd.DataFrame(report).transpose()]
  except:
    return ["Can't apply the selected model on data"]
  # print()
  # print('\nAccuracy : {}',fostr(rmat(accuracy_score(y_test,lr_predict)*100)))

def model_training(x_train,y_train,x_test,y_test,model_type):
  if model_type=='Linear Regression':
    return linear_regression(x_train,y_train,x_test,y_test)
  elif model_type=='Linear SVR':
    return linear_SVR(x_train,y_train,x_test,y_test)
  elif model_type=='Decision Tree Regressor':
    return decisiontreeregressor(x_train,y_train,x_test,y_test)
  elif model_type=='K Neighbour Regressor':
    return kneighborsregressor(x_train,y_train,x_test,y_test)
  elif model_type=='Logistic Regressor':
    return logisticregression(x_train,y_train,x_test,y_test)
  elif model_type=='Naive Bayes':
    return naivebayes(x_train,y_train,x_test,y_test)
  elif model_type=='K Neighbours Classifier':
    return kneighborsclassifier(x_train,y_train,x_test,y_test)
  elif model_type=='Decision Tree Classifier':
    return decisiontreeclassifier(x_train,y_train,x_test,y_test)
  elif model_type=='Linear SVC':
    return linear_SVC(x_train,y_train,x_test,y_test)