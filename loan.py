import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Case Study\\Case Study 3\\Dataset")

FullRaw = pd.read_csv("BankCreditCard.csv")

FullRaw.isnull().sum()

FullRaw.drop(['Customer_ID'], axis =1, inplace =True)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw, test_size = 0.3, random_state =123)

Train['Source'] = 'Train'
Test['Source'] = 'Test'

FullRaw = pd.concat([Train,Test], axis =0)

FullRaw.loc[FullRaw['Source'] == 'Train', 'Default_Payment'].value_counts()/FullRaw.loc[FullRaw['Source'] == 'Train'].shape[0]*100

Variable_to_update = 'Gender'
FullRaw[Variable_to_update].unique()
Condition = [FullRaw[Variable_to_update] == 1,FullRaw[Variable_to_update] == 2]
Choice = ['Male','Female']
FullRaw[Variable_to_update] = np.select(Condition,Choice)
FullRaw[Variable_to_update].unique()

Variable_to_update = 'Academic_Qualification'
FullRaw[Variable_to_update].unique()
Condition = [FullRaw[Variable_to_update] ==1,FullRaw[Variable_to_update] ==2,
             FullRaw[Variable_to_update] ==3,FullRaw[Variable_to_update] ==4,
             FullRaw[Variable_to_update] ==5,FullRaw[Variable_to_update] ==6]
Choice = ['Undergraduate', 'Graduate', 'Postgraduate', 'Professional', 'Others', 'Unknown']
FullRaw[Variable_to_update] = np.select(Condition,Choice)
FullRaw[Variable_to_update].unique()

Variable_to_update = 'Marital'
FullRaw[Variable_to_update].unique()
Condition = [FullRaw[Variable_to_update] ==0,FullRaw[Variable_to_update] ==1,
             FullRaw[Variable_to_update] ==2,FullRaw[Variable_to_update] ==3]
Choice = ['Unknown','Married', 'Single', 'Unknown']
FullRaw[Variable_to_update] = np.select(Condition,Choice)
FullRaw[Variable_to_update].unique()

import seaborn as sns

Corrdf = FullRaw.corr()
sns.heatmap(Corrdf, xticklabels= Corrdf.columns,
            yticklabels= Corrdf.columns,cmap = 'coolwarm_r')


Categorical_vars = (FullRaw.dtypes == 'object') & (FullRaw.columns != 'Source')
dummyDf = pd.get_dummies(FullRaw.loc[:,Categorical_vars],drop_first = True)

FullRaw2 = pd.concat([FullRaw.loc[:,~Categorical_vars],dummyDf], axis =1)
FullRaw2['My_Intercept'] = 1

Train2 = FullRaw2[FullRaw2['Source'] == 'Train'].drop(['Source'], axis =1)
Test2 = FullRaw2[FullRaw2['Source'] == 'Test'].drop(['Source'], axis =1)

Train_X = Train2.drop(['Default_Payment'], axis =1)
Train_Y = Train2['Default_Payment'].copy()
Test_X = Test2.drop(['Default_Payment'], axis =1)
Test_Y = Test2['Default_Payment'].copy()

from statsmodels.stats.outliers_influence import variance_inflation_factor

Max_VIF = 10
counter = 1
Train_X_Copy = Train_X.copy()
High_VIF_Col_Names = []

while Max_VIF >= 10:
    print(counter)
    
    VIF_Df = pd.DataFrame()
    VIF_Df['VIF'] = [variance_inflation_factor(Train_X_Copy.values,i) for i in range(Train_X_Copy.shape[1])]
    VIF_Df['Column_Name'] = Train_X_Copy.columns
    
    Max_VIF = max(VIF_Df['VIF'])
    Temp_Column_Name = VIF_Df.loc[VIF_Df['VIF'] == Max_VIF,'Column_Name']
    print(Temp_Column_Name,":", Max_VIF)
    
    Train_X_Copy = Train_X_Copy.drop(Temp_Column_Name, axis =1)
    High_VIF_Col_Names.extend(Temp_Column_Name)
    
    counter = counter + 1

High_VIF_Col_Names

Train_X = Train_X.drop(High_VIF_Col_Names, axis =1)
Test_X = Test_X.drop(High_VIF_Col_Names, axis =1)

from statsmodels.api import Logit

M1 = Logit(Train_Y,Train_X).fit()
M1.summary()

Col_To_Drop = ['Academic_Qualification_Postgraduate']
M2 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M2.summary()

Col_To_Drop.append('Marital_Unknown')
M3 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M3.summary()

Col_To_Drop.append('June_Bill_Amount')
M4 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M4.summary()

Col_To_Drop.append('Age_Years')
M5 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M5.summary()
    
Col_To_Drop.append('Repayment_Status_Feb')
M6 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M6.summary()

Col_To_Drop.append('Academic_Qualification_Unknown')
M7 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M7.summary()

Col_To_Drop.append('Academic_Qualification_Undergraduate')
M8 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M8.summary()

Col_To_Drop.append('Previous_Payment_May')
M9 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M9.summary()

Col_To_Drop.append('Repayment_Status_April')
M10 = Logit(Train_Y,Train_X.drop(Col_To_Drop, axis =1)).fit()
M10.summary()

Train_X = Train_X.drop(Col_To_Drop, axis =1)
Test_X = Test_X.drop(Col_To_Drop, axis =1)

Test_pred = M10.predict(Test_X)
Test['Test_prob'] = Test_pred
Test['Test_Class'] = np.where(Test['Test_prob'] >= 0.5,1,0)
Test['Test_Class'].value_counts()
 
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score

Con_Mat = confusion_matrix(Test['Test_Class'],Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100 # 81.90%

recall_score(Test['Test_Class'],Test_Y)*100
precision_score(Test['Test_Class'],Test_Y)*100
f1_score(Test['Test_Class'],Test_Y)*100

from sklearn.ensemble import RandomForestClassifier

RF_Model = RandomForestClassifier(random_state =123).fit(Train_X,Train_Y)

RF_pred = RF_Model.predict(Test_X)
RF_Mat = confusion_matrix(RF_pred,Test_Y)
sum(np.diag(RF_Mat))/Test_Y.shape[0]*100 # 80.28%

from sklearn.model_selection import GridSearchCV

n_tree = [100,200]
n_split = [50,75,100]
my_param_grid = {'n_estimators': n_tree, 'min_samples_split': n_split}

Grid = GridSearchCV(RandomForestClassifier(random_state =123),param_grid = my_param_grid,
                    scoring ='accuracy', cv =5).fit(Train_X,Train_Y)
Grid.best_score_

Grid_Df = pd.DataFrame.from_dict(Grid.cv_results_)

RF_Model2 = RandomForestClassifier(n_estimators =200,min_samples_split =75,
                                   random_state=123).fit(Train_X,Train_Y)

RF_pred2 = RF_Model2.predict(Test_X)
RF_Mat2 = confusion_matrix(RF_pred2,Test_Y)
sum(np.diag(RF_Mat2))/Test_Y.shape[0]*100

import pickle

pickle.dump(RF_Model2,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
