import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.model_selection import cross_val_score
import random
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# import sklearn.model_selection as model_selection
from imblearn.over_sampling import RandomOverSampler

#应用主题

#应用标题
st.title('Machine Learning Application for Predicting Osteoporotic Fracture')



# conf
# st.sidebar.markdown('## Variables')
# #Age = st.sidebar.selectbox('Age',('<55','>=55'),index=0)
# Sex = st.sidebar.selectbox('Sex',('Female','Male'),index=0)
# T = st.sidebar.selectbox("T stage",('T1','T2','T3','T4'))
# N = st.sidebar.selectbox("N stage",('N0','N1','N2','N3'))
# #Race = st.sidebar.selectbox("Race",('American Indian/Alaska Native','Asian or Pacific Islander','Black','White'),index=3)
# Grade = st.sidebar.selectbox("Grade",('Ⅰ','Ⅱ','Ⅲ','Ⅳ'),index=0)
# Laterality =  st.sidebar.selectbox("Laterality",('Left','Right','Bilateral'))
# Histbehav =  st.sidebar.selectbox("Histbehav",('Adenocarcinoma','Squamous cell carcinoma'
#                                                ,'Adenosquamous carcinoma','Large cell carcinoma','other'))
# Chemotherapy = st.sidebar.selectbox("Chemotherapy",('No','Yes'))
#Marital_status = st.sidebar.selectbox("Marital status",('Married','Unmarried'))
col1, col2, col3 = st.columns(3)
age = col1.number_input('Age',step=1,value=50)
tuberculosis = col1.selectbox("tuberculosis",('No','Yes'))
History_of_fracture_surgery = col1.selectbox("fracture surgery history",('No','Yes'))
ALP = col2.number_input('ALP',value=60)
calcium = col2.number_input('calcium',value=2.20)
hemoglobin = col2.number_input('hemoglobin',value=100)
Mean_corpuscular_volume = col3.number_input('Mean corpuscular volume',value=90.00)
absolute_value_of_lymphocytes = col3.number_input('absolute value of lymphocytes',value=1.50)
Fibrinogen = col3.number_input('Fibrinogen',value=3.50)

# str_to_int

map = {'<55':1,'>=55':2,'Female':1,'Male':0,'American Indian/Alaska Native':1,'Asian or Pacific Islander':2,'Black':3,
       'White':4,'Ⅰ':0,'Ⅱ':1,'Ⅲ':2,'Ⅳ':3,'T1':0,'T2':1,'T3':2,'T4':3,
       'N0':0,'N1':1,'N2':2,'N3':3,
       'Adenocarcinoma':0,'Squamous cell carcinoma':1,
       'Adenosquamous carcinoma':2,'Large cell carcinoma':3,'other':4,
       'Married':1,'Unmarried':2,
       'Left':0,'Right':1,'Bilateral':2,
       'No':0,'Yes':1}

tuberculosis =map[tuberculosis]
History_of_fracture_surgery =map[History_of_fracture_surgery]
# T =map[T]
# N =map[N]
# Laterality =map[Laterality]
# Histbehav =map[Histbehav]
# Chemotherapy =map[Chemotherapy]

# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
thyroid_train['fracture'] = thyroid_train['fracture'].apply(lambda x : +1 if x==1 else 0)
#thyroid_test = pd.read_csv('test.csv', low_memory=False)
#thyroid_test['BM'] = thyroid_test['BM'].apply(lambda x : +1 if x==1 else 0)
features = [ 'age',
         'tuberculosis',
       'History_of_fracture_surgery',
       'ALP', 'calcium', 'hemoglobin', 'Mean_corpuscular_volume',
        'absolute_value_of_lymphocytes',
       'Fibrinogen']
target='fracture'

#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

#train and predict
#RF = sklearn.ensemble.RandomForestClassifier(n_estimators=7,criterion='entropy',max_features='log2',max_depth=5,random_state=12)
#RF.fit(thyroid_train[features],thyroid_train[target])
XGB = XGBClassifier(random_state=32,max_depth=3,n_estimators=10)
XGB.fit(X_ros, y_ros)
#读之前存储的模型

#with open('RF.pickle', 'rb') as f:
#    RF = pickle.load(f)


sp = 0.54
#figure
is_t = (XGB.predict_proba(np.array([[age,
                                    tuberculosis,
                                    History_of_fracture_surgery,
                                    ALP, calcium, hemoglobin, Mean_corpuscular_volume,
                                    absolute_value_of_lymphocytes,
                                    Fibrinogen]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[age,
                                    tuberculosis,
                                    History_of_fracture_surgery,
                                    ALP, calcium, hemoglobin, Mean_corpuscular_volume,
                                    absolute_value_of_lymphocytes,
                                    Fibrinogen]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for fracture:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of fracture:  '+str(prob)+'%')
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行



# st.title("")
# st.title("")
# st.title("")
# st.title("")
#st.warning('This is a warning')
#st.error('This is an error')

#st.info('Information of the model: Auc: 0. ;Accuracy: 0. ;Sensitivity(recall): 0. ;Specificity :0. ')
#st.success('Affiliation: The First Affiliated Hospital of Nanchang University, Nanchnag university. ')





