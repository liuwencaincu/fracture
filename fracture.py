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
st.set_page_config(
    page_title="ML Medicine",
    page_icon="🐇",
)
#应用标题
st.title('Machine Learning Application for Predicting Osteoporotic Fracture')



# conf
# st.sidebar.markdown('## Variables')
# #Age = st.sidebar.selectbox('Age',('<55','>=55'),index=0)
# Sex = st.sidebar.selectbox('Sex',('Female','Male'),index=0)
# T = st.sidebar.selectbox("T stage",('T1','T2','T3','T4'))
# HGB = st.sidebar.slider("HGB", 0, 200, value=100, step=1)
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
#tuberculosis = col1.selectbox("tuberculosis",('No','Yes'))
History_of_fracture_surgery = col1.selectbox("fracture surgery history",('No','Yes'))
ALP = col2.number_input('ALP',value=60)
calcium = col3.number_input('calcium',step=0.01,format="%.2f",value=2.20)
albumin = col2.selectbox("albumin",('>35','30-35','25-30','<25'))
#hemoglobin = col2.number_input('hemoglobin',value=100)
#Mean_corpuscular_volume = col3.number_input('Mean corpuscular volume',value=90.00)
#absolute_value_of_lymphocytes = col3.number_input('absolute value of lymphocytes',value=1.50)
Fibrinogen = col3.number_input('Fibrinogen',value=3.50)

# str_to_int

map = {'Female':0,'Male':1,
       'Married':1,'Unmarried':2,
       'Left':0,'Right':1,'Bilateral':2,
        '>35':0,'30-35':1,'25-30':2,'<25':3,
       'No':0,'Yes':1}
albumin = map[albumin]
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
       'History_of_fracture_surgery',
       'ALP', 'albumin','calcium',
       'Fibrinogen']
target='fracture'

#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

#train and predict
#RF = sklearn.ensemble.RandomForestClassifier(n_estimators=7,criterion='entropy',max_features='log2',max_depth=5,random_state=12)
#RF.fit(thyroid_train[features],thyroid_train[target])
XGB = XGBClassifier(random_state=32,max_depth=3,n_estimators=30)
XGB.fit(X_ros, y_ros)
#读之前存储的模型

#with open('RF.pickle', 'rb') as f:
#    RF = pickle.load(f)


sp = 0.5
#figure
is_t = (XGB.predict_proba(np.array([[age,
       History_of_fracture_surgery,
       ALP, albumin,calcium,
       Fibrinogen]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[age,
       History_of_fracture_surgery,
       ALP, albumin,calcium,
       Fibrinogen]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    #st.markdown('###### threshold = 0.5')
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





