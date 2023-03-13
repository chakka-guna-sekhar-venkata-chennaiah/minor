import pandas as pd
import streamlit as st
import sweetviz as sv
import os
from sklearn.metrics import roc_auc_score,roc_curve
import joblib
import hashlib
import re
from PIL import Image
from sklearn.metrics import roc_curve, auc
import plotly.figure_factory as ff
from sklearn import set_config
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from managed_db import *
import time
import pickle
import base64
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score,f1_score,mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import estimator_html_repr
from xgboost import XGBClassifier




def generated_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verified_hashes(password,hashed_text):
    if generated_hashes(password)==hashed_text:
        return hashed_text
    return False
def home():
    st.title("Loan Prediction App")
    st.write("""
    This web app predicts whether a loan will be approved or not based on various factors such as credit score, income, loan amount, etc.
    """)

    st.image("main.jpg", use_column_width=True)

    st.write("""
    To use this app, please fill out the following form with your loan information and click on the 'Predict' button. You will then see the prediction result (approved or not approved) and the probability of approval.
    """)

    st.write("""
    ### Loan Application Form
    """)

    # Add your loan application form components here

    st.write("""
    ---\n
    Made with ❤️ by Chakka Guna Sekhar Venkata Chennaiah.
    """)
    st.write("Instructions for Using Our App:")
    st.write("* Navigate to the Signup page and complete the process to become eligible for login.")
    st.write("* Once you have signed up, go to the Login page and enter your credentials. If your credentials are incorrect, you will not be able to access the app.")
    st.write("* After logging in, you will see four main pages: Dashboard, Login, Signup, and About. Click on the subpages in the Login tab in the following order: EDA, Model Building, Deploying models without parameters, Deploying models with parameters, Non-Parameter vs Parameter, Prediction, and Model Performance.")
    st.write("* It is crucial to check all checkboxes and buttons across submenus to ensure accurate results. If any of them are not activated, the results may not appear as expected. Please note that one checkbox result is interlinked to another checkbox result, so do not miss any checkbox while working with the application.")
    st.write("* If you have checked a checkbox on a page and visit that page again later, the checkbox will not be checked, but your previous checked results will be stored in the sessions separately. Do not worry about this, but make sure to check all checkboxes at least once.")
    st.write("* We have designed the app with a step-by-step approach, so please visit the subpages in the order listed above.")
    st.write("* To better understand how to use the app, please watch the video that we have provided. You can find the video at below")
    
    

    # Insert the video using the 'video' function and apply CSS to make it responsive
    video_file = open('video.mp4', 'rb').read()
    #video_bytes = video_file.read()
    st.video(video_file)

def login(sub_menu):
    username=st.sidebar.text_input('Username')
    password=st.sidebar.text_input('Password',type='password')
    if st.sidebar.checkbox('Login'):
        create_usertable()
        hashed_pwsd=generated_hashes(password)
        result=login_user(username,verified_hashes(password,hashed_pwsd))
        if result:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for i in range(100):
                progress_bar.progress((i + 1) / 100)
                status_text.text(f"Processing {i+1}%")
                time.sleep(0.01) # Add a delay to simulate processing time

            status_text.text("Processing completed!")
            st.success('Welcome {}'.format(username))
            
            
            subpage=st.sidebar.selectbox('Select the sub page',sub_menu)
            
            if subpage=='EDA':
                eda()
            elif subpage=='Model Building':
                model_building()
                
            elif subpage=='Deploying models without parameters':
                deploying_without_tuning()
            elif subpage=='Deploying models with parameters':
                deploying_with_tuning()
            elif subpage=='Non-Parameters vs Parameters':
                non_tuning_vs_tuning()
            elif subpage=='Prediction':
                prediction()
            elif subpage=='Model Performance':
                actpred()
        else:
            st.warning('Incorrect Username/Password')

    
def sign_up():
    new_username=st.text_input('User name')
    new_password=st.text_input('Password',type='password')
    confirm_password=st.text_input('Confirm Password',type='password')
    
    if new_password==confirm_password and new_password!='':
        st.success('Password Confirmed')
    else:
        st.warning('Passwords not the same' )
        
    if st.button('Submit'):
        create_usertable()
        hashed_new_password=generated_hashes(new_password)
        add_userdata(new_username,hashed_new_password)
        st.success('You are successfully created a new account')
        
        st.info('Login to get started')


def about_page():
    st.write("""
    ## About This App

    This app was created to showcase how to create an About page in Streamlit that contains developer information.

    ## Developer Information

    This app was developed by Chakka Guna Sekhar Venkata Chennaiah.
    If you have any questions or feedback, feel free to reach out to me at email [sekharchennaiah12345ctk@gmail.com].

    ## Technologies Used

    This app was built using the following technologies:
    * Streamlit - For creating the web app
    * Python - For writing the code

    ## Source Code

    The source code for this app is available on [GitHub](https://github.com/software-babooi/yourapp).

    ## Acknowledgements

    Special thanks to [Streamlit](https://streamlit.io) for providing an amazing platform for creating data apps.

    ## About the Dataset

    The dataset used in this app is LoanApprovalPrediction and can be found in my github repo.
    """)

   

    st.write("""
    #### Connect with us:
    [![Email](https://img.shields.io/badge/Email-sekharchennaiah12345ctk@gmail.com-blue)](mailto:sekharchennaiah12345ctk@gmail.com)
    [![Instagram](https://img.shields.io/badge/Instagram-vinayaka_03-orange)](https://www.instagram.com/vinayaka_03/)
    """)

    st.write("""
    #### Version:
    1.0.0
    """)
    


    
def eda():
    if 'eda' not in st.session_state:
        st.session_state.eda=0
    
    sub1=['Sweetviz']
    st.subheader("Perform Exploratory data Analysis with sweetviz Library")
    #data_file= st.file_uploader("Upload a csv file", type=["csv"])
    df=pd.read_csv('LoanApprovalPrediction.csv')
    status=st.selectbox('Which type of EDA is needed?',sub1)
                
    if status=='Sweetviz':
        if st.button('Analyze'):
            if df is not None:
                
                
                st.header('*User Input DataFrame*')
                st.write(df)
                st.write('---')
                st.subheader('*Exploratory Data Analysis Report Using Sweetviz*')
                report = sv.analyze(df)
                report_html=report.show_html()
                st.components.v1.html(report_html, height=700, scrolling=True)
                st.session_state.eda=1
                            
            else:
                st.warning('File not found')
    

def model_building():
    
    st.warning('''Please note that all checkboxes and buttons across sub-menus are interlinked. 
                    If any of them are not activated, the results may not appear as expected. We recommend ensuring that all checkboxes and buttons are properly activated before proceeding.''')
  
    if True:
        pass
        
        st.session_state.model_building=1
        
    
        if "preprocessing" not in st.session_state:
            st.session_state.preprocessing = 0
        if "xtrain" not in st.session_state:
            st.session_state.xtrain = 0
        if "xtest" not in st.session_state:
            st.session_state.xtest = 0
        if "ytrain" not in st.session_state:
            st.session_state.ytrain = 0
        if "ytest" not in st.session_state:
            st.session_state.ytest = 0

        
        st.warning('We are using the same csv for model building')
        #new_data_file= st.file_uploader("Upload a csv file", type=["csv"])
        df=pd.read_csv('LoanApprovalPrediction.csv')
        if st.checkbox('Lets Start'):

            if df is not None:
                if 'model_building' not in st.session_state:
                    st.session_state.model_building=0
                                
                
                    
                columns=df.columns
                    
                if st.checkbox('Checking column names of your df'):
                    st.write(columns)
                    
                null_values=df.isna().sum().sum()
                if st.checkbox('Checking null values in the data frame'):
                    if null_values==0:
                        st.write("Your data set doesn't contain any null values")
                    else:
                        st.write('Null values by column wise:')
                        st.write(df.isna().sum())
                
                if st.checkbox('Dropping the column Loan ID'):
                    df.drop(columns=['Loan_ID'],axis=1,inplace=True)
                    st.success('Successfully the Loan ID column deleted!')
                
                    
                df1=df.copy()
                if st.checkbox('After deletion'):
                                    
                        
                    st.write('New DataFrame')
                    st.write(df1)
                    
                                
                
                    
                catcols=df1.select_dtypes(include='object')
                catcols=catcols.columns.tolist()
                catcols=', '.join(catcols)
                catcols=catcols.split(', ')
                catcols=catcols[:len(catcols)-1]
                numcols=df1.select_dtypes(include='number')
                numcols=list(numcols.columns)   
                numcols=', '.join(numcols)
                numcols=numcols.split(', ')
                    
                if st.checkbox('Checking categorical cols in new df'):
                    st.write(catcols)
                
                if st.checkbox('Checking numerical cols in new df'):
                    st.write(numcols)
                    
                if st.checkbox('Filling the null values with most_frequent of independent feature'):
                    df1['Loan_Status']=df1['Loan_Status'].fillna(df1['Loan_Status'].mode()[0])
                    st.write('The Independent feature of the data frame is:')
                    st.write(df['Loan_Status'])
                    count=df1['Loan_Status'].isna().sum()
                    st.write('Count of null values in independent feature is: ',count)
                
                
                    
                df1['ApplicantIncome']=df1['ApplicantIncome'].astype('float64')

                if st.checkbox('Seperating the independent and dependent features'):
                    x=df1.drop('Loan_Status',axis=1)
                    y=df1['Loan_Status']
                    dependent_features=x
                    independent_features=y
                    st.write('Dependent Features are as follows:')
                    st.write(dependent_features)
                    st.write('Independent Features are as follows:')
                    st.write(independent_features)
                
                if st.checkbox('Splitting data into training and testing'):
                                    
                    st.info('We are using 20 percent of data for testing')
                        
                                    
                                    
                    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=50)
                    st.write('Shape of xtrain {}'.format(xtrain.shape))
                    st.write('Shape of xtest {}'.format(xtest.shape))
                    st.write('Shape of ytrain {}'.format(ytrain.shape))
                    st.write('Shape of ytest {}'.format(ytest.shape))
                    st.session_state.xtrain=xtrain
                    st.session_state.xtest=xtest
                    st.session_state.ytrain=ytrain
                    st.session_state.ytest=ytest
                
                if st.checkbox('PipeLine Building for treating missing values and  model deploying'):
                    numerical=['mean','median']
                    categorical=['most_frequent','constant']
                    ns=st.radio('Filling for numerical columns',numerical)
                    cs=st.radio('Filling for numerical columns',categorical)
                
                if st.checkbox('Numerical Pipeline building for treating np.nan values'):
                    numerical_cols=Pipeline(
                                        steps=[
                                        ('Filling missing values with {}'.format(ns),SimpleImputer(strategy=ns)),
                                        ('Scaler',StandardScaler()),
                                        ]
                                    )
                    st.success('Successfully numerical pipleline is built!')
                
                if st.checkbox('Categorical Pipeline building for treating np.nan values'):
                    categorical_cols=Pipeline(
                                        steps=[
                                        ('Filling missing values with {}'.format(cs),SimpleImputer(strategy=cs)),
                                        ('Encoding',OneHotEncoder(handle_unknown='ignore')),
                                        ]
                                    )
                    st.success('Successfully categorical pipleline is built!')
                    categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
                    numeric_features = ['ApplicantIncome','Dependents','CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
                
                if st.checkbox('Combing both transformers using column transformers'):
                    preprocessing=ColumnTransformer(
                                    [
                                        ('categorical columns',categorical_cols,categorical_features),
                                        ('numerical columns',numerical_cols,numeric_features),

                                        ]


                                    )
                    st.session_state.preprocessing=preprocessing
                    st.success('Column tranformers are built')
                    st.info('Lets jump into deploying by clickng in the sub page menu!')
                st.session_state.model_building=1

           
        
        
                    
            
        
    else:
        st.warning('Please visit eda page!')
    
        
    
        
        

        
        
        












def deploying_without_tuning():
    st.warning('''Please note that all checkboxes and buttons across sub-menus are interlinked. 
                    If any of them are not activated, the results may not appear as expected. We recommend ensuring that all checkboxes and buttons are properly activated before proceeding.''')
    if 'model_building' not in st.session_state:
        st.session_state.model_building=0
    model_building=st.session_state.model_building
    if model_building==1:
        pass
        if "ac" not in st.session_state:
            st.session_state.ac = 0
        if "f1score" not in st.session_state:
            st.session_state.f1score = 0
        if "recall_score" not in st.session_state:
            st.session_state.recall_score = 0
        if "precision_score" not in st.session_state:
            st.session_state.precision_score = 0
        if 'lr' not in st.session_state:
            st.session_state.lr=0
        if 'dt' not in st.session_state:
            st.session_state.dt=0
        if 'knn' not in st.session_state:
            st.session_state.knn=0
        if 'gnb' not in st.session_state:
            st.session_state.gnb=0
        if 'svm' not in st.session_state:
            st.session_state.svm=0
        if 'rf' not in st.session_state:
            st.session_state.rf=0
        if 'gb' not in st.session_state:
            st.session_state.gb=0
        if 'high' not in st.session_state:
            st.session_state.high=0
        if 'lrp' not in st.session_state:
            st.session_state.lrp=0
        if 'dtp' not in st.session_state:
            st.session_state.dtp=0
        if 'knnp' not in st.session_state:
            st.session_state.knnp=0
        if 'gnbp' not in st.session_state:
            st.session_state.gnbp=0
        if 'svmp' not in st.session_state:
            st.session_state.svmp=0
        if 'rfp' not in st.session_state:
            st.session_state.rfp=0
        if 'gbp' not in st.session_state:
            st.session_state.gbp=0

        
    
        st.info('We are building models without any hyper parameters')
        models_menu=['','Logistic Regression','Decision Tree','KNN','Gaussian NB','SVM','Random Forest','Gradient Boosting']
        selection=st.selectbox('Choose the following normal models',models_menu)
        if "preprocessing" not in st.session_state:
            st.session_state.preprocessing = 0
        if "xtrain" not in st.session_state:
            st.session_state.xtrain = 0
        if "xtest" not in st.session_state:
            st.session_state.xtest = 0
        if "ytrain" not in st.session_state:
            st.session_state.ytrain = 0
        if "ytest" not in st.session_state:
            st.session_state.ytest = 0
        preprocessing=st.session_state.preprocessing
        xtrain=st.session_state.xtrain
        xtest=st.session_state.xtest
        ytrain=st.session_state.ytrain
        ytest=st.session_state.ytest
    
        if selection=='Logistic Regression':
            
            
            
        
            
            lr=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',LogisticRegression())
                                    ])
            
            lr.fit(xtrain,ytrain)
            
            ypred=lr.predict(xtest)
        
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.lr=lr
            
            
        elif selection=='Decision Tree':
            dt=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',DecisionTreeClassifier())
                                    ])
            dt.fit(xtrain,ytrain)
            ypred=dt.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.dt=dt
        
        elif selection=='KNN':
            knn=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',KNeighborsClassifier())
                                    ])
            knn.fit(xtrain,ytrain)
            ypred=knn.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.knn=knn
            
        elif selection=='Gaussian NB':
            gnb=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',GaussianNB())
                                    ])
            gnb.fit(xtrain,ytrain)
            ypred=gnb.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.gnb=gnb
            
        elif selection=='SVM':
            svm=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',SVC())
                                    ])
            svm.fit(xtrain,ytrain)
            ypred=svm.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
           # st.session_state.svm=svm
            
        elif selection=='Random Forest':
            rf=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',RandomForestClassifier())
                                    ])
            rf.fit(xtrain,ytrain)
            ypred=rf.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.rf=rf
            
        

            
        elif selection=='Gradient Boosting':
            gb=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',GradientBoostingClassifier())
                                    ])
            gb.fit(xtrain,ytrain)
            ypred=gb.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.gb=gb
        if 'c1_value' not in st.session_state:
            st.session_state.c1_value=0
        new_heading='Comparision of all the above algorithm results'
        c1_value=st.checkbox(new_heading)
        if c1_value:
            models=[
                                        LogisticRegression(),
                                        DecisionTreeClassifier(),
                                        KNeighborsClassifier(),
                                        GaussianNB(),
                                        SVC(),
                                        RandomForestClassifier(),
                                        
                                        GradientBoostingClassifier()
                                    ]
            ac={}
            f1score={}
            recall_score={}
            precision_score={}
            
            for model in models:
                


                pipe= Pipeline(steps=[('preprocessing', preprocessing),
                                                        ('regression_model', model)])
                
                pipe.fit(xtrain,ytrain)
                ypred=pipe.predict(xtest)
                
                
                cr=classification_report(ytest,ypred,output_dict=True)
                cr1=classification_report(ytest,ypred)
                accuracy = cr['accuracy']*100
                ac[str(model)]=accuracy
                f1 = cr['Y']['f1-score']
                f1score[str(model)]=f1
                r1 = cr['Y']['recall']
                recall_score[str(model)]=r1
                p1 = cr['Y']['precision']
                precision_score[str(model)]=p1

                st.write('Models is: {}'.format(model))
                st.write('Accuracy {}'.format(accuracy))
                st.write('F1-Score {}'.format(f1))
                st.write('Recall Score {}'.format(r1))
                st.write('Precision Score {}'.format(p1))
                if model.__class__.__name__ == 'LogisticRegression':
                    st.session_state.lrp= ypred
                elif model.__class__.__name__ == 'DecisionTreeClassifier':
                    st.session_state.dtp = ypred
                elif model.__class__.__name__ == 'KNeighborsClassifier':
                    st.session_state.knnp = ypred
                elif model.__class__.__name__ == 'GaussianNB':
                    st.session_state.gnbp = ypred
                elif model.__class__.__name__ == 'SVC':
                    st.session_state.svmp = ypred
                elif model.__class__.__name__ == 'RandomForestClassifier':
                    st.session_state.rfp = ypred
                elif model.__class__.__name__ == 'GradientBoostingClassifier':
                    st.session_state.gbp = ypred
                
                



                st.write('-' * 60)
                if model.__class__.__name__ == 'LogisticRegression':
                    st.session_state.lr= pipe
                elif model.__class__.__name__ == 'DecisionTreeClassifier':
                    st.session_state.dt = pipe
                elif model.__class__.__name__ == 'KNeighborsClassifier':
                    st.session_state.knn = pipe
                elif model.__class__.__name__ == 'GaussianNB':
                    st.session_state.gnb = pipe
                elif model.__class__.__name__ == 'SVC':
                    st.session_state.svm = pipe
                elif model.__class__.__name__ == 'RandomForestClassifier':
                    st.session_state.rf = pipe
                elif model.__class__.__name__ == 'GradientBoostingClassifier':
                    st.session_state.gb = pipe
                
               
                
                
            
            
            st.session_state.ac=ac
            st.session_state.f1score=f1score
            st.session_state.recall_score=recall_score
            st.session_state.precision_score=precision_score
            st.session_state.c1_value=1
        if 'c2_value' not in st.session_state:
            st.session_state.c2_value=0
        new='Click to see the viusalizations of all models accuracy, f1-score, precision score and recall score!'
        c2_value=st.checkbox(new)
        if c2_value:
            small_select=['','Comparision between algorithms accuracy','Comparision between algorithms f1,recall and precision score']
            small_result_set=st.selectbox('Choose the following visualizations:',small_select)
            if small_result_set=='':
                pass
            elif small_result_set=='Comparision between algorithms accuracy':
                ac_scores_before_tuning = [j for i, j in enumerate(ac.values())]
                ac_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Random Forest','GradientBoostingClassifier']
                sorted_indices = np.argsort(ac_scores_before_tuning)
                ac_scores_before_tuning_sorted = [ac_scores_before_tuning[i] for i in sorted_indices]
                ac_labels_sorted = [ac_labels[i] for i in sorted_indices]

                pos = np.arange(len(ac_labels_sorted))
                width = 0.25
                fig, ax = plt.subplots(figsize=(20, 10))
                
                #ax.spines['bottom']
                #ax.spines['left']
                
                ax.tick_params(axis='y',labelsize=30)

                #ax.yaxis.label.set_color('white')
                #ax.xaxis.label.set_color('white')
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color('black')

                rects1 = ax.bar(pos - width/2, ac_scores_before_tuning_sorted, width, label='Before Tuning',color='red')
                ax.set_xticks(pos)
                ax.set_xticklabels(ac_labels_sorted, rotation=45, ha='right',size=30)
                ax.tick_params(axis='x',labelsize=30)
                

                ax.set_ylabel('Accuracy Score',size=40)
                ax.set_ylim([0, 100])
                ax.set_title('Comparison of all algorithms',size=40)
                ax.legend(fontsize=20)
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate('{:.2f}'.format(height),
                                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                                    xytext=(0, 3),
                                                    textcoords="offset points",
                                                    ha='center', va='bottom',rotation=40,size=30)
                autolabel(rects1)
                st.pyplot(fig)
                if 'b2' not in st.session_state:
                    st.session_state.b2=0
                heading2='Click to see the most efficient algorithm for the problem'
                b2=st.checkbox(heading2)

                if b2:
                    high=ac_labels_sorted[-1]
                    st.session_state.high=high
                    with st.spinner('Analyzing data...'):
                        time.sleep(5)
                    
                    st.markdown(
                        f"""
                        <style>
                        .algorithm-name {{
                            color: #0072C6;
                            font-weight: bold;
                            font-size: 24px;
                            margin: 0 0 10px 0;
                        }}
                        
                        .bomb {{
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            animation: explode 0.5s ease-in-out 4s forwards;
                        }}

                        @keyframes explode {{
                            0% {{
                                transform: scale(1);
                                opacity: 1;
                            }}
                            100% {{
                                transform: scale(10);
                                opacity: 0;
                            }}
                        }}
                        
                        .animate__animated {{
                            animation-duration: 1s;
                            animation-fill-mode: both;
                        }}
                        
                        .animate__zoomIn {{
                            animation-name: zoomIn;
                        }}
                        
                        @keyframes zoomIn {{
                            from {{
                                opacity: 0;
                                transform: scale3d(0.3, 0.3, 0.3);
                            }}
                        
                            50% {{
                                opacity: 1;
                            }}
                        
                            to {{
                                transform: scale3d(1, 1, 1);
                            }}
                        }}
                        </style>
                        
                        <div class="animate__animated animate__zoomIn">
                            <h2>Awesome! You found the best algorithm.</h2>
                            <p>The highest accuracy score was gained by <span class="algorithm-name">{high}</span> algorithm.</p>
                        </div>
                        <div class="bomb"></div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.balloons()
                    st.session_state.b2=1
            elif small_result_set=='Comparision between algorithms f1,recall and precision score':
                plt.style.use('dark_background')
                ac1_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Random Forest','GradientBoostingClassifier']
                f1_scores_before_tuning = [j for i, j in enumerate(f1score.values())]
                fig, ax = plt.subplots(3,1,figsize=(16,32))
                fig.subplots_adjust(hspace=1.0)
                                
                                    
                ax[0].plot( f1_scores_before_tuning, label='Before Tuning',linewidth=5,c='red')
                ax[0].set_title('Comparison of F1 Scores Before Tuning',fontsize=40)
                ax[0].tick_params(axis='x',labelsize=20)
                ax[0].tick_params(axis='y',labelsize=20)

                ax[0].legend(fontsize=20)
                ax[0].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
                recall_before_tuning = [j for i, j in enumerate(recall_score.values())]
                ax[1].plot(recall_before_tuning, label='Before Tuning',linewidth=5,c='red')
                ax[1].set_title('Comparison of Recall Before Tuning',fontsize=40)
                ax[1].tick_params(axis='x',labelsize=20)
                ax[1].tick_params(axis='y',labelsize=20)

                ax[1].legend(fontsize=20)
                ax[1].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
                precision_before_tuning = [j for i, j in enumerate(precision_score.values())]
                ax[2].plot(precision_before_tuning, label='Before Tuning',linewidth=5,c='red')
                ax[2].set_title('Comparison of Precision Before Tuning',fontsize=40)
                ax[2].tick_params(axis='x',labelsize=20)
                ax[2].tick_params(axis='y',labelsize=20)

                ax[2].legend(fontsize=20)
                ax[2].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
                for i in range(len(ax)):
                    ax[i].set_xticks(range(len(ac1_labels)))
                    ax[i].set_xticklabels(ac1_labels, rotation=45, ha='right')
                
                    
                st.pyplot(fig)
        st.session_state.c2_value=1
    else:
        st.warning('First visit the model building page')
        
    
    


def deploying_with_tuning():
    st.warning('''Please note that all checkboxes and buttons across sub-menus are interlinked. 
                    If any of them are not activated, the results may not appear as expected. We recommend ensuring that all checkboxes and buttons are properly activated before proceeding.''')
    if 'model_building' not in st.session_state:
        st.session_state.model_building=0
    model_building=st.session_state.model_building
    if model_building==1:
        pass
        if 'without_parameters' not in st.session_state:
            st.session_state.without_parameters=0
        
        if "ac1" not in st.session_state:
            st.session_state.ac1 = 0
        if "f1score1" not in st.session_state:
            st.session_state.f1score1 = 0
        if "recall_score1" not in st.session_state:
            st.session_state.recall_score1 = 0
        if "precision_score1" not in st.session_state:
            st.session_state.precision_score1 = 0
        if 'lr1' not in st.session_state:
            st.session_state.lr1=0
        if 'dt1' not in st.session_state:
            st.session_state.dt1=0
        if 'knn1' not in st.session_state:
            st.session_state.knn1=0
        if 'gnb1' not in st.session_state:
            st.session_state.gnb1=0
        if 'svm1' not in st.session_state:
            st.session_state.svm1=0
        if 'rf1' not in st.session_state:
            st.session_state.rf1=0
        if 'gb1' not in st.session_state:
            st.session_state.gb1=0
        if 'high1' not in st.session_state:
            st.session_state.high1=0
        if 'lrp1' not in st.session_state:
            st.session_state.lrp1=0
        if 'dtp1' not in st.session_state:
            st.session_state.dtp1=0
        if 'knnp1' not in st.session_state:
            st.session_state.knnp1=0
        if 'gnbp1' not in st.session_state:
            st.session_state.gnbp1=0
        if 'svmp1' not in st.session_state:
            st.session_state.svmp1=0
        if 'rfp1' not in st.session_state:
            st.session_state.rfp1=0
        if 'gbp1' not in st.session_state:
            st.session_state.gbp1=0
        st.info('We are building models with some hyper parameters')
        models_menu1=['','Logistic Regression','Decision Tree','KNN','Gaussian NB','SVM','Random Forest','Gradient Boosting']
        selection1=st.selectbox('Choose the following parameterized models',models_menu1)
        if "preprocessing" not in st.session_state:
            st.session_state.preprocessing = 0
        if "xtrain" not in st.session_state:
            st.session_state.xtrain = 0
        if "xtest" not in st.session_state:
            st.session_state.xtest = 0
        if "ytrain" not in st.session_state:
            st.session_state.ytrain = 0
        if "ytest" not in st.session_state:
            st.session_state.ytest = 0
        preprocessing=st.session_state.preprocessing
        xtrain=st.session_state.xtrain
        xtest=st.session_state.xtest
        ytrain=st.session_state.ytrain
        ytest=st.session_state.ytest
        if selection1=='Logistic Regression':
                            
            lr1=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',LogisticRegression(
                                        C=0.01,
                                        class_weight='balanced',
                                        fit_intercept=False,
                                        max_iter=1000,
                                        multi_class='multinomial',
                                        penalty='l2',
                                        solver='lbfgs',
                                        tol=0.0001
                                        ))
                                    ])
            lr1.fit(xtrain,ytrain)
            ypred=lr1.predict(xtest)

            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.lr1=lr1
        elif selection1=='Decision Tree':
            dt1=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',DecisionTreeClassifier(
                                        ccp_alpha=0.0,
                                        criterion='gini',
                                        max_depth=4,
                                        max_features='sqrt',
                                        min_samples_leaf=1,
                                        min_samples_split=2,
                                        random_state=456
                                        ))
                                    ])
            dt1.fit(xtrain,ytrain)
            ypred=dt1.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.dt1=dt1
        elif selection1=='KNN':
            knn1=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',KNeighborsClassifier(
                                        algorithm='ball_tree',
                                            leaf_size=20,
                                            metric='euclidean',
                                            n_neighbors=7,
                                            p=1,
                                            weights='uniform'
                                        ))
                                    ])
            knn1.fit(xtrain,ytrain)
            ypred=knn1.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.knn1=knn1
        elif selection1=='Gaussian NB':
            gnb1=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',GaussianNB(
                                        var_smoothing= 1e-09
                                        ))
                                    ])
            gnb1.fit(xtrain,ytrain)
            ypred=gnb1.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.gnb1=gnb1
        elif selection1=='SVM':
            svm1=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',SVC(
                                        C=1, degree=3, gamma='auto', kernel='poly', shrinking=True
                                        ))
                                    ])
            svm1.fit(xtrain,ytrain)
            ypred=svm1.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.svm1=svm1
        elif selection1=='Random Forest':
            rf1=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',RandomForestClassifier(
                                        bootstrap=True,
                                        max_depth=None,
                                        max_features='auto',
                                        min_samples_split=10,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=200
                                        ))
                                    ])
            rf1.fit(xtrain,ytrain)
            ypred=rf1.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
            #st.session_state.rf1=rf1
        elif selection1=='Gradient Boosting':
            gb1=Pipeline(
                            steps=[
                                        ('preprocessing',preprocessing),
                                        ('classification model',GradientBoostingClassifier(
                                        learning_rate=0.01,
                                        loss='deviance',
                                        max_depth=7,
                                        min_samples_leaf=4,
                                        min_samples_split=5,
                                        subsample=0.7
                                        ))
                                    ])
            gb1.fit(xtrain,ytrain)
            ypred=gb1.predict(xtest)
            cr=classification_report(ytest,ypred,output_dict=True)
            cr1=classification_report(ytest,ypred)
            cm=confusion_matrix(ytest,ypred)
            st.write('Classificaton Report :')
            st.write(cr)
            ac=cr['accuracy']*100
                        
            st.write('Accuracy is: {}'.format(ac))
                            
                            
            pscore=cr['Y']['precision']
            rscore=cr['Y']['recall']
            f1score=cr['Y']['f1-score']
            st.write('Precession score: {}'.format(pscore))
            st.write('Recall score: {}'.format(rscore))
            st.write('F1-Score: {}'.format(f1score))
            st.write('Confusion Matrix')
            cm=confusion_matrix(ytest,ypred)
            st.write(cm)
           # st.session_state.gb1=gb1
        if 'c3_value' not in st.session_state:
            st.session_state.c3_value=0
        new_heading1='Comparision of all the above parameterized algorithm results'
        c3_value=st.checkbox(new_heading1)
        #new_heading='Comparision of all the above parameterized algorithm results'
        if c3_value:
            models1=[
                                        LogisticRegression(),
                                        DecisionTreeClassifier(),
                                        KNeighborsClassifier(),
                                        GaussianNB(),
                                        SVC(),
                                        RandomForestClassifier(),
                                        
                                        GradientBoostingClassifier()
                                    ]
            
            models=[
                            LogisticRegression(
                                                    C=0.01,
                                                    class_weight='balanced',
                                                    fit_intercept=False,
                                                    max_iter=1000,
                                                    multi_class='multinomial',
                                                    penalty='l2',
                                                    solver='lbfgs',
                                                    tol=0.0001
                            ),
                            DecisionTreeClassifier(
                                                    ccp_alpha=0.0,
                                                    criterion='gini',
                                                    max_depth=4,
                                                    max_features='sqrt',
                                                    min_samples_leaf=1,
                                                    min_samples_split=2,
                                                    random_state=456
                            ),
                            KNeighborsClassifier(
                                                    algorithm='ball_tree',
                                                    leaf_size=20,
                                                    metric='euclidean',
                                                    n_neighbors=7,
                                                    p=1,
                                                    weights='uniform'
                                                        ),
                            GaussianNB(var_smoothing= 1e-09),
                            SVC(C=1, degree=3, gamma='auto', kernel='poly', shrinking=True),
                            RandomForestClassifier(
                                                    bootstrap=True,
                                                    max_depth=None,
                                                    max_features='auto',
                                                    min_samples_split=10,
                                                    min_weight_fraction_leaf=0.0,
                                                    n_estimators=200
                            ),
                            
                            GradientBoostingClassifier(
                                                    learning_rate=0.01,
                                                    loss='deviance',
                                                    max_depth=7,
                                                    min_samples_leaf=4,
                                                    min_samples_split=5,
                                                    subsample=0.7
                            )
                        ]
            ac1={}
            f1score1={}
            recall_score1={}
            precision_score1={}
            
            
            for model in models:
                pipe = Pipeline(steps=[('preprocessing', preprocessing),
                                                    ('regression_model', model)])
            
                pipe.fit(xtrain,ytrain)

                ypred=pipe.predict(xtest)
                cr=classification_report(ytest,ypred,output_dict=True)
                cr1=classification_report(ytest,ypred)
                accuracy = cr['accuracy']*100
                ac1[str(model)]=accuracy
                f1 = cr['Y']['f1-score']
                f1score1[str(model)]=f1
                r1 = cr['Y']['recall']
                recall_score1[str(model)]=r1
                p1 = cr['Y']['precision']
                precision_score1[str(model)]=p1
                st.write('Models is: {}'.format(model))
                st.write('Accuracy {}'.format(accuracy))
                st.write('F1-Score {}'.format(f1))
                st.write('Recall Score {}'.format(r1))
                st.write('Precision Score {}'.format(p1))
                if model.__class__.__name__ == 'LogisticRegression':
                    st.session_state.lrp1= ypred
                elif model.__class__.__name__ == 'DecisionTreeClassifier':
                    st.session_state.dtp1 = ypred
                elif model.__class__.__name__ == 'KNeighborsClassifier':
                    st.session_state.knnp1 = ypred
                elif model.__class__.__name__ == 'GaussianNB':
                    st.session_state.gnbp1 = ypred
                elif model.__class__.__name__ == 'SVC':
                    st.session_state.svmp1 = ypred
                elif model.__class__.__name__ == 'RandomForestClassifier':
                    st.session_state.rfp1 = ypred
                elif model.__class__.__name__ == 'GradientBoostingClassifier':
                    st.session_state.gbp1 = ypred
            
            
    
            



                st.write('-' * 60)
                if model.__class__.__name__ == 'LogisticRegression':
                    st.session_state.lr1 = pipe
                elif model.__class__.__name__ == 'DecisionTreeClassifier':
                    st.session_state.dt1 = pipe
                elif model.__class__.__name__ == 'KNeighborsClassifier':
                    st.session_state.knn1 = pipe
                elif model.__class__.__name__ == 'GaussianNB':
                    st.session_state.gnb1 = pipe
                elif model.__class__.__name__ == 'SVC':
                    st.session_state.svm1 = pipe
                elif model.__class__.__name__ == 'RandomForestClassifier':
                    st.session_state.rf1 = pipe
                elif model.__class__.__name__ == 'GradientBoostingClassifier':
                    st.session_state.gb1 = pipe
                
                st.session_state.without_parameters=1
            st.session_state.ac1=ac1
            st.session_state.f1score1=f1score1
            st.session_state.recall_score1=recall_score1
            st.session_state.precision_score1=precision_score1
            st.session_state.c3_value=1
        if 'c4_value' not in st.session_state:
            st.session_state.c4_value=0
        new1='Click to see the viusalizations of all parameterized models accuracy, f1-score, precision score and recall score!'
        c4_value=st.checkbox(new1)
        if c4_value:
            small_select=['','Comparision between parameterized algorithms accuracy','Comparision between parameterized algorithms f1,recall and precision score']
            small_result_set=st.selectbox('Choose the following visualizations:',small_select)
            if small_result_set=='':
                pass
            elif small_result_set=='Comparision between parameterized algorithms accuracy':
                ac_scores_before_tuning = [j for i, j in enumerate(ac1.values())]
                ac_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Random Forest','GradientBoostingClassifier']
                sorted_indices = np.argsort(ac_scores_before_tuning)
                ac_scores_before_tuning_sorted = [ac_scores_before_tuning[i] for i in sorted_indices]
                ac_labels_sorted = [ac_labels[i] for i in sorted_indices]

                pos = np.arange(len(ac_labels_sorted))
                width = 0.25
                fig, ax = plt.subplots(figsize=(20, 10))
                
                #ax.spines['bottom'] s
                #ax.spines['left']
                ax.tick_params(axis='x')
                ax.tick_params(axis='y')

                #ax.yaxis.label.set_color('white')
                #ax.xaxis.label.set_color('white')
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color('black')

                rects1 = ax.bar(pos - width/2, ac_scores_before_tuning_sorted, width, label='After Tuning')
                ax.set_xticks(pos)
                ax.set_xticklabels(ac_labels_sorted, rotation=45, ha='right',size=30)
                ax.tick_params(axis='x',labelsize=30)
                ax.tick_params(axis='y',labelsize=30)

                ax.set_ylabel('Accuracy Score',size=40)
                ax.set_ylim([0, 100])
                ax.set_title('Comparison of all algorithms accuracy',size=40)
                ax.legend(fontsize=20)
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate('{:.2f}'.format(height),
                                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                                    xytext=(0, 3),
                                                    textcoords="offset points",
                                                    ha='center', va='bottom',rotation=40,size=30)
                autolabel(rects1)
                st.pyplot(fig)
                if 'b1' not in st.session_state:
                    st.session_state.b1=0
                heading1='Click to see the most efficient algorithm for the problem'
                b1=st.checkbox(heading1)
                if b1:
                    high1=ac_labels_sorted[-1]
                    st.session_state.high1=high1
                    with st.spinner('Analyzing data...'):
                        time.sleep(5)
                    
                    st.markdown(
                        f"""
                        <style>
                        .algorithm-name {{
                            color: #0072C6;
                            font-weight: bold;
                            font-size: 24px;
                            margin: 0 0 10px 0;
                        }}
                        
                        .bomb {{
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            animation: explode 0.5s ease-in-out 4s forwards;
                        }}

                        @keyframes explode {{
                            0% {{
                                transform: scale(1);
                                opacity: 1;
                            }}
                            100% {{
                                transform: scale(10);
                                opacity: 0;
                            }}
                        }}
                        
                        .animate__animated {{
                            animation-duration: 1s;
                            animation-fill-mode: both;
                        }}
                        
                        .animate__zoomIn {{
                            animation-name: zoomIn;
                        }}
                        
                        @keyframes zoomIn {{
                            from {{
                                opacity: 0;
                                transform: scale3d(0.3, 0.3, 0.3);
                            }}
                        
                            50% {{
                                opacity: 1;
                            }}
                        
                            to {{
                                transform: scale3d(1, 1, 1);
                            }}
                        }}
                        </style>
                        
                        <div class="animate__animated animate__zoomIn">
                            <h2>Awesome! You found the best algorithm.</h2>
                            <p>The highest accuracy score was gained by <span class="algorithm-name">{high1}</span> algorithm.</p>
                        </div>
                        <div class="bomb"></div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.balloons()
                    st.session_state.b1=1
            elif small_result_set=='Comparision between parameterized algorithms f1,recall and precision score':
                plt.style.use('dark_background')
                ac1_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Random Forest','GradientBoostingClassifier']
                f1_scores_before_tuning = [j for i, j in enumerate(f1score1.values())]
                fig, ax = plt.subplots(3,1,figsize=(16,32))
                fig.subplots_adjust(hspace=1.0)
                                
                                    
                ax[0].plot( f1_scores_before_tuning, label='After Tuning',linewidth=5,c='red')
                ax[0].set_title('Comparison of F1 Scores after Tuning',fontsize=40)
                ax[0].tick_params(axis='x',labelsize=20)
                ax[0].tick_params(axis='y',labelsize=20)

                ax[0].legend(fontsize=20)
                ax[0].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
                recall_before_tuning = [j for i, j in enumerate(recall_score1.values())]
                ax[1].plot(recall_before_tuning, label='After Tuning',linewidth=5,c='red')
                ax[1].set_title('Comparison of Recall after Tuning',fontsize=40)
                ax[1].tick_params(axis='x',labelsize=20)
                ax[1].tick_params(axis='y',labelsize=20)

                ax[1].legend(fontsize=20)
                ax[1].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
                precision_before_tuning = [j for i, j in enumerate(precision_score1.values())]
                ax[2].plot(precision_before_tuning, label='After Tuning',linewidth=5,c='red')
                ax[2].set_title('Comparison of Precision after Tuning',fontsize=40)
                ax[2].tick_params(axis='x',labelsize=20)
                ax[2].tick_params(axis='y',labelsize=20)

                ax[2].legend(fontsize=20)
                ax[2].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
                for i in range(len(ax)):
                    ax[i].set_xticks(range(len(ac1_labels)))
                    ax[i].set_xticklabels(ac1_labels, rotation=45, ha='right')
                
                    
                st.pyplot(fig)
            #st.write('Lets do the prediction')
        st.session_state.c4_value=1
    else:
        st.warning('Please visit Model Building')

    
    
def non_tuning_vs_tuning():
    st.warning('''Please note that all checkboxes and buttons across sub-menus are interlinked. 
                    If any of them are not activated, the results may not appear as expected. We recommend ensuring that all checkboxes and buttons are properly activated before proceeding.''')
    if 'model_building' not in st.session_state:
        st.session_state.model_building=0
    
    model_building=st.session_state.model_building
    if model_building==1:
        pass
        if "ac" not in st.session_state:
            st.session_state.ac = 0
        if "f1score" not in st.session_state:
            st.session_state.f1score = 0
        if "recall_score" not in st.session_state:
            st.session_state.recall_score = 0
        if "precision_score" not in st.session_state:
            st.session_state.precision_score = 0
        if "ac1" not in st.session_state:
            st.session_state.ac1 = 0
        if "f1score1" not in st.session_state:
            st.session_state.f1score1 = 0
        if "recall_score1" not in st.session_state:
            st.session_state.recall_score1 = 0
        if "precision_score1" not in st.session_state:
            st.session_state.precision_score1 = 0
        ac=st.session_state.ac
        f1score=st.session_state.f1score
        recall_score=st.session_state.recall_score
        precision_score=st.session_state.precision_score
        ac1=st.session_state.ac1
        f1score1=st.session_state.f1score1
        recall_score1=st.session_state.recall_score1
        precision_score1=st.session_state.precision_score1
        
        list=['','Comparison of accuracies between non-tuning vs tuning','Comparision between algorithms f1,recall and precision score']
        checking=st.selectbox('Choose the following visualizations:',list)
        if checking=='':
            pass
        elif checking=='Comparison of accuracies between non-tuning vs tuning':
            sns.set_style('darkgrid')
            # Accuracy scores before tuning
            ac_scores_before_tuning = [j for i, j in enumerate(ac.values())]
            ac_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Random Forest','GradientBoostingClassifier']
            # Accuracy scores after tuning
            ac_scores_after_tuning = [j for i, j in enumerate(ac1.values())]

            ac1_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Random Forest','GradientBoostingClassifier']

            # Set the positions of the bars on the x-axis
            pos = np.arange(len(ac_labels))

            # Set the width of the bars
            width = 0.25

            # Create a figure and axis object
            fig, ax = plt.subplots(figsize=(16, 8))

            # Plot the before tuning bars
            rects1 = ax.bar(pos - width/2, ac_scores_before_tuning, width, label='Before Tuning',color='red')

            # Plot the after tuning bars
            rects2 = ax.bar(pos + width/2, ac_scores_after_tuning, width, label='After Tuning',color='blue')

            # Set the x-axis labels and tick marks
            ax.set_xticks(pos)
            ax.set_xticklabels(ac_labels, rotation=45, ha='right',fontsize=20,color='white')

            # Set the y-axis label and limit
            ax.set_ylabel('Accuracy Score',fontsize=20,color='white')
            ax.set_ylim([0, 100])
            ax.tick_params(axis='y',labelsize=20,colors='white')

            # Set the plot title and legend
            ax.set_title('Comparison of all algorithms accuracy',color='white',fontsize=30)
            ax.legend(fontsize=18)

            # Add labels for the bar heights
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{:.2f}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',rotation=90,size=20)

            # Add the bar height labels
            autolabel(rects1)
            autolabel(rects2)

            st.pyplot(fig)
        elif checking=='Comparision between algorithms f1,recall and precision score':
            plt.style.use('dark_background')
            ac2_labels= ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Random Forest','GradientBoostingClassifier']

            f1_scores_before_tuning = [j for i, j in enumerate(f1score.values())]
            f1_scores_after_tuning = [j for i, j in enumerate(f1score1.values())]

            fig, ax = plt.subplots(3,1,figsize=(16,32))
            fig.subplots_adjust(hspace=1.0)
            plt.rcParams['grid.color'] = 'black'
            sns.set_style('darkgrid')

            ax[0].plot( f1_scores_before_tuning, label='Before Tuning',linewidth=5,c='red')
            ax[0].plot(f1_scores_after_tuning, label='After Tuning',linewidth=5,c='blue')
            ax[0].set_title('Comparison of F1 Scores Before and After Tuning',fontsize=30)
            ax[0].tick_params(axis='x',labelsize=20)
            ax[0].tick_params(axis='y',labelsize=20)

            ax[0].legend(fontsize=18)
            ax[0].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)

            recall_before_tuning = [j for i, j in enumerate(recall_score.values())]
            recall_after_tuning = [j for i, j in enumerate(recall_score1.values())]
            ax[1].plot(recall_before_tuning, label='Before Tuning',linewidth=5,c='red')
            ax[1].plot(recall_after_tuning, label='After Tuning',linewidth=5,c='blue')
            ax[1].set_title('Comparison of Recall Before and After Tuning',fontsize=30)
            ax[1].tick_params(axis='x',labelsize=20)
            ax[1].tick_params(axis='y',labelsize=20)

            ax[1].legend(fontsize=18)
            ax[1].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)

            precision_before_tuning = [j for i, j in enumerate(precision_score.values())]
            precision_after_tuning = [j for i, j in enumerate(precision_score1.values())]
            ax[2].plot(precision_before_tuning, label='Before Tuning',linewidth=5,c='red')
            ax[2].plot(precision_after_tuning, label='After Tuning',linewidth=5,c='blue')
            ax[2].set_title('Comparison of Precision Before and After Tuning',fontsize=30)
            ax[2].tick_params(axis='x',labelsize=20)
            ax[2].tick_params(axis='y',labelsize=20)

            ax[2].legend(fontsize=18)
            ax[2].grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
            for i in range(len(ax)):
                ax[i].set_xticks(range(len(ac2_labels)))
                ax[i].set_xticklabels(ac2_labels, rotation=45, ha='right')
            st.pyplot(fig)
    else:
        st.warning('First visit Model Building page')
    
def prediction():
    st.warning('''Please note that all checkboxes and buttons across sub-menus are interlinked. 
                    If any of them are not activated, the results may not appear as expected. We recommend ensuring that all checkboxes and buttons are properly activated before proceeding.''')
    if 'model_building' not in st.session_state:
        st.session_state.model_building=0
    model_building=st.session_state.model_building
    if 'c1_value' not in st.session_state:
        st.session_state.c1_value=0
    c1=st.session_state.c1_value
    if 'c2_value' not in st.session_state:
        st.session_state.c2_value=0
    c2=st.session_state.c2_value
    if 'c3_value' not in st.session_state:
        st.session_state.c3_value=0
    c3=st.session_state.c3_value
    if 'c4_value' not in st.session_state:
        st.session_state.c4_value=0
    c4=st.session_state.c4_value
    if 'b1' not in st.session_state:
        st.session_state.b1=0
    b1=st.session_state.b1
    if 'b2' not in st.session_state:
        st.session_state.b2=0
    b2=st.session_state.b2

    if model_building==1:
        pass
        if 'lr' not in st.session_state:
            st.session_state.lr=0
        if 'dt' not in st.session_state:
            st.session_state.dt=0
        if 'knn' not in st.session_state:
            st.session_state.knn=0
        if 'gnb' not in st.session_state:
            st.session_state.gnb=0
        if 'svm' not in st.session_state:
            st.session_state.svm=0
        if 'rf' not in st.session_state:
            st.session_state.rf=0
        if 'gb' not in st.session_state:
            st.session_state.gb=0
        if 'high' not in st.session_state:
            st.session_state.high=0
        if 'lr1' not in st.session_state:
            st.session_state.lr1=0
        if 'dt1' not in st.session_state:
            st.session_state.dt1=0
        if 'knn1' not in st.session_state:
            st.session_state.knn1=0
        if 'gnb1' not in st.session_state:
            st.session_state.gnb1=0
        if 'svm1' not in st.session_state:
            st.session_state.svm1=0
        if 'rf1' not in st.session_state:
            st.session_state.rf1=0
        if 'gb1' not in st.session_state:
            st.session_state.gb1=0
        if 'high1' not in st.session_state:
            st.session_state.high1=0
        lr=st.session_state.lr
        dt=st.session_state.dt
        knn=st.session_state.knn
        gnb=st.session_state.gnb
        svm=st.session_state.svm
        rf=st.session_state.rf
        gb=st.session_state.gb
        high=st.session_state.high
        lr1=st.session_state.lr1
        dt1=st.session_state.dt1
        knn1=st.session_state.knn1
        gnb1=st.session_state.gnb1
        svm1=st.session_state.svm1
        rf1=st.session_state.rf1
        gb1=st.session_state.gb1
        high1=st.session_state.high1
        column_desc = {
                        'Gender': 'Gender of the applicant',
                        'Married': 'Marital status of the applicant',
                        'Education': 'Education level of the applicant',
                        'Self_Employed': 'Self-employment status of the applicant',
                        'Property_Area': 'Location of the property',
                        'ApplicantIncome': 'Income of the applicant in dollars',
                        'Dependents': 'Number of dependents of the applicant',
                        'CoapplicantIncome': 'Income of the co-applicant in dollars (if any)',
                        'LoanAmount': 'Loan amount in dollars',
                        'Loan_Amount_Term': 'Term of the loan in months',
                        'Credit_History': 'Credit history of the applicant: 1 denotes a good credit history, 0 denotes a poor credit history'
                    }

        # Display the column descriptions
        st.write('Please provide the following information:')
        for col in column_desc:
            st.write(f'{col}: {column_desc[col]}')
        if st.checkbox('Lets start the prediction'):
            if (c1==1 and b2==1) and (c3==1 and b1==1):

                # Create a dictionary to encode categorical variables
                encoder_dict = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 
                                                    'Graduate': 1, 'Not Graduate': 0, 'Urban': 2, 
                                                    'Semiurban': 1, 'Rural': 0}

                                # Create a list of column names in the correct order
                column_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                                                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                                    'Loan_Amount_Term', 'Credit_History', 'Property_Area']

                                # Create a function to transform the user input
                def transform_input(user_input):
                                        # Convert the user input into a DataFrame
                    user_input_df = pd.DataFrame(user_input, index=[0])

                                        # Replace categorical values with numerical codes using the encoder dictionary
                    user_input_encoded = user_input_df.replace(encoder_dict)

                                        # Reorder the columns to match the original dataset
                    user_input_ordered = user_input_encoded[column_order]

                                        # Standardize the numerical columns
                    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
                    scaler = StandardScaler()
                    user_input_ordered[num_cols] = scaler.fit_transform(user_input_ordered[num_cols])

                    return user_input_ordered

                                    # Create a Streamlit app
                        

                                    # Collect user inputs
                st.subheader('Enter Applicant Information')
                gender = st.radio('Gender', ['Male', 'Female'])
                married = st.radio('Marital Status', ['Yes', 'No'])
                dependents = st.slider('Number of Dependents', 0, 3, 0)
                education = st.radio('Education', ['Graduate', 'Not Graduate'])
                employed = st.radio('Employment', ['Yes', 'No'])
                income = st.number_input('Applicant Income', min_value=0)
                co_income = st.number_input('Co-Applicant Income', min_value=0)
                loan_amount = st.number_input('Loan Amount', min_value=0)
                term = st.number_input('Loan Amount Term (in months)', min_value=0)
                credit_history = st.slider('Credit History (0 = No, 1 = Yes)', 0, 1, 0)
                property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

                                    # Store user inputs as a dictionary
                user_input = {'Gender': gender, 'Married': married, 'Dependents': dependents, 'Education': education,
                                                'Self_Employed': employed, 'ApplicantIncome': income, 'CoapplicantIncome': co_income,
                                                'LoanAmount': loan_amount, 'Loan_Amount_Term': term, 'Credit_History': credit_history,
                                                'Property_Area': property_area}

                                    # Transform user input into a format that can be used by the model
                user_input_transformed = transform_input(user_input)


                    
                models=['',
                                                        'LogisticRegression',
                                                        'DecisionTreeClassifier',
                                                        'KNeighborsClassifier',
                                                        'GaussianNB',
                                                        'SVC',
                                                        'RandomForestClassifier',
                                                        
                                                        'GradientBoostingClassifier'
                                                    ]
                d={'':'',
                                                        'Logistic Regression':'lr',
                                                        'Decision Tree':'dt',
                                                        'K-Nearest Neighbors':'knn',
                                                        'Gaussian Navie Bayes':'gnb',
                                                        'Support Vector Machine':'svm',
                                                        'Random Forest':'rf',
                                                        
                                                        'GradientBoostingClassifier':'gb'}
                #selection=st.selectbox('Choose the following model for predicting result using normal models',models)
                result=d[high]
                    
                if st.checkbox('Prediction is done by taking the normal algorithm which has high accuracy score'):
                    if result=='':
                        pass
                    elif result=='lr':
                        if st.button('Predict'):
                                            
                            prediction=lr.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result=='dt':
                        if st.button('Result'):
                            prediction=dt.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result=='knn':
                        if st.button('Click to see the result'):
                            prediction=knn.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result=='gnb':
                        if st.button('Press to see the result'):
                            prediction=gnb.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result=='svm':
                        if st.button('Hit karo bhai!'):
                            prediction=svm.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result=='rf':
                        if st.button('Answer'):
                            prediction=rf.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result=='gb':
                        if st.button('Forecast'):
                            prediction=gb.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    else:
                        pass
                            
                            
                
                

                d1={'':'',
                                                        'Logistic Regression':'lr1',
                                                        'Decision Tree':'dt1',
                                                        'K-Nearest Neighbors':'knn1',
                                                        'Gaussian Navie Bayes':'gnb1',
                                                        'Support Vector Machine':'svm1',
                                                        'Random Forest':'rf1',
                                                        
                                                        'GradientBoostingClassifier':'gb1'}
                    #selection1=st.selectbox('Choose the following model for predicting result using parameter models',models)
                result1=d1[high1]
                if st.checkbox('Prediction is done by taking the parameterised algorithm which has high accuracy score'):
                    if result1=='':
                        pass
                    elif result1=='lr1':
                        if st.button('Predict1'):
                                        
                            prediction=lr1.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result1=='dt1':
                        if st.button('Result1'):
                            prediction=dt1.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result1=='knn1':
                        if st.button('Click to see the result-1'):
                            prediction=knn1.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result1=='gnb1':
                        if st.button('Press to see the result-1'):
                            prediction=gnb1.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result1=='svm1':
                        if st.button('Hit karo bhai!-1'):
                            prediction=svm1.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result1=='rf1':
                        if st.button('Answer-1'):
                            prediction=rf1.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                    elif result1=='gb1':
                        if st.button('Forecast-1'):
                            prediction=gb1.predict(user_input_transformed)
                            if prediction[0]=='Y':
                                st.write('Congratulations! Your eligible for a loan')
                                st.image('approved.jpg',width=200)
                                st.audio('eligible.mp3', format='audio/mp3')
                                st.balloons()
                            else:
                                st.write('Sorry, your loan application has been rejected.')
                                st.image('rejected.jpg',width=200)
                                st.audio('rejection.mp3', format='audio/mp3')
                        
                    else:
                        pass
            else:
                st.warning('You missed to check some checkboxes in the previous pages!')
        
        
    else:
        st.warning('first visit the model building page')    
        

        
    


def actpred():
    if 'model_building' not in st.session_state:
        st.session_state.model_building=0
    model_building=st.session_state.model_building
    if model_building==1:
        pass
        if 'lrp' not in st.session_state:
            st.session_state.lrp=0
        if 'dtp' not in st.session_state:
            st.session_state.dtp=0
        if 'knnp' not in st.session_state:
            st.session_state.knnp=0
        if 'gnbp' not in st.session_state:
            st.session_state.gnbp=0
        if 'svmp' not in st.session_state:
            st.session_state.svmp=0
        if 'rfp' not in st.session_state:
            st.session_state.rfp=0
        if 'gbp' not in st.session_state:
            st.session_state.gbp=0
        if 'ytest' not in st.session_state:
            st.session_state=0
        if 'lrp1' not in st.session_state:
            st.session_state.lrp1=0
        if 'dtp1' not in st.session_state:
            st.session_state.dtp1=0
        if 'knnp1' not in st.session_state:
            st.session_state.knnp1=0
        if 'gnbp1' not in st.session_state:
            st.session_state.gnbp1=0
        if 'svmp1' not in st.session_state:
            st.session_state.svmp1=0
        if 'rfp1' not in st.session_state:
            st.session_state.rfp1=0
        if 'gbp1' not in st.session_state:
            st.session_state.gbp1=0
        if 'ytest' not in st.session_state:
            st.session_state=0
        ytest=st.session_state.ytest
        lrp=st.session_state.lrp
        dtp=st.session_state.dtp
        knnp=st.session_state.knnp
        gnbp=st.session_state.gnbp
        svmp=st.session_state.svmp
        rfp=st.session_state.rfp
        gbp=st.session_state.gbp
        lrp1=st.session_state.lrp1
        dtp1=st.session_state.dtp1
        knnp1=st.session_state.knnp1
        gnbp1=st.session_state.gnbp1
        svmp1=st.session_state.svmp1
        rfp1=st.session_state.rfp1
        gbp1=st.session_state.gbp1
        col1,col2=st.columns(2)
        
        models=['',
                                                'LogisticRegression',
                                                'DecisionTreeClassifier',
                                                'KNeighborsClassifier',
                                                'GaussianNB',
                                                'SVC',
                                               'RandomForestClassifier',
                                                
                                                'GradientBoostingClassifier'
                                            ]
        selection=st.selectbox('Choose the following algorithm',models)
        if selection=='':
            pass
        elif selection=='LogisticRegression':

            if st.button('Visualize'):
                ytest=pd.Series(ytest)
                lrp=pd.Series(lrp)
                lrp=lrp.replace({'Y':1,'N':0})
                lrp1=pd.Series(lrp1)
                lrp1=lrp1.replace({'Y':1,'N':0})
                ytest=ytest.replace({'Y':1,'N':0})

                fpr, tpr, thresholds = roc_curve(ytest, lrp)
                roc_auc = auc(fpr, tpr)
                fpr1, tpr1, thresholds = roc_curve(ytest, lrp1)
                roc_auc1 = auc(fpr1, tpr1)

                fig, ax = plt.subplots(2,1,figsize=(16, 16),sharex=True, gridspec_kw={'hspace': 0.5})
                ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[0].plot([0, 1], [0, 1],  color='red', lw=2, linestyle='--')
                ax[0].set_xlim([0.0, 1.0])
                ax[0].set_ylim([0.0, 1.05])
                ax[0].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[0].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[0].tick_params(axis='x', colors='white',labelsize=20)
                ax[0].tick_params(axis='y', colors='white',labelsize=20)
                ax[0].set_title('Receiver operating characteristic example of normal model', color='white',fontsize=30)
                ax[0].legend(loc="lower right",fontsize=20)

                # Set the background color of the plot to black
                fig.patch.set_facecolor('black')
                ax[0].set_facecolor('black')

                ax[1].plot(fpr1, tpr1,  color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                ax[1].set_xlim([0.0, 1.0])
                ax[1].set_ylim([0.0, 1.05])
                ax[1].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[1].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[1].tick_params(axis='x', colors='white',labelsize=20)
                ax[1].tick_params(axis='y', colors='white',labelsize=20)
                ax[1].set_title('Receiver operating characteristic example of parameterized model', color='white',fontsize=30)
                ax[1].legend(loc="lower right",fontsize=20)
                ax[1].set_facecolor('black')

                # Show plot in Streamlit
                st.pyplot(fig)


        elif selection=='DecisionTreeClassifier':
            if st.checkbox('Visualize'):
                ytest=pd.Series(ytest)
                dtp=pd.Series(dtp)
                dtp=dtp.replace({'Y':1,'N':0})
                dtp1=pd.Series(dtp1)
                dtp1=dtp1.replace({'Y':1,'N':0})
                ytest=ytest.replace({'Y':1,'N':0})

                fpr, tpr, thresholds = roc_curve(ytest, dtp)
                roc_auc = auc(fpr, tpr)
                fpr1, tpr1, thresholds = roc_curve(ytest, dtp1)
                roc_auc1 = auc(fpr1, tpr1)

                fig, ax = plt.subplots(2,1,figsize=(16, 16),sharex=True, gridspec_kw={'hspace': 0.5})
                ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[0].plot([0, 1], [0, 1],  color='red', lw=2, linestyle='--')
                ax[0].set_xlim([0.0, 1.0])
                ax[0].set_ylim([0.0, 1.05])
                ax[0].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[0].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[0].tick_params(axis='x', colors='white',labelsize=20)
                ax[0].tick_params(axis='y', colors='white',labelsize=20)
                ax[0].set_title('Receiver operating characteristic example of normal model', color='white',fontsize=30)
                ax[0].legend(loc="lower right",fontsize=20)

                # Set the background color of the plot to black
                fig.patch.set_facecolor('black')
                ax[0].set_facecolor('black')

                ax[1].plot(fpr1, tpr1,  color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                ax[1].set_xlim([0.0, 1.0])
                ax[1].set_ylim([0.0, 1.05])
                ax[1].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[1].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[1].tick_params(axis='x', colors='white',labelsize=20)
                ax[1].tick_params(axis='y', colors='white',labelsize=20)
                ax[1].set_title('Receiver operating characteristic example of parameterized model', color='white',fontsize=30)
                ax[1].legend(loc="lower right",fontsize=20)
                ax[1].set_facecolor('black')

                # Show plot in Streamlit
                st.pyplot(fig)



                   
                
        elif selection=='KNeighborsClassifier':
            if st.checkbox('Visualize'):
                ytest=pd.Series(ytest)
                knnp=pd.Series(knnp)
                knnp=knnp.replace({'Y':1,'N':0})
                knnp1=pd.Series(knnp1)
                knnp1=knnp1.replace({'Y':1,'N':0})
                ytest=ytest.replace({'Y':1,'N':0})

                fpr, tpr, thresholds = roc_curve(ytest, knnp)
                roc_auc = auc(fpr, tpr)
                fpr1, tpr1, thresholds = roc_curve(ytest, knnp1)
                roc_auc1 = auc(fpr1, tpr1)

                fig, ax = plt.subplots(2,1,figsize=(16, 16),sharex=True, gridspec_kw={'hspace': 0.5})
                ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[0].plot([0, 1], [0, 1],  color='red', lw=2, linestyle='--')
                ax[0].set_xlim([0.0, 1.0])
                ax[0].set_ylim([0.0, 1.05])
                ax[0].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[0].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[0].tick_params(axis='x', colors='white',labelsize=20)
                ax[0].tick_params(axis='y', colors='white',labelsize=20)
                ax[0].set_title('Receiver operating characteristic example of normal model', color='white',fontsize=30)
                ax[0].legend(loc="lower right",fontsize=20)

                # Set the background color of the plot to black
                fig.patch.set_facecolor('black')
                ax[0].set_facecolor('black')

                ax[1].plot(fpr1, tpr1,  color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                ax[1].set_xlim([0.0, 1.0])
                ax[1].set_ylim([0.0, 1.05])
                ax[1].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[1].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[1].tick_params(axis='x', colors='white',labelsize=20)
                ax[1].tick_params(axis='y', colors='white',labelsize=20)
                ax[1].set_title('Receiver operating characteristic example of parameterized model', color='white',fontsize=30)
                ax[1].legend(loc="lower right",fontsize=20)
                ax[1].set_facecolor('black')

                # Show plot in Streamlit
                st.pyplot(fig)


                  
        elif selection=='GaussianNB':
            if st.button('Visualize'):
                ytest=pd.Series(ytest)
                gnbp=pd.Series(gnbp)
                gnbp=gnbp.replace({'Y':1,'N':0})
                gnbp1=pd.Series(gnbp1)
                gnbp1=gnbp1.replace({'Y':1,'N':0})
                ytest=ytest.replace({'Y':1,'N':0})

                fpr, tpr, thresholds = roc_curve(ytest, gnbp)
                roc_auc = auc(fpr, tpr)
                fpr1, tpr1, thresholds = roc_curve(ytest, gnbp1)
                roc_auc1 = auc(fpr1, tpr1)

                fig, ax = plt.subplots(2,1,figsize=(16, 16),sharex=True, gridspec_kw={'hspace': 0.5})
                ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[0].plot([0, 1], [0, 1],  color='red', lw=2, linestyle='--')
                ax[0].set_xlim([0.0, 1.0])
                ax[0].set_ylim([0.0, 1.05])
                ax[0].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[0].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[0].tick_params(axis='x', colors='white',labelsize=20)
                ax[0].tick_params(axis='y', colors='white',labelsize=20)
                ax[0].set_title('Receiver operating characteristic example of normal model', color='white',fontsize=30)
                ax[0].legend(loc="lower right",fontsize=20)

                # Set the background color of the plot to black
                fig.patch.set_facecolor('black')
                ax[0].set_facecolor('black')

                ax[1].plot(fpr1, tpr1,  color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                ax[1].set_xlim([0.0, 1.0])
                ax[1].set_ylim([0.0, 1.05])
                ax[1].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[1].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[1].tick_params(axis='x', colors='white',labelsize=20)
                ax[1].tick_params(axis='y', colors='white',labelsize=20)
                ax[1].set_title('Receiver operating characteristic example of parameterized model', color='white',fontsize=30)
                ax[1].legend(loc="lower right",fontsize=20)
                ax[1].set_facecolor('black')

                # Show plot in Streamlit
                st.pyplot(fig)
        elif selection=='SVC':
            if st.button('Visualize'):
                ytest=pd.Series(ytest)
                svmp=pd.Series(svmp)
                svmp=svmp.replace({'Y':1,'N':0})
                svmp1=pd.Series(svmp1)
                svmp1=svmp1.replace({'Y':1,'N':0})
                ytest=ytest.replace({'Y':1,'N':0})

                fpr, tpr, thresholds = roc_curve(ytest, svmp)
                roc_auc = auc(fpr, tpr)
                fpr1, tpr1, thresholds = roc_curve(ytest, svmp1)
                roc_auc1 = auc(fpr1, tpr1)

                fig, ax = plt.subplots(2,1,figsize=(16, 16),sharex=True, gridspec_kw={'hspace': 0.5})
                ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[0].plot([0, 1], [0, 1],  color='red', lw=2, linestyle='--')
                ax[0].set_xlim([0.0, 1.0])
                ax[0].set_ylim([0.0, 1.05])
                ax[0].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[0].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[0].tick_params(axis='x', colors='white',labelsize=20)
                ax[0].tick_params(axis='y', colors='white',labelsize=20)
                ax[0].set_title('Receiver operating characteristic example of normal model', color='white',fontsize=30)
                ax[0].legend(loc="lower right",fontsize=20)

                # Set the background color of the plot to black
                fig.patch.set_facecolor('black')
                ax[0].set_facecolor('black')

                ax[1].plot(fpr1, tpr1,  color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                ax[1].set_xlim([0.0, 1.0])
                ax[1].set_ylim([0.0, 1.05])
                ax[1].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[1].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[1].tick_params(axis='x', colors='white',labelsize=20)
                ax[1].tick_params(axis='y', colors='white',labelsize=20)
                ax[1].set_title('Receiver operating characteristic example of parameterized model', color='white',fontsize=30)
                ax[1].legend(loc="lower right",fontsize=20)
                ax[1].set_facecolor('black')

                # Show plot in Streamlit
                st.pyplot(fig)
        elif selection=='RandomForestClassifier':
            if st.button('Visualize'):
                ytest=pd.Series(ytest)
                rfp=pd.Series(rfp)
                rfp=rfp.replace({'Y':1,'N':0})
                rfp1=pd.Series(rfp1)
                rfp1=rfp1.replace({'Y':1,'N':0})
                ytest=ytest.replace({'Y':1,'N':0})

                fpr, tpr, thresholds = roc_curve(ytest, rfp)
                roc_auc = auc(fpr, tpr)
                fpr1, tpr1, thresholds = roc_curve(ytest, rfp1)
                roc_auc1 = auc(fpr1, tpr1)

                fig, ax = plt.subplots(2,1,figsize=(16, 16),sharex=True, gridspec_kw={'hspace': 0.5})
                ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[0].plot([0, 1], [0, 1],  color='red', lw=2, linestyle='--')
                ax[0].set_xlim([0.0, 1.0])
                ax[0].set_ylim([0.0, 1.05])
                ax[0].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[0].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[0].tick_params(axis='x', colors='white',labelsize=20)
                ax[0].tick_params(axis='y', colors='white',labelsize=20)
                ax[0].set_title('Receiver operating characteristic example of normal model', color='white',fontsize=30)
                ax[0].legend(loc="lower right",fontsize=20)

                # Set the background color of the plot to black
                fig.patch.set_facecolor('black')
                ax[0].set_facecolor('black')

                ax[1].plot(fpr1, tpr1,  color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                ax[1].set_xlim([0.0, 1.0])
                ax[1].set_ylim([0.0, 1.05])
                ax[1].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[1].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[1].tick_params(axis='x', colors='white',labelsize=20)
                ax[1].tick_params(axis='y', colors='white',labelsize=20)
                ax[1].set_title('Receiver operating characteristic example of parameterized model', color='white',fontsize=30)
                ax[1].legend(loc="lower right",fontsize=20)
                ax[1].set_facecolor('black')

                # Show plot in Streamlit
                st.pyplot(fig)
        elif selection=='GradientBoostingClassifier':
            if st.button('Visualize'):
                ytest=pd.Series(ytest)
                gbp=pd.Series(gbp)
                gbp=gbp.replace({'Y':1,'N':0})
                gbp1=pd.Series(gbp1)
                gbp1=gbp1.replace({'Y':1,'N':0})
                ytest=ytest.replace({'Y':1,'N':0})

                fpr, tpr, thresholds = roc_curve(ytest, gbp)
                roc_auc = auc(fpr, tpr)
                fpr1, tpr1, thresholds = roc_curve(ytest, gbp1)
                roc_auc1 = auc(fpr1, tpr1)

                fig, ax = plt.subplots(2,1,figsize=(16, 16),sharex=True, gridspec_kw={'hspace': 0.5})
                ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[0].plot([0, 1], [0, 1],  color='red', lw=2, linestyle='--')
                ax[0].set_xlim([0.0, 1.0])
                ax[0].set_ylim([0.0, 1.05])
                ax[0].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[0].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[0].tick_params(axis='x', colors='white',labelsize=20)
                ax[0].tick_params(axis='y', colors='white',labelsize=20)
                ax[0].set_title('Receiver operating characteristic example of normal model', color='white',fontsize=30)
                ax[0].legend(loc="lower right",fontsize=20)

                # Set the background color of the plot to black
                fig.patch.set_facecolor('black')
                ax[0].set_facecolor('black')

                ax[1].plot(fpr1, tpr1,  color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                ax[1].set_xlim([0.0, 1.0])
                ax[1].set_ylim([0.0, 1.05])
                ax[1].set_xlabel('False Positive Rate', color='white',fontsize=20)
                ax[1].set_ylabel('True Positive Rate', color='white',fontsize=20)
                ax[1].tick_params(axis='x', colors='white',labelsize=20)
                ax[1].tick_params(axis='y', colors='white',labelsize=20)
                ax[1].set_title('Receiver operating characteristic example of parameterized model', color='white',fontsize=30)
                ax[1].legend(loc="lower right",fontsize=20)
                ax[1].set_facecolor('black')

                # Show plot in Streamlit
                st.pyplot(fig)
                                


        

        

        
       

              
    
    else:
        st.warning('Visit model building page initially!')
            



    
    


                
        


        
    


    
    
    
def main():
    menu=['Home','Login','SignUp','About']
    sub_menu=['EDA','Model Building','Deploying models without parameters','Deploying models with parameters','Non-Parameters vs Parameters','Prediction','Model Performance']
    page=st.sidebar.selectbox('Select a page',menu)
    
    if page=='Home':
        home()
    elif page=='Login':
        login(sub_menu)
    elif page=='SignUp':
        sign_up()
    elif page=='About':
        about_page()

if __name__ == '__main__':
    main()
