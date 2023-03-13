import pandas as pd
import streamlit as st
import sweetviz as sv
import os
from sklearn.metrics import roc_auc_score,roc_curve
import joblib
import hashlib
import re
import plotly.figure_factory as ff
from sklearn import set_config
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from managed_db import *
import time

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

menu=['Home','Login','SignUp','About']
sub_menu=['EDA','Model Building']
choice=st.sidebar.selectbox('Menu',menu)

if choice=='Home':
    st.title('Loan Prediction app')
    
elif choice=='Login':
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
            
            activity=st.sidebar.selectbox('Activity',sub_menu)
            
            if activity=='EDA':
                sub1=['Sweetviz']
                st.subheader("Perform Exploratory data Analysis with sweetviz Library")
                data_file= st.file_uploader("Upload a csv file", type=["csv"])
                status=st.selectbox('Which type of EDA is needed?',sub1)
                
                if status=='Sweetviz':
                    if st.button('Analyze'):
                        if data_file is not None:
                            @st.cache
                            def load_csv():
                                csv=pd.read_csv(data_file)
                                return csv
                            
                            df=load_csv()
                            st.header('*User Input DataFrame*')
                            st.write(df)
                            st.write('---')
                            st.subheader('*Exploratory Data Analysis Report Using Sweetviz*')
                            report = sv.analyze(df)
                            report_html=report.show_html()
                            st.components.v1.html(report_html, height=700, scrolling=True)
                            
                        else:
                            st.warning('File not found')
            elif activity=='Model Building':
                st.warning('Please use same csv for model building')
                new_data_file= st.file_uploader("Upload a csv file", type=["csv"])
                
                if new_data_file is not None:
                        
                    def load_csv():
                        csv=pd.read_csv(new_data_file)
                        return csv
                                
                    df=load_csv()
                    guna=df.info()
                    st.text(guna)
                        

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
                    columns_list=list(df.columns)
                    columns_to_do_delete=st.multiselect('Select columns to delete',columns_list)
                    df1=df.copy()
                    if st.checkbox('Delete the selected columns'):
                        
                        df1.drop(columns=columns_to_do_delete,inplace=True)
                        st.write('New DataFrame')
                        st.write(df1)
                     
                        st.write('old df shape:{}'.format(df.shape))
                        st.write('new df shape:{}'.format(df1.shape))
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

                    if st.checkbox('Seperating the independent and dependent features'):
                        x=df1.iloc[:,0:11]
                        y=df1['Loan_Status']
                        dependent_features=x
                        independent_features=y
                        st.write(dependent_features)
                        st.write(independent_features)
                    
                           
                
                    if st.checkbox('Splitting data into training and testing'):
                        
                        st.info('We are using 20 percent of data for testing')
                        
                        
                        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=50)
                        st.write('Shape of xtrain {}'.format(xtrain.shape))
                        st.write('Shape of xtest {}'.format(xtest.shape))
                        st.write('Shape of ytrain {}'.format(ytrain.shape))
                        st.write('Shape of ytest {}'.format(ytest.shape))
                    if st.checkbox('PipeLine Building for treating missing values and  model deploying'):
                        numerical=['mean','median']
                        categorical=['most_frequent','constant']
                        ns=st.radio('Filling for numerical columns',numerical)
                        cs=st.radio('Filling for numerical columns',categorical)
                    set_config(display='diagram')
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
                            ('Encoding',OneHotEncoder()),
                            ]
                        )
                        st.success('Successfully categorical pipleline is built!')
                        
                    
                 
                   
                    
                    if st.checkbox('Combing both transformers using column transformers'):
                        preprocessing=ColumnTransformer(
                          [
                            ('categorical columns',categorical_cols,[i for i in catcols]),
                            ('numerical columns',numerical_cols,[i for i in numcols]),

                            ]


                        )
                        
                        st.success('Column tranformers are built')
                   
                    st.subheader('Time for deploying models without tuning:')
                    st.info('We are building models without any hyper parameters')
                    models_menu=['','Logistic Regression','Decision Tree','KNN','Gaussian NB','SVM','Random Forest','XGB','Gradient Boosting']
                    selection=st.selectbox('Choose the following models',models_menu)
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
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['Y']['precision']
                        rscore=cr['Y']['recall']
                        f1score=cr['Y']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
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
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['Y']['precision']
                        rscore=cr['Y']['recall']
                        f1score=cr['Y']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
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
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['Y']['precision']
                        rscore=cr['Y']['recall']
                        f1score=cr['Y']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
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
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['Y']['precision']
                        rscore=cr['Y']['recall']
                        f1score=cr['Y']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
                    elif selection=='SVM':
                        sv_machine=Pipeline(
                                steps=[
                                    ('preprocessing',preprocessing),
                                    ('classification model',SVC())
                                ])
                        sv_machine.fit(xtrain,ytrain)   
                        ypred=sv_machine.predict(xtest)
                        cr=classification_report(ytest,ypred,output_dict=True)
                        cr1=classification_report(ytest,ypred)
                        cm=confusion_matrix(ytest,ypred)
                        st.write('Classificaton Report :')
                        st.write(cr)
                        ac=cr['accuracy']*100
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['Y']['precision']
                        rscore=cr['Y']['recall']
                        f1score=cr['Y']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
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
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['Y']['precision']
                        rscore=cr['Y']['recall']
                        f1score=cr['Y']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
                    elif selection=='XGB':
                        xgb=Pipeline(
                                steps=[
                                    ('preprocessing',preprocessing),
                                    ('classification model',XGBClassifier())
                                ])
                        le=LabelEncoder()
                        ytrain1=le.fit_transform(ytrain)
                        ytest1=le.transform(ytest)
                        xgb.fit(xtrain,ytrain1)   
                        ypred=xgb.predict(xtest)
                        cr=classification_report(ytest1,ypred,output_dict=True)
                        cr1=classification_report(ytest1,ypred)
                        cm=confusion_matrix(ytest1,ypred)
                        st.write('Classificaton Report :')
                        st.write(cr)
                        ac=cr['accuracy']*100
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['1']['precision']
                        rscore=cr['1']['recall']
                        f1score=cr['1']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
                    elif selection=='Gradient Boosting':
                        gbc=Pipeline(
                                steps=[
                                    ('preprocessing',preprocessing),
                                    ('classification model',GradientBoostingClassifier())
                                ])
                        gbc.fit(xtrain,ytrain)   
                        ypred=gbc.predict(xtest)
                        cr=classification_report(ytest,ypred,output_dict=True)
                        cr1=classification_report(ytest,ypred)
                        cm=confusion_matrix(ytest,ypred)
                        st.write('Classificaton Report :')
                        st.write(cr)
                        ac=cr['accuracy']*100
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['Y']['precision']
                        rscore=cr['Y']['recall']
                        f1score=cr['Y']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
                    new_heading='Comparision of all the above algorithm results except xgboost'
                    if st.checkbox(new_heading):
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
                            pipe = Pipeline(steps=[('preprocessing', preprocessing),
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
                            st.write('Confusion matrix')
                            st.write(cr)
                            
                            st.write('-' * 60)
                    small_select=['Comparision between algorithms accuracy','Comparision between algorithms f1,recall and precision score']
                    small_result_set=st.selectbox('Choose anyone of the following:',small_select)
                    if small_result_set=='Comparision between algorithms accuracy':
                        ac_scores_before_tuning = [j for i, j in enumerate(ac.values())]
                        ac_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Random Forest','GradientBoostingClassifier']
                        sorted_indices = np.argsort(ac_scores_before_tuning)
                        ac_scores_before_tuning_sorted = [ac_scores_before_tuning[i] for i in sorted_indices]
                        ac_labels_sorted = [ac_labels[i] for i in sorted_indices]

                        pos = np.arange(len(ac_labels_sorted))
                        width = 0.25
                        fig, ax = plt.subplots(figsize=(20, 10))
                        sns.set_style('darkgrid')
                        sns.set_palette('dark')
                        ax.set_facecolor('black')
                        ax.spines['bottom'].set_color('white')
                        ax.spines['left'].set_color('white')
                        ax.tick_params(axis='x', colors='white')
                        ax.tick_params(axis='y', colors='white')

                        ax.yaxis.label.set_color('white')
                        ax.xaxis.label.set_color('white')
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_color('white')

                        rects1 = ax.bar(pos - width/2, ac_scores_before_tuning_sorted, width, label='Before Tuning',color='red')
                        ax.set_xticks(pos)
                        ax.set_xticklabels(ac_labels_sorted, rotation=45, ha='right',size=30)
                        ax.tick_params(axis='x',labelsize=30)
                        ax.tick_params(axis='y',labelsize=30)

                        ax.set_ylabel('Accuracy Score',size=40)
                        ax.set_ylim([0, 100])
                        ax.set_title('Comparison of all algorithms',size=30)
                        ax.legend(fontsize=20)
                        def autolabel(rects):
                            for rect in rects:
                                height = rect.get_height()
                                ax.annotate('{:.2f}'.format(height),
                                            xy=(rect.get_x() + rect.get_width() / 2, height),
                                            xytext=(0, 3),
                                            textcoords="offset points",
                                            ha='center', va='bottom',rotation=40,color='white',size=30)
                        autolabel(rects1)
                        st.pyplot(fig)



                        

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
                    st.subheader('Time for deploying models with tuning:')
                    st.info('We are not implementing hyper parameter tuning. Just adding some of the best parameters in order to increase the model accuracy')
                    new_models_menu=['',"Logistic Regression",'Decision Tree','KNN','Gaussian NB','SVM','Random Forest','XGB','Gradient Boosting']
                    selection1=st.selectbox('Choose the following models with tuning',new_models_menu)
                    if selection1=='Logistic Regression':
                         
                        lr=Pipeline(
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
                        lr.fit(xtrain,ytrain)
                        ypred=lr.predict(xtest)
                        cr=classification_report(ytest,ypred,output_dict=True)
                        cr1=classification_report(ytest,ypred)
                        cm=confusion_matrix(ytest,ypred)
                        st.write('Classificaton Report :')
                        st.write(cr)
                        ac=cr['accuracy']*100
                        pass
                        st.write('Accuracy is: {}'.format(ac))
                        
                        
                        pscore=cr['Y']['precision']
                        rscore=cr['Y']['recall']
                        f1score=cr['Y']['f1-score']
                        st.write('Precession score: {}'.format(pscore))
                        st.write('Recall score: {}'.format(rscore))
                        st.write('F1-Score: {}'.format(f1score))
                        labels = ['Class 0', 'Class 1']
                        colors = ['green','red']
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        sns.set(font_scale=1.4)
                        sns.heatmap(cm, annot=True, fmt='g', cmap=colors, xticklabels=labels, yticklabels=labels)

                        # Set the axis labels and title
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')

                        # Display the plot in Streamlit
                        st.pyplot()
                       
                      
                            



                        
                        
        else:
            st.warning('Incorrect Username/Password')
            
elif choice == "SignUp":
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
        pass
        st.info('Login to get started')
    
elif choice=='About':
    st.write('I am a developer')

