import streamlit as st
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import base64

pa = 'View our training dataset'
df1 = pd.read_csv('loanone.csv')
df2=pd.read_csv('loanplus.csv')
one = pickle.load(open('one.pkl', 'rb'))
mod=pickle.load(open('model.pkl','rb'))


def predict(df3,df4):
    df3['Loan_Status']=mod.predict(df3)
    df3['Loan_ID']=df4['Loan_ID']
    E=df3[['Loan_ID','Loan_Status']]
    E.loc[E.Loan_Status == 0, 'Loan_Status'] = 'N'
    E.loc[E.Loan_Status == 1, 'Loan_Status'] = 'Y'
    return E


def main():
    st.title("HOME LOAN APPROVAL PREDICTOR")
    st.subheader('Hello!! This is a loan predictor web app, which you can use to predict your loan approval chances. Have Fun!!')
    st.subheader('Remember to upload csv files that have columns as per the dataset displayed below[LoanAmount in 1000s]')
    st.write(df1.head())
    st.subheader('An overview of numerical data')
    st.write(df1.describe())
    st.subheader('Correlation values between different features are:')
    sns.heatmap(df1.corr(),annot=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def indi():
    st.title("INDIVIDUAL LOAN APPROVAL PREDICTOR")
    g = st.subheader('Enter the following details')
    f = 'Male'
    st.markdown("""
        <style>
        body {
            color: #AA1B51;
            background-color: #111;
            etc. 
        }
        </style>
            """, unsafe_allow_html=True)
    f = st.selectbox('Select your gender', tuple(['Male', 'Female']), key='1')
    h = 'No'
    h = st.selectbox('Are you married?', tuple(['Yes', 'No']), key='2')
    a = 0
    a = st.selectbox('Choose number of dependents (max 3)', tuple([0, 1, 2, 3]), key='3')
    b = 'Graduate'
    b = st.selectbox('Education level', tuple(['Graduate', 'Not Graduate']), key='4')
    c = 'No'
    c = st.selectbox('Are you self-employed?', tuple(['Yes', 'No']), key='5')
    d = st.number_input('Applicant Income', 0, key='6')
    e = st.number_input('CoApplicant Income', 0.0, key='7')
    i = st.number_input('Loan Amount', 0.0, key='8')
    j = st.number_input('Loan Amount Term', 1.0, key='9')
    k = 0.0
    k = st.selectbox('Credit_History', tuple([0.0, 1.0]), key='10')
    o = 'Semiurban'
    o = st.selectbox('Property Area', tuple(['Semiurban', 'Rural', 'Urban']), key='11')
    li= {'Gender':f,'Married':h,'Dependents':a,'Education':b,'Self_Employed':c,'Credit_History':k,'Property_Area':o,'Loan_Status':0,'Total':d+e,'Debt/Income':(i*1000/(d+e+1)),'EMI':(i*(0.09)*((1.09)**j))/((1.09)**(j-1))}
    gh = pd.DataFrame(li,index=[0])
    t = st.button('Predict')
    if t:
        y=pd.DataFrame()
        pi=['Gender','Married','Education','Self_Employed','Property_Area']
        J=pd.DataFrame(one.transform(gh[pi]))
        J.index=gh.index
        S=gh.drop(pi,axis=1)
        FD=pd.concat([J,S],axis=1)
        FD.drop(['Loan_Status'],axis=1,inplace=True)
        y['Loan_ID']=list(['000'])
        st.write('The prediction based on given input is:')
        st.write(predict(FD,y))


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'


def bank():
    st.markdown("""
        <style>
        body {
            color: #066C9A;
            background-color: #111;
            etc. 
        }
        </style>
            """, unsafe_allow_html=True)
    st.header('Loan prediction for a large dataset in csv')
    st.subheader('Hope you enjoy it!')
    v = st.file_uploader("Upload File", type=['csv'])
    if v:
        df=pd.read_csv(v)
        for col in df.select_dtypes(exclude='object').columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
        df.loc[df['Dependents'] == '3+', 'Dependents'] = '3'
        T = df.drop("Loan_ID", axis=1)
        T['Loan_Status']=0
        T['Total'] = T['ApplicantIncome']+T['CoapplicantIncome']
        T['Debt/Income'] = (T['LoanAmount']*1000)/T['Total']
        T['EMI'] = ((T['LoanAmount']*(0.09)*((1.09) ** T['Loan_Amount_Term']))/((1.09) ** (T['Loan_Amount_Term']-1)))
        T.drop(['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)
        T['Dependents'] = T['Dependents'].astype(int)
        lis = []
        for col in T.columns:
            if T[col].dtype == 'object':
                lis.append(col)
        K = pd.DataFrame(one.transform(T[lis]))
        K.index = T.index
        L = T.drop(lis, axis=1)
        M = pd.concat([K, L], axis=1)
    bu=st.button('Predict')
    if bu:
       M.drop(['Loan_Status'],axis=1,inplace=True)
       cs1=predict(M,df)
       st.write(cs1)
       st.markdown(get_table_download_link(cs1), unsafe_allow_html=True)


pages = {
    'View our training dataset': main,
    'Predict for a large dataset': bank,
    'Predict for an individual': indi
}


def intro():
    st.sidebar.title('You are here to:')
    global pa
    pa = st.sidebar.selectbox('Select your page', tuple(pages.keys()))
    pages[pa]()


if __name__ == '__main__':
    st.markdown("""
    <style>
    body {
        color: #118B51;
        background-color: #250620;
        etc. 
    }
    </style>
        """, unsafe_allow_html=True)
    intro()
