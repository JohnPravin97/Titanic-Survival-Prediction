import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, f1_score, roc_auc_score, roc_curve, plot_roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier
import joblib
import streamlit as st
import pickle
from PIL import Image


class make_column_transformerz(TransformerMixin):
    def __init__(self, estimator, col_list):
        self.estimator=estimator
        self.col_list=col_list
        self.make_= None
    def fit(self, X, y=0):
        self.make_= make_column_transformer((self.estimator, self.col_list), remainder='drop')
        self.make_.fit(X)
        return self
    def transform(self,X):
        dummy=self.make_.transform(X)
        #cols=list(X.columns).remove(str(self.col_list))
        transformed=pd.DataFrame(dummy, columns=self.col_list)
        X.drop(self.col_list, axis=1, inplace=True)
        X=pd.concat([X,transformed], axis=1, join='inner')
        return X

class columnseparater(TransformerMixin):
    def __init__(self, cols):
        self.cols=cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_new = X[self.cols]
        return X_new

class standardscaler(TransformerMixin):
    def __init__(self):
        self.ss=None
        self.mean_=None
        self.scale_=None
    def fit(self, X, y=None):
        self.ss=StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self
    def transform(self, X):
        x=self.ss.transform(X)
        numeric=pd.DataFrame(x, columns=X.columns)
        return numeric

class categorical_data(TransformerMixin):
    def __init__(self):
        self.value=None
        self.categories_=None
        self.columns=[]
    def fit(self, X, y=None):
        self.value=OneHotEncoder()
        self.value.fit(X)
        self.categories_=pd.Series(self.value.categories_)
        return self
    def transform(self, X):
        x=self.value.transform(X)
        for i in range(len(self.categories_)):
            self.columns+=list(self.categories_[i]) #columns remove pannuna work aaghudhu
        cate=pd.DataFrame(x.toarray())
        return cate
    
def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value

class DFFeatureUnion(TransformerMixin):
    # FeatureUnion but for pandas DataFrames
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for t in self.transformer_list:
            t.fit(X)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X) for t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion


class SessionState(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


    def get(**kwargs):
    
        # Hack to get the session object from Streamlit.
    
        ctx = ReportThread.get_report_ctx()
    
        this_session = None
    
        current_server = Server.get_current()
        if hasattr(current_server, '_session_infos'):
            # Streamlit < 0.56
            session_infos = Server.get_current()._session_infos.values()
        else:
            session_infos = Server.get_current()._session_info_by_id.values()
    
        for session_info in session_infos:
            s = session_info.session
            if (
                # Streamlit < 0.54.0
                (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
                or
                # Streamlit >= 0.54.0
                (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
                or
                # Streamlit >= 0.65.2
                (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr)
            ):
                this_session = s
    
        if this_session is None:
            raise RuntimeError(
                "Oh noes. Couldn't get your Streamlit Session object. "
                'Are you doing something fancy with threads?')
    
        # Got the session object! Now let's attach some state into it.
    
        if not hasattr(this_session, '_custom_session_state'):
            this_session._custom_session_state = SessionState(**kwargs)
    
        return this_session._custom_session_state
    
    
try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server
    
def prediction(X):
    output = []
    joblib_file_hard =  r"Voting_hard_classifier_model.pkl"
    joblib_file_soft =  r"Voting_soft_classifier_model.pkl"
    joblib_file_rf = r"random_forest_classifier_model.pkl"

    vc_hard = joblib.load(joblib_file_hard)
    vc_soft = joblib.load(joblib_file_soft)
    rf = joblib.load(joblib_file_rf)
    
    output.append(vc_hard.predict(X))
    output.append(vc_soft.predict(X))
    output.append(rf.predict(X))
    unique, counts = np.unique(np.asarray(output), return_counts=True)
    
    return output, unique, counts

    
def titanic_pre(df):
    df['Family_group'] = pd.cut(
        df.SibSp, bins=[-1, 2, 4, 10], labels=['Small', 'Medium', 'Large']).astype('object')
    df['Parent_Count'] = pd.cut(
        df.Parch, bins=[-1, 0.5, 1.5, 10], labels=['No_parent', 'single_parent', 'big_family']).astype('object')
    
    return df

#Pipeline Section
num = ['Age', 'Fare']
cat = ['Pclass', 'Name', 'Embarked', 'Family_group', 'Parent_Count']
pipeline = make_pipeline(
    make_pipeline(
        make_column_transformerz(SimpleImputer(
            strategy='most_frequent'), ['Embarked']),
        make_column_transformerz(SimpleImputer(
            strategy='most_frequent'), ['Fare']),
        make_column_transformerz(SimpleImputer(
            strategy='most_frequent'), ['Age']),
        make_column_transformerz(FunctionTransformer(np.log), ['Fare']),
    ),
    DFFeatureUnion([
        make_pipeline(
            columnseparater(num),
            standardscaler()
        ),
        make_pipeline(
            columnseparater(cat),
            categorical_data()
        )
    ])
)

filename = "train_dataset.pkl"
train_data = pickle.load(open(filename, 'rb'))
train_data = pipeline.fit_transform(train_data)

st.set_page_config(layout="wide")  
state = SessionState.get(flag=False)
  
if st.sidebar.button('Close Me'):
    st.markdown('Go away but do come back')
    
else:
    st.sidebar.markdown('''# Welcome to Titanic Prediction Tool\n
### Below are the creator details
#### Name : John Pravin A (<johnpravina@gmail.com>)\n
#### LinkedIn :  <https://www.linkedin.com/in/john-pravin-88a35014b/> 
#### GitHub : <https://github.com/JohnPravin97> 
### Special thanks to
#### Name : Aakash N (<gn.aakash@gmail.com>)''')
    
    st.sidebar.markdown('''### Feedback \n''')
    feedlist = ['Better', 'Normal', 'Worst']
    feedback = st.sidebar.radio('Please provide your feedback below', feedlist)
    if st.sidebar.button('Send Feedback'):
        st.sidebar.write('Thanks for your '+ '"'+str(feedback)+'"' +' feedback')
    st.sidebar.markdown('''### Disclaimer''') 
    st.sidebar.write('Input name and data are being stored for further improvements of the tool')
        
    # Main Coding
    st.markdown('''<div align="center"> <h1> <b> Welcome to Titanic Survival Prediction Tool </b> </h1> </div>''', unsafe_allow_html=True)
    
    img = st.beta_columns(3)
    img[1].image(Image.open('Titanic.jpg'), width=425, caption='Titanic')
    
    
    st.markdown('''<u>
    <h3> <b> INTRODUCTION: </b> </h3> </u>''', unsafe_allow_html=True)
    
    st.markdown('''Ever had a dream of travelling on the titanic? ever wondered if you would have survived the unfortunate incident if you got a ticket aboard the dreamy cruise for a journey in the year 1912? We got you covered:\n
This is a prediction system which can be used to predict the chance of a person surviving the titanic incident. The system asks the user input for data and predicts the chance of survival of the user in the titanic incident. This model is trained using the original titanic data set with basic ML techniques like regression, and to increase the accuracy of prediction it was later combined with advanced ML techniques such as the ensemble learning and integrated all of these into an end to end pipeline architecture.\n
The following are the data one has to provide : Pclass, Name, Sex, Age, Fare, SibSp, Parch, Embarked. Refer the below describtion for more detailed explanation on each parameters''')
    
    markdwn = st.beta_columns(3)
    markdw = st.beta_columns(2)
    
    markdwn[0].markdown('''<u>
    <h3> <b> VARIABLE DESCRIPTIONS: </b> </h3> </u>''', unsafe_allow_html=True)
    
    markdw[0].markdown('''
        \t **Pclass**   -> Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)\n
        **Name**     -> Name of the passenger\n
        **Sex**     -> Sex\n
        **Age**    -> Age\n
    ''')
    
    markdw[1].markdown('''
    **Sibsp**   -> Number of Siblings/Spouses Aboard\n
    **Parch**    -> Number of Parents/Children Aboard\n
    **Fare**     -> Passenger Fare (British pound)\n
    **Embarked** -> Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)\n
    ''')
    
    st.markdown('''<u>
    <h3> <b> USER GUIDES: </b> </h3> </u>''', unsafe_allow_html=True)
    st.markdown(''' Points to consider while working with this tool: \n
                1. User has to fill the Pclass and press 'Enter' button to get the remaining parameters.
                2. User has to fill his first name to proceed further with the prediction.
                3. User has to follow and provide the values between the range mentioned for Age and Fare.
                4. Upon filling all the required parameters, press 'Predict' button to proceed.
                5. After filling all the parameters, the users are provided with dataframe containing their inputs for reference.
                6. Sex parameter is automatically taken based on the Title given.
                7. Family Group and Parent Count are calculated based on the SibSp and Parch parameters. 
                8. Press 'Refresh' button to try with new data
                9. Press 'Close Me' button located on the top left corner to close the tool
                ''')
    st.markdown('''<u>
    <h3> <b> LETS GET STARTED: </b> </h3> </u>''', unsafe_allow_html=True)            
    initial_cols = st.beta_columns(3)
    
    initial_lis = {'Pclass': ['<select>',1,2,3]}
    Pclass = (initial_cols[1].selectbox('Pclass', initial_lis['Pclass']))
    
    if (Pclass=='<select>'):
        initial_cols[1].write('Please select the Pclass to Proceed')
        state.flag=False
        
    else:    
        if (initial_cols[1].button('Enter') or state.flag ==True): 
            Pclass = int(Pclass)
            
            if Pclass==3: 
                st.markdown('''<div align="center"> <h4> You have selected Pclass as 3 and provide information below details to predict your titanic survival fate<br>  </br> </h4> </div>''', unsafe_allow_html=True                        )
                name_cols = st.beta_columns(5)
                cols = st.beta_columns(5)
                lis = {
                       'Name':['Mr.', 'Miss.', 'Mrs.', 'Master.'],  
                       'SibSp':[0,1,2,3,4,5,8], 
                       'Parch':[0,1,2,3,4,5,6], 
                       'Embarked':['S', 'C', 'Q']
                       }
                Name = name_cols[1].selectbox('Title', lis['Name'])
                First_Name = name_cols[2].text_input('Please Enter Your First Name')
                Last_Name = name_cols[3].text_input('Please Enter Your last Name')
    
                #To select age range from the Name
                if Name == 'Master.':
                    min_age_value=0
                    max_age_value=12
                elif Name=='Mr.':
                    min_age_value=15
                    max_age_value=74
                elif Name=='Miss.':
                    min_age_value=0
                    max_age_value=45
                    
                elif Name=='Mrs.':
                    min_age_value=15
                    max_age_value=65 
                    
                    
                Age = cols[0].text_input('Enter the Age ' + str(min_age_value) + ' to ' + str(max_age_value))
                SibSp = int(cols[1].selectbox('SibSp', lis['SibSp']))
                Parch = int(cols[2].selectbox('Parch', lis['Parch']))
                Fare = cols[3].text_input('Enter the Fare btw 4-70')
                Embarked = cols[4].selectbox('Embarked', lis['Embarked'])
                
     
                    
                #To select the sex of the person based on the Name
                if Name in ['Mr.', 'Master.']:
                    Sex = 'Male'
                else:
                    Sex = 'Female'
                    
                state.flag=True
                if (not First_Name):
                    st.write('Provide first name to proceed')
                else:
                    if (cols[0].button('Predict')): 
                        dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                        X = pd.DataFrame(dic, index=[0])
                        try: 
                            X.Age = X.Age.astype('int64')
                            X.Fare = X.Fare.astype('int64')
                            if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 70 or X.Fare.values < 4:
                                st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 4-70 and Select Predict to continue')
                            else:
                                st.markdown(' <h4> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                st.dataframe(titanic_pre(X))
                            
                                x_test = pipeline.transform(X)
                                output, unique, counts = prediction(x_test)
                                if unique[counts.argmax()] == 1:
                                    st.markdown(' <h4> <b> RESULT: You would have Survived based on the input data </b> <h4>', unsafe_allow_html=True)
                                    st.write(' Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                    st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                    if st.button('Refresh'):
                                        state=False
                                elif unique[counts.argmax()] ==0:
                                    st.write('<h4> <b> RESULT: Nee travel Pannirundha Pooiturupa </b> <h4>', unsafe_allow_html=True)
                                    st.write('Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                    st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                    st.write('**Thanks for using the app**. Please press **"Refresh" button** to continue with new prediction. Also, please provide **feedback from sidebar options** once done')
                                    if st.button('Refresh'):
                                        state=False
                                else:
                                    st.write('invaild output - please start from beginning')
                                    
                        except:
                            st.write('Please enter ' + str(min_age_value) + ' to ' + str(max_age_value) +' in Age text box and between 4-70 in Fare text box. Please don\'t provide any string values')
            
            elif Pclass==2: 
                st.markdown('''<div align="center"> <h4> You have selected Pclass as 2 and provide information below details to predict your titanic survival fate<br>  </br> </h4> </div>''', unsafe_allow_html=True                        )
                name_cols = st.beta_columns(5)
                cols = st.beta_columns(5)
                lis = {
                       'Name':['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Rev.', 'Dr.', 'Ms.'], 
                       'SibSp':[0,1,2,3], 
                       'Parch':[0,1,2,3], 
                       'Embarked':['S', 'C', 'Q']
                       }
                Name = name_cols[1].selectbox('Title', lis['Name'])
                First_Name = name_cols[2].text_input('Please Enter Your First Name')
                Last_Name = name_cols[3].text_input('Please Enter Your last Name')
    
                #To select age range from the Name
                if Name == 'Master.':
                    min_age_value=0
                    max_age_value=8
                    
                elif Name=='Mr.':
                    min_age_value=16
                    max_age_value=70
                    
                elif Name=='Miss.':
                    min_age_value=2
                    max_age_value=50
                    
                elif Name=='Mrs.':
                    min_age_value=14
                    max_age_value=55
                    
                elif Name=='Rev.':
                    min_age_value=27
                    max_age_value=57 
                    
                elif Name=='Dr.':
                    min_age_value=23
                    max_age_value=54
                 
                elif Name=='Ms.':
                    min_age_value=20
                    max_age_value=30
                    
                Age = cols[0].text_input('Enter the Age ' + str(min_age_value) + ' to ' + str(max_age_value))
                SibSp = int(cols[1].selectbox('SibSp', lis['SibSp']))
                Parch = int(cols[2].selectbox('Parch', lis['Parch']))
                Fare = cols[3].text_input('Enter the Fare btw 10-75')
                Embarked = cols[4].selectbox('Embarked', lis['Embarked'])
                    
                #To select the sex of the person based on the Name
                if Name in ['Mr.', 'Master.', 'Rev.', 'Dr.', 'Capt.', 'Col.', 'Major.', 'Don.']:
                    Sex = 'Male'
                else:
                    Sex = 'Female'
                    
                state.flag=True
                if (not First_Name):
                    st.write('Provide first name to proceed')
                else:
                    if (cols[0].button('Predict')): 
                        dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                        X = pd.DataFrame(dic, index=[0])
                        try: 
                            X.Age = X.Age.astype('int64')
                            X.Fare = X.Fare.astype('int64')
                            if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 75 or X.Fare.values < 10:
                                st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 10-75 and Select Predict to continue')
                            else:
                                st.markdown('<h4> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                st.dataframe(titanic_pre(X))
                            
                                x_test = pipeline.transform(X)
                                output, unique, counts = prediction(x_test)
                                if unique[counts.argmax()] == 1:
                                    st.markdown(' <h4> <b> RESULT: You would have Survived based on the input data </b> <h4>', unsafe_allow_html=True)
                                    st.write(' Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                    st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                    st.write('**Thanks for using the app**. Please press **"Refresh" button** to continue with new prediction. Also, please provide **feedback from sidebar options** once done')
                                    if st.button('Refresh'):
                                        state=False
                                        
                                elif unique[counts.argmax()] ==0:
                                    st.write('<h4> <b> RESULT: Nee travel Pannirundha Pooiturupa </b> <h4>', unsafe_allow_html=True)
                                    st.write('Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                    st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                    st.write('**Thanks for using the app**. Please press **"Refresh" button** to continue with new prediction. Also, please provide **feedback from sidebar options** once done')
                                    if st.button('Refresh'):
                                        state=False
                                else:
                                    st.write('invaild output - please start from beginning')
                                    
                        except:
                            st.write('Please enter ' + str(min_age_value) + ' to ' + str(max_age_value) +' in Age text box and between 10-75 in Fare text box. Please don\'t provide any string values')
                            
            elif Pclass==1: 
                st.markdown('''<div align="center"> <h4> You have selected Pclass as 1 and provide information below details to predict your titanic survival fate<br>  </br> </h4> </div>''', unsafe_allow_html=True)
                name_cols = st.beta_columns(5)
                cols = st.beta_columns(5)
                lis = {
                       'Name':['Mr.', 'Miss.', 'Mrs.', 'Dr.', 'Master.', 'Mlle.', 'Col.', 'Major.', 'Capt.', 'Don.'], 
                       'Sex':['Male', 'Female'], 
                       'SibSp':[0,1,2,3], 
                       'Parch':[0,1,2,4], 
                       'Embarked':['S', 'C', 'Q']
                       }
                Name = name_cols[1].selectbox('Title', lis['Name'])
                First_Name = name_cols[2].text_input('Please Enter Your First Name')
                Last_Name = name_cols[3].text_input('Please Enter Your last Name')
                #To select age range from the Name
                if Name == 'Master.':
                    min_age_value=0
                    max_age_value=11
                    
                elif Name=='Mr.':
                    min_age_value=17
                    max_age_value=80
                    
                elif Name=='Miss.':
                    min_age_value=2
                    max_age_value=60
                    
                elif Name=='Mrs.':
                    min_age_value=17
                    max_age_value=60
                    
                elif Name=='Rev.':
                    min_age_value=27
                    max_age_value=57 
                    
                elif Name=='Dr.':
                    min_age_value=32
                    max_age_value=50
                 
                elif Name=='Ms.':
                    min_age_value=20
                    max_age_value=30
                    
                elif Name=='Mlle.':
                    min_age_value=20
                    max_age_value=30     
                    
                elif Name=='Col.':
                    min_age_value=55
                    max_age_value=60
                    
                elif Name=='Major.':
                    min_age_value=45
                    max_age_value=50 
                    
                elif Name=='Capt.':
                    min_age_value=65
                    max_age_value=75
                    
                elif Name=='Don.':
                    min_age_value=40
                    max_age_value=50    
  
                    
                Age = cols[0].text_input('Enter the Age ' + str(min_age_value) + ' to ' + str(max_age_value))
                SibSp = int(cols[1].selectbox('SibSp', lis['SibSp']))
                Parch = int(cols[2].selectbox('Parch', lis['Parch']))
                Fare = cols[3].text_input('Enter the Fare btw 5-500')
                Embarked = cols[4].selectbox('Embarked', lis['Embarked'])
                    
                #To select the sex of the person based on the Name
                if Name in ['Mr.', 'Master.', 'Rev.', 'Dr.']:
                    Sex = 'Male'
                else:
                    Sex = 'Female'
                    
                state.flag=True
                if (not First_Name):
                    st.write('Provide first name to proceed')
                else:
                    if (cols[0].button('Predict')): 
                        dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                        X = pd.DataFrame(dic, index=[0])
                        try: 
                            X.Age = X.Age.astype('int64')
                            X.Fare = X.Fare.astype('int64')
                            if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 500 or X.Fare.values < 5:
                                st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 5-500 and Select Predict to continue')
                            else:
                                st.markdown('<h4> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                st.dataframe(titanic_pre(X))
                            
                                x_test = pipeline.transform(X)
                                output, unique, counts = prediction(x_test)
                                if unique[counts.argmax()] == 1:
                                    st.markdown(' <h4> <b> RESULT: You would have Survived based on the input data </b> <h4>', unsafe_allow_html=True)
                                    st.write(' Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                    st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                    st.write('**Thanks for using the app**. Please press **"Refresh" button** to continue with new prediction. Also, please provide **feedback from sidebar options** once done')
                                    if st.button('Refresh'):
                                        state=False
                                        
                                elif unique[counts.argmax()] ==0:
                                    st.write('<h4> <b> RESULT: Nee travel Pannirundha Pooiturupa </b> <h4>', unsafe_allow_html=True)
                                    st.write('Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                    st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                    st.write('**Thanks for using the app**. Please press **"Refresh" button** to continue with new prediction. Also, please provide **feedback from sidebar options** once done')
                                    if st.button('Refresh'):
                                        state=False
                                else:
                                    st.write('invaild output - please start from beginning')
                                    
                        except:
                            st.write('Please enter ' + str(min_age_value) + ' to ' + str(max_age_value) +' in Age text box and between 5-500 in Fare text box. Please don\'t provide any string values')
           
           
