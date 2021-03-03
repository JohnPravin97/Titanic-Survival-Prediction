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
from EDA_utils import make_column_transformerz, columnseparater, standardscaler, categorical_data, DFFeatureUnion
import joblib
import streamlit as st
import pickle
import SessionState
from PIL import Image
import plotly.express as px

try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
def prediction(X):
    output = []
    joblib_file_hard =  "Voting_hard_classifier_model.pkl"
    joblib_file_soft =  "Voting_soft_classifier_model.pkl"
    joblib_file_rf = "random_forest_classifier_model.pkl"

    vc_hard = joblib.load(joblib_file_hard)
    vc_soft = joblib.load(joblib_file_soft)
    rf = joblib.load(joblib_file_rf)
    
    output.append(vc_hard.predict(X))
    output.append(vc_soft.predict(X))
    output.append(rf.predict(X))
    unique, counts = np.unique(np.asarray(output), return_counts=True)
    
    return output, unique, counts


def visualize_titanic_pre(df):
    df['Family_group'] = pd.cut(
        df.SibSp, bins=[-1, 2, 4, 10], labels=['Small', 'Medium', 'Large'])
    df['Parent_Count'] = pd.cut(
        df.Parch, bins=[-1, 0.5, 1.5, 10], labels=['No_parent', 'single_parent', 'big_family'])
    df.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)
    for i in range(1, 4):
        df.Fare[df.Pclass.eq(i)] = df.Fare[df.Pclass.eq(i)].replace(
            0, df.Fare[df.Pclass.eq(i)].mean())
    df.Age = df.Age.replace(np.NaN, 0)
    age_mean = np.mean(df.Age)
    df.Age = (df.Age.replace(0, age_mean))
    df.dropna(inplace=True)
    df.Survived = df.Survived.replace(1, 'Survived')
    df.Survived = df.Survived.replace(0, 'Not Survived')
    return df

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

theme = st.beta_columns(8)
agree = theme[7].checkbox('Dark Theme')

if agree: 
    st.markdown(
        f"""
        <style>
        .reportview-container {{
    background-color: #000000;
    background-image: none;
    color: #ffffff          
        }}
    
        </style>
        """,
        unsafe_allow_html=True
    )
        
    st.markdown("""<style>
    .css-1aumxhk {
    background-color: #000000;
    background-image: none;
    color: #ffffff
    }
    </style>""",  unsafe_allow_html=True)
            
    dataset = 'Titanic_Complete_Dataset.csv'
    df = visualize_titanic_pre(pd.read_csv(dataset))
    inp = st.sidebar.selectbox('', ['Titanic Prediction', 'Titanic Visualization'])
        
    if inp == 'Titanic Visualization':
        local_css("style.css")
        state = SessionState.get(flag=False)
        st.sidebar.markdown('''<h1 style="color:#ffffff"> Welcome to Titanic Prediction Tool</h1>''', unsafe_allow_html=True)
        st.sidebar.markdown('''<h3 style="color:#ce222f"> <b> <u>
    Below are the creator details: </u></b></h3>''', unsafe_allow_html=True)
        st.sidebar.markdown('''#### Name : John Pravin (<johnpravina@gmail.com>)\n
#### LinkedIn :  <https://www.linkedin.com/in/john-pravin/> 
#### GitHub : <https://github.com/JohnPravin97> ''')
        st.sidebar.markdown('''<h3 style="color:#ce222f"> <u> Special thanks to </u> </h3>''', unsafe_allow_html=True)
        st.sidebar.markdown('''
                            #### Name : Aakash Nagarajan (<https://www.linkedin.com/in/aakash-nagarajan-28325510a/>)''')
        
        
        #Main Coding
        st.markdown('''<div align="center"> <h1> <b> Welcome to Titanic Visualization Tool </b> </h1> </div>''', unsafe_allow_html=True)
        img = st.beta_columns(3)
        img[1].image(Image.open('Titanic_Visualization.jpg'), width=425, caption='Titanic')
        
        st.markdown('''
        <h3 style="color:#ce222f"> <u> <b> INTRODUCTION: </b> </h3> </u>''', unsafe_allow_html=True)
        
        st.markdown('''This is a visual representation of the original titanic dataset, the plot represents the details of entries in the dataset. The details about attributes of these entries can be found in the visual representation.\n
    ''')
        
        st.markdown('''
        <h3 style="color:#ce222f"> <b> <u> USER GUIDES: </b> </h3> </u>''', unsafe_allow_html=True)
        st.markdown(''' Points to consider while working with this tool: \n
                    1. User can press me 'Visualize Me' button with the default selections to visualize the dataset. 
                    2. Plotly library is used to make the visualization of original titanic dataset. 
                    3. Thus, plotly features are available on the top right corner of the plot. 
                    4. The Animations and Size columns has two options: 1. Pclass, 2. Sex and 1. SibSp, 2.Parch respectively.
                    5. User has to switch back to 'Titanic Prediction' for description of the columns. 
                    6. User has to switch back to 'Titanic Prediction' for providing feedback and comments.
                    ''')
        st.markdown('''
        <h3 style="color:#ce222f"> <u> <b> LETS GET STARTED: </b> </h3> </u>''', unsafe_allow_html=True)
            
        cols = st.beta_columns(6)
        x_axis = cols[0].selectbox('X-axis', ['Age'])
        y_axis = cols[1].selectbox('Y_axis', ['Fare'])
        colors = cols[2].selectbox('Colors', ['Survived'])
        columns = cols[3].selectbox('Facet-Cols', ['Sex', 'Pclass'])
        size = cols[5].selectbox('Size', ['Parch', 'SibSp'])
        hover_name = 'Name'
        hover_data = ['SibSp', 'Parch', 'Embarked']
        if columns == 'Pclass':
            animation = cols[4].selectbox('Animations', ['Sex'])
        elif columns =='Sex':
            animation = cols[4].selectbox('Animations', ['Pclass'])
            
        if cols[3].button('Visualize Me'):
            st.markdown('***Please click "Autoscale" in plotly to visualize it efficiently***')
            st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, title='Titanic Visualization',size=size, color = colors, facet_col = columns, hover_name=hover_name, hover_data=hover_data, animation_frame=animation, height=600, width=1000))
                              
      
    else:
        local_css("style.css")
        state = SessionState.get(flag=False)
        st.sidebar.markdown('''<h1 style="color:#ffffff"> Welcome to Titanic Prediction Tool</h1>''', unsafe_allow_html=True)
        st.sidebar.markdown('''<h3 style="color:#ce222f"> <b> <u>
    Below are the creator details: </u></b></h3>''', unsafe_allow_html=True)
        st.sidebar.markdown('''#### Name : John Pravin (<johnpravina@gmail.com>)\n
#### LinkedIn :  <https://www.linkedin.com/in/john-pravin/> 
#### GitHub : <https://github.com/JohnPravin97> ''')
        st.sidebar.markdown('''<h3 style="color:#ce222f"> <u> Special thanks to </u> </h3>''', unsafe_allow_html=True)
        st.sidebar.markdown('''
    #### Name : Aakash Nagarajan (<https://www.linkedin.com/in/aakash-nagarajan-28325510a/>)''')
        feedback_save = pd.HDFStore('feedback.h5')
        comment_save = pd.HDFStore('comment.h5')      
        
        st.sidebar.markdown('''<h3 style="color:#ce222f"> <u>Feedback: </u></h3>''', unsafe_allow_html=True)
        Name = st.sidebar.text_input('Please Enter Your Name*')
        feedlist = ['Better', 'Normal', 'Worst']
        feedback = st.sidebar.radio('Please provide your feedback below*', feedlist)
      
        feedback_comment = st.sidebar.text_input('Additional Comments')
        st.sidebar.markdown('''<p style="font-size:60%;"> * mandatory fields  </p>''', unsafe_allow_html=True)
        if st.sidebar.button('Send Feedback'):
            if (not Name):
                st.sidebar.markdown('''<p style="font-size:80%;"> <b> Please provide a name to save your feedback </b></p>''', unsafe_allow_html=True)
                
            else: 
                feedback_df = pd.DataFrame(feedback, index=[0], columns=['Feedback'])
                feedback_df['Name']=Name.upper()
                feedback_save.put(Name.upper(), feedback_df)
                feedback_save.close()
                if (not feedback_comment):
                    pass
                else:
                    comment_df = pd.DataFrame(feedback_comment, index=[0], columns=['Comments'])
                    comment_df['Name']=Name.upper()
                    comment_save.put(Name.upper(), comment_df)
                    comment_save.close()                
                st.sidebar.write('Thanks for your feedback')
                
        st.sidebar.markdown('''<h3 style="color:#ce222f"><u> Disclaimer: </u> </h3>''', unsafe_allow_html=True) 
        st.sidebar.write('Input name and data are being stored for further improvements of the tool')
            
        # Main Coding
        st.markdown('''<div align="center"> <h1 style="color:#ffffff">Welcome to Titanic Survival Prediction Tool </b> </h1> </div>''', unsafe_allow_html=True)
    
        img = st.beta_columns(3)
        img[1].image(Image.open('Titanic.jpg'), width=425, caption='Titanic')
        
        
        st.markdown('''
        <h3 style="color:#ce222f"><u>  <b> INTRODUCTION: </b> </h3> </u>''', unsafe_allow_html=True)
        
        st.markdown('''<p style="color:#ffffff">Ever had a dream of travelling on the titanic? ever wondered if you would have survived the unfortunate incident if you got a ticket aboard the dreamy cruise for a journey in the year 1912? We got you covered:\n
<p style="color:#ffffff">This is a prediction system which can be used to predict the chance of a person surviving the titanic incident. The system asks the user input for data and predicts the chance of survival of the user in the titanic incident. This model is trained using the original titanic data set with basic machine learning techniques like regression, and to increase the accuracy of prediction it was later combined with advanced machine learning techniques such as the ensemble learning and integrated all of these into an end to end pipeline architecture.\n
<p style="color:#ffffff"> The following are the data one has to provide : Pclass, Name, Sex, Age, Fare, SibSp, Parch, Embarked. Refer the below describtion for more detailed explanation on each parameters </p>''', unsafe_allow_html=True)
        
        markdwn = st.beta_columns(3)
        markdw = st.beta_columns(2)
        
        markdwn[0].markdown('''
        <h3 style="color:#ce222f"> <u> <b> VARIABLE DESCRIPTIONS: </b> </h3> </u>''', unsafe_allow_html=True)
        
        markdw[0].markdown(''' <div style="color:#ffffff">
            \t <b> Pclass </b>  -> Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)\n <div style="color:#ffffff">
            <b>Name  </b>   -> Name of the passenger\n <div style="color:#ffffff">
            <b>Sex </b>    -> Sex\n <div style="color:#ffffff">
            <b>Age </b>   -> Age\n </div> <div style="color:#ffffff">
        ''', unsafe_allow_html=True)
        
        markdw[1].markdown('''<div style="color:#ffffff">
        <b>Sibsp</b>  -> Number of Siblings/Spouses Aboard\n <div style="color:#ffffff">
        <b>Parch</b>    -> Number of Parents/Children Aboard\n <div style="color:#ffffff">
        <b>Fare</b>     -> Passenger Fare (British pound)\n <div style="color:#ffffff">
        <b>Embarked</b> -> Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)\n
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <h3 style="color:#ce222f"> <u> <b> USER GUIDES: </b> </h3> </u>''', unsafe_allow_html=True)
        st.markdown(''' Points to consider while working with this tool: \n
                    1. User has to fill the Pclass and press 'Enter' button to get the remaining parameters.
                    2. User has to fill his first name to proceed further with the prediction.
                    3. User has to follow and provide the values between the range mentioned for Age and Fare.
                    4. Upon filling all the required parameters, press 'Predict' button to proceed.
                    5. After filling all the parameters, the users are provided with dataframe containing their inputs for reference.
                    6. Sex parameter is automatically taken based on the Title given.
                    7. Family Group and Parent Count are calculated based on the SibSp and Parch parameters. 
                    8. Press 'Refresh' button to try with new data.
                    9. Select 'Titanic Visualization' box to visualize the original titanic dataset. 
                    10. Provide feedback and comments from sidebar for further improving the tool.
                    11. Please note that same name with two predictions are not allowed. User need to change the name and data for predicting again.
                    ''')
        st.markdown('''
        <h3 style="color:#ce222f"><u> <b> LETS GET STARTED: </b> </h3> </u>''', unsafe_allow_html=True)            
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
                    st.markdown('''<div align="center" style="color:#ffffff"> <h4> You have selected Pclass as 3 and provide information below details to predict your titanic survival fate<br>  </br> </h4> </div>''', unsafe_allow_html=True                        )
                    name_cols = st.beta_columns(5)
                    cols = st.beta_columns(5)
                    lis = {
                           'Name':['Mr.', 'Miss.', 'Mrs.', 'Master.'],  
                           'SibSp':[0,1,2,3,4,5,8], 
                           'Parch':[0,1,2,3,4,5,6], 
                           'Embarked':['S', 'C', 'Q']
                           }
                    Name = name_cols[1].selectbox('Pronoun', lis['Name'])
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
                    
                    name_temp = str(First_Name).upper()
                        
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
                            dataframe_save = pd.HDFStore('titanic_search_dataframe.h5')
                            try: 
                                dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                                X = pd.DataFrame(dic, index=[0])
                                X.Age = X.Age.astype('int64')
                                X.Fare = X.Fare.astype('int64')
                                if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 70 or X.Fare.values < 4:
                                    st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 4-70 and Select Predict to continue')
                                else:
                                    if ('/' + name_temp) in dataframe_save.keys():
                                        st.markdown('**Prediction for this data is already done. Please change the data and first name to continue**')
                                        df = dataframe_save.get(name_temp)
                                        st.dataframe(df)
                                    else:
                                        st.markdown(' <h4 style="color:#ce222f"> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                        temp = titanic_pre(X)
                                        st.dataframe(temp)
                                        x_test = pipeline.transform(X)
                                        output, unique, counts = prediction(x_test)
                                        temp_2 = unique[counts.argmax()]
                                        temp['Survived'] = temp_2
                                        dataframe_save.put(name_temp, temp, data_columns=True)
                                        dataframe_save.close()
                                        if unique[counts.argmax()] == 1:
                                            st.markdown(' <h4 style="color:#ce222f"> <b> RESULT: You would have Survived based on the input data </b> <h4>', unsafe_allow_html=True)
                                            st.write(' Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                            st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                            if st.button('Refresh'):
                                                state=False
                                        elif unique[counts.argmax()] ==0:
                                            st.write('<h4 style="color:#ce222f"> <b> RESULT: Nee travel Pannirundha Pooiturupa </b> <h4>', unsafe_allow_html=True)
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
                    Name = name_cols[1].selectbox('Pronoun', lis['Name'])
                    First_Name = name_cols[2].text_input('Please Enter Your First Name')
                    Last_Name = name_cols[3].text_input('Please Enter Your last Name')
                    
                    name_temp = str(First_Name).upper()
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
                            dataframe_save = pd.HDFStore('titanic_search_dataframe.h5')
                            try:
                                dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                                X = pd.DataFrame(dic, index=[0])
                                X.Age = X.Age.astype('int64')
                                X.Fare = X.Fare.astype('int64')
                                if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 75 or X.Fare.values < 10:
                                    st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 10-75 and Select Predict to continue')
                                else:
                                    if ('/' + name_temp) in dataframe_save.keys():
                                        st.markdown('**Prediction for this data is already done. Please change the data and first name to continue**')
                                        df = dataframe_save.get(name_temp)
                                        st.dataframe(df)
                                    else:
                                        st.markdown('<h4 style="color:#ce222f"> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                        temp = titanic_pre(X)
                                        st.dataframe(temp)
                                        x_test = pipeline.transform(X)
                                        output, unique, counts = prediction(x_test)
                                        temp_2 = unique[counts.argmax()]
                                        temp['Survived'] = temp_2
                                        dataframe_save.put(name_temp, temp, data_columns=True)
                                        dataframe_save.close()
                                        if unique[counts.argmax()] == 1:
                                            st.markdown(' <h4 style="color:#ce222f"> <b> RESULT: You would have Survived based on the input data </b> <h4>', unsafe_allow_html=True)
                                            st.write(' Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                            st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                            st.write('**Thanks for using the app**. Please press **"Refresh" button** to continue with new prediction. Also, please provide **feedback from sidebar options** once done')
                                            if st.button('Refresh'):
                                                state=False
                                                
                                        elif unique[counts.argmax()] ==0:
                                            st.write('<h4 style="color:#ce222f"> <b> RESULT: Nee travel Pannirundha Pooiturupa </b> <h4>', unsafe_allow_html=True)
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
                    Name = name_cols[1].selectbox('Pronoun', lis['Name'])
                    First_Name = name_cols[2].text_input('Please Enter Your First Name')
                    Last_Name = name_cols[3].text_input('Please Enter Your last Name')
                    
                    name_temp = str(First_Name).upper()
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
                            dataframe_save = pd.HDFStore('titanic_search_dataframe.h5')
                            try: 
                                dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                                X = pd.DataFrame(dic, index=[0])
                                X.Age = X.Age.astype('int64')
                                X.Fare = X.Fare.astype('int64')
                                if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 500 or X.Fare.values < 5:
                                    st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 5-500 and Select Predict to continue')
                                else:
                                    if ('/' + name_temp) in dataframe_save.keys():
                                        st.markdown('**Prediction for this data is already done. Please change the data and first name to continue**')
                                        df = dataframe_save.get(name_temp)
                                        st.dataframe(df)
                                    else:
                                        st.markdown('<h4 style="color:#ce222f"> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                        temp = titanic_pre(X)
                                        st.dataframe(temp)
                                        x_test = pipeline.transform(X)
                                        output, unique, counts = prediction(x_test)
                                        temp_2 = unique[counts.argmax()]
                                        temp['Survived'] = temp_2
                                        dataframe_save.put(name_temp, temp, data_columns=True)
                                        dataframe_save.close()
                                        if unique[counts.argmax()] == 1:
                                            st.markdown(' <h4 style="color:#ce222f"> <b> RESULT: You would have Survived based on the input data </b> <h4>', unsafe_allow_html=True)
                                            st.write(' Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                            st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                            st.write('**Thanks for using the app**. Please press **"Refresh" button** to continue with new prediction. Also, please provide **feedback from sidebar options** once done')
                                            if st.button('Refresh'):
                                                state=False
                                                
                                        elif unique[counts.argmax()] ==0:
                                            st.write('<h4 style="color:#ce222f"> <b> RESULT: Nee travel Pannirundha Pooiturupa </b> <h4>', unsafe_allow_html=True)
                                            st.write('Please refer the below dataframe for each model outputs: **0 -> Not Survived, 1 -> Survived** ', unsafe_allow_html=True)
                                            st.dataframe(pd.DataFrame(output, index=['Voting_Classifier_Hard', 'Voting_Classifier_Soft', 'RandomForest'], columns=['Output']))
                                            st.write('**Thanks for using the app**. Please press **"Refresh" button** to continue with new prediction. Also, please provide **feedback from sidebar options** once done')
                                            if st.button('Refresh'):
                                                state=False
                                        else:
                                            st.write('invaild output - please start from beginning')
                            except:
                                st.write('Please enter ' + str(min_age_value) + ' to ' + str(max_age_value) +' in Age text box and between 5-500 in Fare text box. Please don\'t provide any string values')
                   
   
else:
    dataset = 'Titanic_Complete_Dataset.csv'
    df = visualize_titanic_pre(pd.read_csv(dataset))
    inp = st.sidebar.selectbox('', ['Titanic Prediction', 'Titanic Visualization'])
        
    if inp == 'Titanic Visualization':
        state = SessionState.get(flag=False)
        st.sidebar.markdown('''# Welcome to Titanic Visualization Tool
### Below are the creator details
#### Name : John Pravin (<johnpravina@gmail.com>)
#### LinkedIn :  <https://www.linkedin.com/in/john-pravin/> 
#### GitHub : <https://github.com/JohnPravin97> 
### Special thanks to
#### Name : Aakash Nagarajan (<https://www.linkedin.com/in/aakash-nagarajan-28325510a/>)''')
        
        #Main Coding
        st.markdown('''<div align="center"> <h1> <b> Welcome to Titanic Visualization Tool </b> </h1> </div>''', unsafe_allow_html=True)
        img = st.beta_columns(3)
        img[1].image(Image.open('Titanic_Visualization.jpg'), width=425, caption='Titanic')
        
        st.markdown('''<u>
        <h3> <b> INTRODUCTION: </b> </h3> </u>''', unsafe_allow_html=True)
        
        st.markdown('''This is a visual representation of the original titanic dataset, the plot represents the details of entries in the dataset. The details about attributes of these entries can be found in the visual representation.\n
    ''')
        
        st.markdown('''<u>
        <h3> <b> USER GUIDES: </b> </h3> </u>''', unsafe_allow_html=True)
        st.markdown(''' Points to consider while working with this tool: \n
                    1. User can press me 'Visualize Me' button with the default selections to visualize the dataset. 
                    2. Plotly library is used to make the visualization of original titanic dataset. 
                    3. Thus, plotly features are available on the top right corner of the plot. 
                    4. The Animations and Size columns has two options: 1. Pclass, 2. Sex and 1. SibSp, 2.Parch respectively.
                    5. User has to switch back to 'Titanic Prediction' for description of the columns. 
                    6. User has to switch back to 'Titanic Prediction' for providing feedback and comments.
                    ''')
        st.markdown('''<u>
        <h3> <b> LETS GET STARTED: </b> </h3> </u>''', unsafe_allow_html=True)
            
        cols = st.beta_columns(6)
        x_axis = cols[0].selectbox('X-axis', ['Age'])
        y_axis = cols[1].selectbox('Y_axis', ['Fare'])
        colors = cols[2].selectbox('colors', ['Survived'])
        columns = cols[3].selectbox('Facet-Cols', ['Sex', 'Pclass'])
        size = cols[5].selectbox('Size', ['Parch', 'SibSp'])
        hover_name = 'Name'
        hover_data = ['SibSp', 'Parch', 'Embarked']
        if columns == 'Pclass':
            animation = cols[4].selectbox('Animations', ['Sex'])
        elif columns =='Sex':
            animation = cols[4].selectbox('Animations', ['Pclass'])
            
        if cols[3].button('Visualize Me'):
            st.markdown('***Please wait while it is loading and click "Autoscale" in plotly to visualize it efficiently***')
            st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, title='Titanic Visualization',size=size, color = colors, facet_col = columns, hover_name=hover_name, hover_data=hover_data, animation_frame=animation, height=600, width=1200))
                              
      
    else:
        state = SessionState.get(flag=False)
        st.sidebar.markdown('''# Welcome to Titanic Prediction Tool\n
### Below are the creator details
#### Name : John Pravin (<johnpravina@gmail.com>)\n
#### LinkedIn :  <https://www.linkedin.com/in/john-pravin/> 
#### GitHub : <https://github.com/JohnPravin97> 
### Special thanks to
#### Name : Aakash Nagarajan (<https://www.linkedin.com/in/aakash-nagarajan-28325510a/>)''')
        feedback_save = pd.HDFStore('feedback.h5')
        comment_save = pd.HDFStore('comment.h5')      
        
        st.sidebar.markdown('''### Feedback \n''')
        Name = st.sidebar.text_input('Please Enter Your Name*')
        feedlist = ['Better', 'Normal', 'Worst']
        feedback = st.sidebar.radio('Please provide your feedback below*', feedlist)
      
        feedback_comment = st.sidebar.text_input('Additional Comments')
        st.sidebar.markdown('''<p style="font-size:60%;"> * mandatory fields  </p>''', unsafe_allow_html=True)
        if st.sidebar.button('Send Feedback'):
            if (not Name):
                st.sidebar.markdown('''<p style="font-size:80%;"> <b> Please provide a name to save your feedback </b></p>''', unsafe_allow_html=True)
                
            else: 
                feedback_df = pd.DataFrame(feedback, index=[0], columns=['Feedback'])
                feedback_df['Name']=Name.upper()
                feedback_save.put(Name.upper(), feedback_df)
                feedback_save.close()
                if (not feedback_comment):
                    pass
                else:
                    comment_df = pd.DataFrame(feedback_comment, index=[0], columns=['Comments'])
                    comment_df['Name']=Name.upper()
                    comment_save.put(Name.upper(), comment_df)
                    comment_save.close()                
                st.sidebar.write('Thanks for your feedback')
                
        st.sidebar.markdown('''### Disclaimer''') 
        st.sidebar.write('Input name and data are being stored for further improvements of the tool')
            
        # Main Coding
        st.markdown('''<div align="center"> <h1> <b> Welcome to Titanic Survival Prediction Tool </b> </h1> </div>''', unsafe_allow_html=True)
    
        img = st.beta_columns(3)
        img[1].image(Image.open('Titanic.jpg'), width=425, caption='Titanic')
        
        
        st.markdown('''<u>
        <h3> <b> INTRODUCTION: </b> </h3> </u>''', unsafe_allow_html=True)
        
        st.markdown('''Ever had a dream of travelling on the titanic? ever wondered if you would have survived the unfortunate incident if you got a ticket aboard the dreamy cruise for a journey in the year 1912? We got you covered:\n
This is a prediction system which can be used to predict the chance of a person surviving the titanic incident. The system asks the user input for data and predicts the chance of survival of the user in the titanic incident. This model is trained using the original titanic data set with basic machine learning techniques like regression, and to increase the accuracy of prediction it was later combined with advanced machine learning techniques such as the ensemble learning and integrated all of these into an end to end pipeline architecture.\n
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
                    8. Press 'Refresh' button to try with new data.
                    9. Select 'Titanic Visualization' box to visualize the original titanic dataset. 
                    10. Provide feedback and comments from sidebar for further improving the tool.
                    11. Please note that same name with two predictions are not allowed. User need to change the name and data for predicting again.
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
                    Name = name_cols[1].selectbox('Pronoun', lis['Name'])
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
                    
                    name_temp = str(First_Name).upper()
                        
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
                            dataframe_save = pd.HDFStore('titanic_search_dataframe.h5')
                            try: 
                                dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                                X = pd.DataFrame(dic, index=[0])
                                X.Age = X.Age.astype('int64')
                                X.Fare = X.Fare.astype('int64')
                                if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 70 or X.Fare.values < 4:
                                    st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 4-70 and Select Predict to continue')
                                else:
                                    if ('/' + name_temp) in dataframe_save.keys():
                                        st.markdown('**Prediction for this data is already done. Please change the data and first name to continue**')
                                        df = dataframe_save.get(name_temp)
                                        st.dataframe(df)
                                    else:
                                        st.markdown(' <h4> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                        temp = titanic_pre(X)
                                        st.dataframe(temp)
                                        x_test = pipeline.transform(X)
                                        output, unique, counts = prediction(x_test)
                                        temp_2 = unique[counts.argmax()]
                                        temp['Survived'] = temp_2
                                        dataframe_save.put(name_temp, temp, data_columns=True)
                                        dataframe_save.close()
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
                    Name = name_cols[1].selectbox('Pronoun', lis['Name'])
                    First_Name = name_cols[2].text_input('Please Enter Your First Name')
                    Last_Name = name_cols[3].text_input('Please Enter Your last Name')
                    
                    name_temp = str(First_Name).upper()
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
                            dataframe_save = pd.HDFStore(r'titanic_search_dataframe.h5')
                            try:
                                dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                                X = pd.DataFrame(dic, index=[0])
                                X.Age = X.Age.astype('int64')
                                X.Fare = X.Fare.astype('int64')
                                if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 75 or X.Fare.values < 10:
                                    st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 10-75 and Select Predict to continue')
                                else:
                                    if ('/' + name_temp) in dataframe_save.keys():
                                        st.markdown('**Prediction for this data is already done. Please change the data and first name to continue**')
                                        df = dataframe_save.get(name_temp)
                                        st.dataframe(df)
                                    else:
                                        st.markdown('<h4> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                        temp = titanic_pre(X)
                                        st.dataframe(temp)
                                        x_test = pipeline.transform(X)
                                        output, unique, counts = prediction(x_test)
                                        temp_2 = unique[counts.argmax()]
                                        temp['Survived'] = temp_2
                                        dataframe_save.put(name_temp, temp, data_columns=True)
                                        dataframe_save.close()
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
                    Name = name_cols[1].selectbox('Pronoun', lis['Name'])
                    First_Name = name_cols[2].text_input('Please Enter Your First Name')
                    Last_Name = name_cols[3].text_input('Please Enter Your last Name')
                    
                    name_temp = str(First_Name).upper()
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
                            dataframe_save = pd.HDFStore(r'C:\Users\jpravijo\Desktop\Anaconda\Streamlit\Titanic Prediction\titanic_search_dataframe.h5')
                            try: 
                                dic = {'Pclass': Pclass, 'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Parch': Parch, 'Fare': Fare, 'Embarked': Embarked}
                                X = pd.DataFrame(dic, index=[0])
                                X.Age = X.Age.astype('int64')
                                X.Fare = X.Fare.astype('int64')
                                if X.Age.values > max_age_value or X.Age.values < min_age_value or X.Fare.values > 500 or X.Fare.values < 5:
                                    st.write('Please provide Age between ' + str(min_age_value) + ' to ' + str(max_age_value) +', Fare value between 5-500 and Select Predict to continue')
                                else:
                                    if ('/' + name_temp) in dataframe_save.keys():
                                        st.markdown('**Prediction for this data is already done. Please change the data and first name to continue**')
                                        df = dataframe_save.get(name_temp)
                                        st.dataframe(df)
                                    else:
                                        st.markdown('<h4> <b> Input Dataframe for reference </b> <h4>', unsafe_allow_html=True)
                                        temp = titanic_pre(X)
                                        st.dataframe(temp)
                                        x_test = pipeline.transform(X)
                                        output, unique, counts = prediction(x_test)
                                        temp_2 = unique[counts.argmax()]
                                        temp['Survived'] = temp_2
                                        dataframe_save.put(name_temp, temp, data_columns=True)
                                        dataframe_save.close()
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
                   
                                   
