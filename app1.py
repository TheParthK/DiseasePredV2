import streamlit as st
import pandas as pd 
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components
from fpdf import FPDF
import wikipedia as wiki


encoder = LabelEncoder()
data = pd.read_csv('Dataset/Training.csv').dropna(axis = 1)
data["prognosis"] = encoder.fit_transform(data["prognosis"])
tuple_symptoms = (i for i in data.head())
loaded_model = pickle.load(open('dp_model_final.pkl', 'rb'))

X = data.iloc[:,:-1]

symptoms = X.columns.values


# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}

# Defining the Function
# Input: string containing symptoms separated by commmas
# Output: Generated prediction by model

def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
        
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][loaded_model.predict(input_data)[0]]
    
    # making final prediction by taking mode of all predictions
    # final_prediction = mode([rf_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
    }
    return predictions

def formattingText(text):
    text = text.split('_')
    text = map(str.title, text)
    return ' '.join(text)

def linebreak(n):
    for i in range(n):
        st.write('-'*10)


def UserData():

    st.title("User Data")
    st.caption("Enter your personal details")
    namei, agei, bldgrpi = st.columns(3)
    with namei:
        st.subheader("Name")
        name = st.text_input('Enter your name', label_visibility='collapsed',placeholder='Enter your name')

    with agei:
        st.subheader("Age")
        age = st.number_input('age', label_visibility='collapsed', min_value=0,)

    with bldgrpi:
        st.subheader("Blood Group")
        bldgrp = st.selectbox('bldgrp', ('unspecified', 'O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'), label_visibility='collapsed',)

    linebreak(1)
    
    st.subheader("Select your Symptoms")
    symptomsSelected = st.multiselect(
        "select symtoms",
        list(map(formattingText, list(tuple_symptoms)[:-1])),
        label_visibility="collapsed"
        )

    # st.write('You selected:', options)
    vsymp = st.checkbox('View selected symptoms')
    if vsymp:
        st.caption('Selected Symtoms:')
        c = 0
        if len(symptomsSelected) == 0:
            st.warning("None Selected")
        for i in symptomsSelected:
            c +=1
            st.caption(f'{c}: {i}')

    string_options = ''

    submit_button=st.button(label='Submit')

    if submit_button:
        if symptomsSelected and name and age>0:
            # submit_button = False
            string_options = ','.join(map(formattingText, symptomsSelected))
            predicted_disease = predictDisease(string_options)["rf_model_prediction"]
            # st.write(predicted_disease)
            # sucMsg, downloadButton = st.columns((3, 1))
            # with sucMsg:
            st.success('Submission successful! Navigate to \'Prognosis Info\' from sidebar to view/download prognosis!')
            # with downloadButton:
            # return predicted_disease
            finalUserData = {
                'name': name,
                'age': age,
                'bldgrp': bldgrp,
                'symptoms': symptomsSelected,
                'pred': predicted_disease
            }

            submit_button = False
            return finalUserData
        st.caption('Couldn\'t Submit Form')
        st.error('Form incomplete')
    return False


with st.sidebar:

    st.title('clinical decision support system');
    st.info('select the options below in order to get the prognosis')
    option = st.radio("select the options",("User Data", "Prognosis Info"), )

if option == "User Data":
    # data = UserData()
    st.session_state['all_info'] = UserData()
    # if data
    #     report = FPDF()
    #     report.add_page()
    #     report.set_font('Courier', size=15)
    #     report.cell(200, 10, txt=data['name'], ln=1, align='L')
    #     report.output('report.pdf')

def Prognosis():
    try:
        title, downl = st.columns((3, 1))
        with title:
            st.title("Prognosis Info")
        userData = st.session_state['all_info']
        with downl:
            st.subheader(' ')
            report = FPDF()
            report.add_page()
            report.set_font('Courier', size=20)
            report.cell(200, 13, txt=f'Prognosis Info', ln=1, align='C', border=2)
            report.set_font('Courier', size=13)
            report.cell(200, 10, txt=f'Name: {userData["name"]}{" "*45}Age: {userData["age"]}', ln=1, align='L')
            report.cell(200, 10, txt=f'Blood Group: {userData["bldgrp"]}', ln=1, align='L')
            report.cell(200, 10, txt=f'Symptoms: ', ln=1, align='L')
            report.set_font('Courier', size=9)
            for i in enumerate(userData['symptoms']):
                report.cell(200, 5, txt=f'{i[0]+1}. {i[1]}', ln=1, align='L')
            report.set_font('Courier', size=13)
    
            report.cell(200, 5, txt=' ', ln=1, align='L')
            report.cell(200, 8, txt=f'Decision: ', ln=1, align='L')
            report.set_font('Courier', size=20)
            report.cell(200, 8, txt=userData['pred'], ln=1, align='L')
            
            
            report.output('report.pdf')
            with open('report.pdf', 'rb') as reportPDF:
                PDFbyte = reportPDF.read()

            st.download_button(label="Download Prognosis", 
            data=PDFbyte,
            file_name="report.pdf",
            mime='application/octet-stream')


        st.caption("Here is your Prognosis information!")
        linebreak(1)
        namei, agei, bldgrpi = st.columns(3)
        with namei:
            st.write('Name')
            st.subheader(userData['name'])
        with agei:
            st.write('Age')
            st.subheader(userData['age'])
        with bldgrpi:
            st.write('Blood Group')
            st.subheader(userData['bldgrp'])
        linebreak(1)
        sympi, dec = st.columns(2)
        with sympi:
            st.subheader('Symptoms')
            for i in enumerate(userData['symptoms']):
                st.caption(f'{i[0]+1}: {i[1]}')
        with dec:
            st.subheader('Decision')
            st.title(userData['pred'])
        linebreak(1)
        st.subheader(f'What is {userData["pred"]}?')
        try:
            wikisearch = wiki.search(userData['pred'])[0]
            # st.write(wiki.summary(wiki.search(userData['pred']))[0])
            # st.write(wiki.summary(userData['pred']))
            st.write(wiki.summary(wikisearch))
            st.caption('~via wikipedia')
        except:
            st.caption('Insufficient information on Wikipedia')


        # st.balloons()

    except:
        st.error('User Data incomplete/ not sumbmitted')

if option == "Prognosis Info":
    with st.spinner('Loading data'):
        Prognosis()
    

