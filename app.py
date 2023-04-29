# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:45:07 2022

@author: joshu
"""
import pandas as pd
import numpy as np
import pickle
import streamlit as st


########################################################################
# Creating a function and loading the model
def Biogas_prediction(input_data):
    Biogas_model=pickle.load(open('model1.sav','rb'))
    scaler_Biogas=pickle.load(open('scaler.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_Biogas.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    Biogas_modelprediction=Biogas_model.predict(input_data_reshaped)
    print(Biogas_modelprediction)
    return Biogas_modelprediction

def CH4_prediction(input_data):
    CH4_model=pickle.load(open('model2.sav','rb'))
    scaler_CH4=pickle.load(open('scaler.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_CH4.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    CH4_modelprediction=CH4_model.predict(input_data_reshaped)
    print(CH4_modelprediction)
    return CH4_modelprediction

def CO2_prediction(input_data):
    CO2_model=pickle.load(open('model3.sav','rb'))
    scaler_CO2=pickle.load(open('scaler.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_CO2.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    CO2_modelprediction=CO2_model.predict(input_data_reshaped)
    print(CO2_modelprediction)
    return CO2_modelprediction

def O2_prediction(input_data):
    O2_model=pickle.load(open('model4.sav','rb'))
    scaler_O2=pickle.load(open('scaler.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_O2.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    O2_modelprediction=O2_model.predict(input_data_reshaped)
    print(O2_modelprediction)
    return O2_modelprediction

def H2S_prediction(input_data):
    H2S_model=pickle.load(open('model5.sav','rb'))
    scaler_H2S=pickle.load(open('scaler.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_H2S.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    H2S_modelprediction=H2S_model.predict(input_data_reshaped)
    print(H2S_modelprediction)
    return H2S_modelprediction



########################################################################
#Create title and slider
def main():
    # Giving a title
    st.title('POME')
    st.text('This app predicts the performance of a ________')
    # Sidebar header
    st.sidebar.header('User Input Parameters')
    # Define user input features
    def user_input_features():
        COD_in = st.sidebar.slider('COD',55000,92000,65000)
        BOD_in = st.sidebar.slider('BOD',23000,47000,30000)
        SS_in = st.sidebar.slider('SS',13000,55000,35000)
        TS_in = st.sidebar.slider('TS',22000,55000,35000)
        Temp = st.sidebar.slider('Temperature', 37, 48, 41)
        pH_in = st.sidebar.slider('pH', 6.8, 7.3, 7.0)
        OLR = st.sidebar.slider('OLR', 0.86, 1.70, 1.1, 0.01,"%f")
        HRT = st.sidebar.slider('HRT', 35, 85, 50)
        POME_in = st.sidebar.slider('POME', 3710, 24000, 12000)
        data = {'COD_in': COD_in,
                'BOD_in': BOD_in,
                'TS_in': TS_in,
                'SS_in': SS_in,
                'Temp': Temp,
                'pH_in': pH_in,
                'OLR': OLR,
                'HRT': HRT,
                'POME_in': POME_in,}
        features = pd.DataFrame(data, index=[0])
        return features
# Create user input parameters title    
    df = user_input_features()
    #st.subheader('User Input Parameters')
    #st.write(df)
    
    
########################################################################
# Create subheaders for main performance indicator  
    new_title = '<p style="font-family:monospace; color:red; font-size: 30px;">Key Performance Indicators</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.text('This section displays the key performance indicators of __________')
    
    col1, col2, col3 = st.columns(3)
    
    col1.subheader('Biogas')
    result_Biogas = Biogas_prediction(df)
    series = pd.Series(result_Biogas[0])
    rounded_Biogas = round(series[0],3)
    col1.write(rounded_Biogas)

    col2.subheader('CH4')
    result_CH4 = CH4_prediction(df)
    series = pd.Series(result_CH4[0])
    rounded_CH4 = round(series[0],3)
    col2.write(rounded_CH4)

    col3.subheader('CO2')
    result_CO2 = CO2_prediction(df)
    series = pd.Series(result_CO2[0])
    rounded_CO2 = round(series[0],3)
    col3.write(rounded_CO2)
    
          
    col4, col5, col6 = st.columns(3)   
    col4.subheader('O2')
    result_O2 = O2_prediction(df)
    series = pd.Series(result_O2[0])
    rounded_O2 = round(series[0],2)
    col4.write(rounded_O2) 
    
    col5.subheader('H2S')
    result_H2S = H2S_prediction(df)
    series = pd.Series(result_H2S[0])
    rounded_H2S = round(series[0],2)
    col5.write(rounded_H2S)



########################################################################
if __name__=='__main__':
    main()
    








    # OLD Code
    # Create subheaders for dependent variables
        #st.subheader('Coefficient of Performance')
        #result_COP = COP_prediction(df)
        #rounded = round(result[0],2)
        #st.write(result_COP)
    
    
    
   # st.subheader('Manual Input Section')
    #Getting the input data from the user 
    #RefrigerantFeed = st.number_input('Refrigerant Feed')
    #MolFractionPropane = st.number_input('Mol fraction of propane')
    #DP_LV9004 = st.number_input('Pressure drop across LV-9004')
    #DP_LV9005 = st.number_input('Pressure drop across LV-9005')
    #CondenserDuty = st.number_input('Condenser duty')
    #S12Ratio = st.number_input('Split fraction of S12')
    
    #output =''
    
    #creating a button for prediction
    #if st.button ('Predict the Coefficient of Performance'):
        #result = COP_prediction([[RefrigerantFeed,MolFractionPropane, DP_LV9004, DP_LV9005, CondenserDuty, S12Ratio]])
        #output = round(result[0],2)
    #st.success(output)
    

    
    
    
    
    
    
    
    
    