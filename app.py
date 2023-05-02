'''
Created on Wed Oct 26 15:45:07 2022
@author: joshua and qianyee
'''

import pandas as pd
import numpy as np
import pickle
import streamlit as st


st.set_page_config(
    page_title="POME biogas predictor",
    page_icon=":palm_tree:",
    layout="wide",
    initial_sidebar_state="expanded",
    )


#hide_menu_style = """
#        <style>
#        #MainMenu {visibility: hidden;}
#        </style>
#        """
#st.markdown(hide_menu_style, unsafe_allow_html=True)

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

def H2S_prediction(input_data):
    H2S_model=pickle.load(open('model4.sav','rb'))
    scaler_H2S=pickle.load(open('scaler.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_H2S.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    H2S_modelprediction=H2S_model.predict(input_data_reshaped)
    print(H2S_modelprediction)
    return H2S_modelprediction
########################################################################

st.title('POME biogas predictor')
st.caption('This app predicts the biogas output from a closed system POME anaerobic digestion proccess')

#Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Prediction models", "Methodology", "Sustainability", "About"])

########################################################################
#Create title and slider
def main():
    # Giving a title
 with tab1:
        # Sidebar header
        st.sidebar.header('Input Parameters')
        # Define user input features
        def user_input_features():
            POME_in = st.sidebar.slider('POME', 3800, 24000, 12000,1000,"%f")
            COD_in = st.sidebar.slider('COD',55000,92000,65000,1000,"%f")
            BOD_in = st.sidebar.slider('BOD',23000,47000,30000,1000,"%f")
            SS_in = st.sidebar.slider('SS',13000,55000,35000,1000,"%f")
            TS_in = st.sidebar.slider('TS',22000,55000,35000,1000,"%f")
            Temp = st.sidebar.slider('Temperature', 37, 48, 41)
            pH_in = st.sidebar.slider('pH', 6.8, 7.3, 7.0,0.1,"%f")
            OLR = st.sidebar.slider('OLR', 0.86, 1.70, 1.1, 0.01,"%f")
            HRT = st.sidebar.slider('HRT', 35, 85, 50,5,"%f")
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
        
        new_title = '<p style="color:red; font-size: 30px;">Predicting biogas components</p>'
        st.markdown(**new_title**, unsafe_allow_html=True)
        st.caption('The **Gaussian Process Regressor (GPR)** model, **Random Forest (RF)** model and **Extreme Gradient Booosting (XGBoost)** model\
        are among the selected predictors for POME biogas components. The accuracy of the respective models, represented by the :blue[R$^{2}$ coefficient of\
        determination] on the prediction of the target outputs are shown in :blue[_italic_].\
        The predicted components include total biogas production, methane (CH$_{4}$), carbon dioxide (CO$_{2}$) and hydrogen sulphide (H$_{2}$S).')   

        # GPR
        new_title = '<p style="font-family:monospace; color:black; font-size: 20px;">Gaussian Process Regressor (GPR)</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.caption('GPR is a **probabilistic model** bassed on non-parametric kernel models.\
        Unlike linear regression, GPR makes predictions in the form of probability values\
        instead of scalar values [1]. This is achieved by assigning a prior probability to a\
        set of functions, with higher probability given to functions that are more representative\
        of the data. The combination of the prior distribution and the available data points\
        results in a posterior distribution.')
        st.caption('GPR is defined by a function which includes the mean function and a covariance\
        function (otherwise known as a kernel). In this tuned GPR model, the **rational quaduatic** kernel\
        is used.')

        col1, col2, col3 , col4= st.columns(4)

        col1.subheader('Biogas')
        col1.caption(':blue[_0.990_]')
        result_Biogas = Biogas_prediction(df)
        series = pd.Series(result_Biogas[0])
        rounded_Biogas = round(series[0],3)
        col1.write(rounded_Biogas)

        col2.subheader('CH$_{4}$')
        col2.caption(':blue[_0.989_]')
        result_CH4 = CH4_prediction(df)
        series = pd.Series(result_CH4[0])
        rounded_CH4 = round(series[0],3)
        col2.write(rounded_CH4)

        col3.subheader('CO$_{2}$')
        col3.caption(':blue[_0.990_]')
        result_CO2 = CO2_prediction(df)
        series = pd.Series(result_CO2[0])
        rounded_CO2 = round(series[0],3)
        col3.write(rounded_CO2)

        col4.subheader('H$_{2}$S')
        col4.caption(':blue[_0.986_]')
        result_H2S = H2S_prediction(df)
        series = pd.Series(result_H2S[0])
        rounded_H2S = round(series[0],2)
        col4.write(rounded_H2S)

        # XGBoost
        new_title = '<p style="font-family:monospace; color:black; font-size: 20px;">Extreme Gradient Boosting (XGBoost)</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.caption('Proposed by Chen and Guestrin in 2016, the XGBoost algorithm is an optimised version of\
        gradient boosting. Boosting assigns weight to observations and increased the weight of the\
        misclassified observations in subsequent training rounds. Results from each tree are then\
        combined to improve the accuracy of the model. Gradient boosting focuses on reducing the\
        gradient of the loss function in previous models through an iterative feedback process\
        to minimise the degree of error in the gradient direction. The main improvement of XGBoost\
        is the **normalisation of the loss function** using Taylor expansion to mitigate model variances\
        and reduce modelling complexities, which could lead to overfitting.')
        st.caption('The objective function contains a loss function and a regularisation function.\
        The aim is to minimise this function.')

        col1, col2, col3 , col4= st.columns(4)

        col1.subheader('Biogas')
        col1.caption(':blue[_0.961_]')
        result_Biogas = Biogas_prediction(df)
        series = pd.Series(result_Biogas[0])
        rounded_Biogas = round(series[0],3)
        col1.write(rounded_Biogas)

        col2.subheader('CH$_{4}$')
        col2.caption(':blue[_0.951_]')
        result_CH4 = CH4_prediction(df)
        series = pd.Series(result_CH4[0])
        rounded_CH4 = round(series[0],3)
        col2.write(rounded_CH4)

        col3.subheader('CO$_{2}$')
        col3.caption(':blue[_0.950_]')
        result_CO2 = CO2_prediction(df)
        series = pd.Series(result_CO2[0])
        rounded_CO2 = round(series[0],3)
        col3.write(rounded_CO2)

        col4.subheader('H$_{2}$S')
        col4.caption(':blue[_0.947_]')
        result_H2S = H2S_prediction(df)
        series = pd.Series(result_H2S[0])
        rounded_H2S = round(series[0],2)
        col4.write(rounded_H2S)
        
        
        # RF
        new_title = '<p style="font-family:monospace; color:black; font-size: 20px;">Random Forest (RF)</p>'
        st.markdown(new_title, unsafe_allow_html=True)  
        st.caption('Random forest is a non-parametric model as partr of the Ensemble of Trees (EoT) system\
        that was proposed by Breiman in 2001. The Classification and Regression Tree (CART) methodology\
        is applied, where subspace randomisation with bagging is conducted to resample the training set\
        with replacement each time a new tree is grown. This technique trains multiple subsets using\
        bootsttrap replicas of the original ttraining dataset with replacement. This resampling approach\
        generates a diverse set of conditions, whereby the final prediction is based upon the average value\
        from the combined prediction value of each ensemble.')

        col1, col2, col3 , col4= st.columns(4)

        col1.subheader('Biogas')
        col1.caption(':blue[_0.920_]')
        result_Biogas = Biogas_prediction(df)
        series = pd.Series(result_Biogas[0])
        rounded_Biogas = round(series[0],3)
        col1.write(rounded_Biogas)

        col2.subheader('CH$_{4}$')
        col2.caption(':blue[_0.916_]')
        result_CH4 = CH4_prediction(df)
        series = pd.Series(result_CH4[0])
        rounded_CH4 = round(series[0],3)
        col2.write(rounded_CH4)

        col3.subheader('CO$_{2}$')
        col3.caption(':blue[_0.916_]')
        result_CO2 = CO2_prediction(df)
        series = pd.Series(result_CO2[0])
        rounded_CO2 = round(series[0],3)
        col3.write(rounded_CO2)

        col4.subheader('H$_{2}$S')
        col4.caption(':blue[_0.922_]')
        result_H2S = H2S_prediction(df)
        series = pd.Series(result_H2S[0])
        rounded_H2S = round(series[0],2)
        col4.write(rounded_H2S)

        mystyle = '''
            <style>
                p {
                    text-align: justify;
                }
            </style>
            '''
        st.write(mystyle, unsafe_allow_html=True)
        

    ########################################################################
if __name__=='__main__':
    main()
    
    
    
with tab2:
    st.title('Development of prediction models')
    st.caption('To develop the models, 4 stages are followed.')
    
    
    
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
    

    
