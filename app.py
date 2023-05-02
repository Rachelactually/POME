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

col1, col2, col3 = st.beta_columns([4,1,1])

with col1:
    st.title('POME biogas predictor')
    
with col2:
    st.write("")

with col3:
    from PIL import Image
    #opening the image
    image = Image.open('UNMC.png')
    #displaying the image on streamlit app
    st.image(image)
    
st.caption('**© 2023 Website is the creation of Qian Yee Ong, Xin Yun Kiew and Joshua Liew Y. L. \
under the :blue[Department of Chemical with Environmental Engineering], University of Nottingham Malaysia.**')
st.caption('This app predicts the biogas output from a closed system POME anaerobic digestion process.')
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
        
        new_title = '<p style="color:red; font-size: 30px;"><strong>Predicting biogas components<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write('The **Gaussian Process Regressor (GPR)** model, **Random Forest (RF)** model and **Extreme Gradient Booosting (XGBoost)** model\
        are among the selected predictors for POME biogas components. The accuracy of the respective models, represented by the :blue[R$^{2}$ coefficient of\
        determination] on the prediction of the target outputs are shown in :blue[_italic_].\
        The predicted components include total biogas production, methane (CH$_{4}$), carbon dioxide (CO$_{2}$) and hydrogen sulphide (H$_{2}$S).')   

        st.markdown("""
        
        """)
        # GPR
        new_title = '<p style="font-size: 20px;"><strong>Gaussian Process Regressor (GPR)<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)
        with st.beta_expander("Learn more about GPR here."):
            st.write('GPR is a **probabilistic model** bassed on non-parametric kernel models.\
            Unlike linear regression, GPR makes predictions in the form of probability values\
            instead of scalar values [1]. This is achieved by assigning a prior probability to a\
            set of functions, with higher probability given to functions that are more representative\
            of the data. The combination of the prior distribution and the available data points\
            results in a posterior distribution. GPR is defined by a function which includes the mean function and a covariance\
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

        st.markdown("""
        
        """)
            
        # XGBoost
        new_title = '<p style="font-size: 20px;"><strong>Extreme Gradient Boosting (XGBoost)<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)
        with st.beta_expander("Learn more about XGBoost here."):
            st.write('Proposed by Chen and Guestrin in 2016, the XGBoost algorithm is an optimised version of\
            gradient boosting [2]. Boosting assigns weight to observations and increased the weight of the\
            misclassified observations in subsequent training rounds. Results from each tree are then\
            combined to improve the accuracy of the model. Gradient boosting focuses on reducing the\
            gradient of the loss function in previous models through an iterative feedback process\
            to minimise the degree of error in the gradient direction. The main improvement of XGBoost\
            is the **normalisation of the loss function** using Taylor expansion to mitigate model variances\
            and reduce modelling complexities, which could lead to overfitting. The objective function contains a loss function and a regularisation function.\
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

        st.markdown("""
        
        """)
        
        # RF
        new_title = '<p style="font-size: 20px;"><strong>Random Forest (RF)<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)
        with st.beta_expander("Learn more about Random Forest here."):
            st.write('Random forest is a non-parametric model as part of the Ensemble of Trees (EoT) system\
            that was proposed by Breiman in 2001 [3]. The Classification and Regression Tree (CART) methodology\
            is applied, where subspace randomisation with bagging is conducted to resample the training set\
            with replacement each time a new tree is grown. This technique trains multiple subsets using\
            bootstrap replicas of the original ttraining dataset with replacement. This resampling approach\
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
        
        
        reference = '<p font-size: 5px;">[1] Karch, J. D., Brandmaier, A. M. and Voelkle, M. C. (2020) ‘Gaussian Process Panel Modeling—Machine Learning Inspired Analysis of Longitudinal Panel Data’, Frontiers in Psychology, 11. doi: 10.3389/fpsyg.2020.00351.\
        <br>[2] Chen, T. and Guestrin, C. (2016) ‘XGBoost: A Scalable Tree Boosting System’, in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York, NY, USA: ACM, pp. 785–794. doi: 10.1145/2939672.2939785.\
        <br>[3] Breiman, L. (2001) ‘Random forests’, Machine Learning. Springer, 45(1), pp. 5–32. doi: 10.1023/A:1010933404324/METRICS.</p>'
                
        st.write('**References**')
        st.caption('[1] Karch, J. D., Brandmaier, A. M. and Voelkle, M. C. (2020) ‘Gaussian Process Panel Modeling—Machine Learning Inspired Analysis of Longitudinal Panel Data’, Frontiers in Psychology, 11. doi: 10.3389/fpsyg.2020.00351.')
        st.caption('[2] Chen, T. and Guestrin, C. (2016) ‘XGBoost: A Scalable Tree Boosting System’, in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York, NY, USA: ACM, pp. 785–794. doi: 10.1145/2939672.2939785.')
        st.caption('[3] Breiman, L. (2001) ‘Random forests’, Machine Learning. Springer, 45(1), pp. 5–32. doi: 10.1023/A:1010933404324/METRICS.')

    ########################################################################
if __name__=='__main__':
    main()
    
    
########################################################################    
with tab2:
    new_title = '<p style="color:red; font-size: 30px;"><strong>Development of prediction models<strong></p>'
    st.markdown(new_title, unsafe_allow_html=True)   
    st.write('To develop the models, 4 stages are followed.')
        
    with st.beta_expander('**Stage 1: Scoping & data collection**'):
        st.write('Real scale industrial data of the covered lagoon POME anerobic digestor used in this study were\
        obtained from four Malaysian plants over a period of 24 months (July 2019 to June 2021).')
        st.write('The plants inlude (i) Lepar Hilir Palm Oil Mill, (ii) Adela Palm Oil Mill, (iii) Keratong Estate Oil Palm Mill and (iv)\
        Felda Lok Heng Palm Oil Mill. All data used were monthly average values. The collected dataset contains 96 data points, where the input parameters \
        are within the range of:')

        st.metric("POME inlet flowrate (m$^{3}$/month)","3600 to 24200", delta=None)

        col21,col22 = st.columns(2)
        col21.metric("Chemical Oxygen Demand, COD (mg/L)","53500 - 92800", delta=None)
        col22.metric("Biological Oxygen Demand, BOD$_{5}$ (mg/L)","22500 - 47500", delta=None)

        col23,col24 = st.columns(2)
        col23.metric("Total solids, TS (mg/L)","20200 - 56500", delta=None)
        col24.metric("Suspended solids, SS (mg/L)","12300 - 57650", delta=None)

        col25,col26 = st.columns(2)
        col25.metric("Organic loading rate, OLR (kg COD in/m$^{3}$ day)","0.85 - 1.80", delta=None)
        col26.metric("Hydraulic retention time, HRT (days)","34 - 88", delta=None)    

        col27,col28 = st.columns(2)
        col27.metric("Temperature (°C)","47 - 62", delta=None)
        col28.metric("pH","6.80 - 7.40", delta=None)

        st.markdown("""

        """)

    ########################################################################   
    
    with st.beta_expander('**Stage 2: Data pre-processing**'):
   
        st.write('Due to limited available plant data, data expansion was carried out using the \
        Synthetic Minority Oversampling Technique (SMOTE) to generate synthetic datasets. SMOTE uses \
        the k-nearest neighbour approach to synthesise new observations based on the existing dataset.')

        st.write('In this study, the SMOTE algorithm for regression developed by [**Larsen**](https://www.mathworks.com/matlabcentral/fileexchange/75401-synthetic-minority-over-sampling-technique-smote) \
        on MATLAB was employed [1]. **Fig 1** illustrates that to construct a synthetic sample with\
        SMOTE, a random observation from the initial dataset (origin) was chosen. Then, among its \
        nearest neighbours, _k_ number of points with distance _b$_{k}$_ was selected. In accordance\
        with a random assigned weight, _w_, a new sample point, _s$_{k}$_, was crerated along vector\
        _b$_{k}$_. The sampling process was repeated for _n_ times until _N_ was fullfilled.')
        st.write('Prior to model training, SMOTE was applied to the raw dataset to perform data \
        expansion. In theory, more training data should coincide with a model of higher accuracy, \
        and it should therefore be logical to create as many synthetic data as possible. \
        However, as SMOTE is an extrapolation performed upon the original data, a higher accuracy \
        cannot be assumed for higher number of synthesized data.')
        st.write('In this study, the ideal setting for SMOTE, (_N, k_) is found to be (7, 7).\
        Using this setting, a total of 672 datasets containing all input and output parameters\
        were synthesised. As synthesised data will only be used for model training, this leaves the\
        train-test ratio to be at 87.5 to 12.5.')

        from PIL import Image
        #opening the image
        image = Image.open('SMOTE.png')
        #displaying the image on streamlit app
        st.image(image, caption='Fig 1: Random point along the vector connecting the origin to the KNN points.')

        st.write('Prior to training, z-score data normalisation technique was applied to the input variables.\
        Upon performing z-score normalisation, the dataset will be converted into a single, standardised\
        data format, where the new mean and standard deviation values are 0 and 1.')

        st.markdown("""

        """)

    ########################################################################   
    
    with st.beta_expander('**Stage 3: Model Development**'):
        st.markdown("""
        - GPR was imported from the [**scikit learn**](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)\
        library under **"gaussian_process.GaussianProcessRegressor"**.
        - XGBoost was directly imported from the [**xgboost**](https://xgboost.readthedocs.io/en/stable/) library.
        - Random Forest was imported from the [**scikit learn**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)\
        library under **"ensemble.RandomForestRegressor"**
        """)
        
        st.markdown("""

        """)

    ########################################################################     

    with st.beta_expander('**Stage 4: Model Tuning and Validation**'):
        
        st.subheader('Hyperparameter tuning')
        st.write('Hyperparameter tuning was performed to optimise all models prior to performance comparisons.\
        In this study, RandomizedSearchCV was imported from [**sklearn.model_selection**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).\
        Random search implemented a “fit” and “score” method, where sampling without replacement \
        is performed from a list of parameters. This approach is advantageous in several aspects, \
        including superior model fitting, reduced computational time and statistical independence of \
        each trial in continuous processing.\
        A parameter grid for the models was created with a variety of hyperparameter options:')

        new_title = '<p style="font-size: 20px;"><strong>GPR [2]<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)
        h1 = '<p font-size: 5px;"><strong>kernel</strong>\
        <em> specifies covariance function of GPR</em>\
        <br>[RBF, Rational Quadratic, Matern, Exponential]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>n_restarts_optimizer </strong>\
        <em> no. of restarts of the optimizer for to maximise the log-marginal likelihood</em>\
        <br>[4, 6, 8, 20, 12, 15]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>alpha</strong>\
        <em> value added to diagonal of kernel matrix during fitting to obtain a positive definite matrix</em>\
        <br>[1e-10, 1e-5, 1e-2, log-uniform]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        
        new_title = '<p style="font-size: 20px;"><strong>XGBoost [3]<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)           
        h1 = '<p font-size: 5px;"><strong>learning_rate</strong>\
        <em> step size shrinkage to prevent overfitting</em>\
        <br>[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>max_depth </strong>\
        <em> max no. of nodes allowed from root to leaf</em>\
        <br>[3, 4, 5, 6, 7, 8, 9, 10]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>n_estimators</strong>\
        <em> no. of trees in ensemble/ no. of boosting rounds</em>\
        <br>[100, 150, 200, 250,300, 350, 400, 450, 500]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>colsample_bytree</strong>\
        <em> fraction of features to use</em>\
        <br>[0.3, 0.5, 0.8, 1]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>subsample</strong>\
        <em> fraction of observations to subsample at each step</em>\
        <br>[0.3, 0.5, 0.7, 1 ]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        
        new_title = '<p style="font-size: 20px;"><strong>Random forest [2]<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)          
        h1 = '<p font-size: 5px;"><strong>max_depth</strong>\
        <em> max no. of levels in each decision tree</em>\
        <br>[20, 30, 50, 80, 100, None]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>max_features</strong>\
        <em> max no. of features considered for node splitting</em>\
        <br>[sqrt, log2, None]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>min_samples_leaf</strong>\
        <em> min no. of points allowed in a leaf node</em>\
        <br>[1, 2, 4]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>min_samples_split</strong>\
        <em> min no. of points placed in a node before split</em>\
        <br>[2, 5, 10]</p>'
        st.markdown(h1, unsafe_allow_html=True)

        h1 = '<p font-size: 5px;"><strong>n_estimators</strong>\
        <em> no. of trees in the forest</em>\
        <br>[100, 150, 200, 250,300, 350, 400, 450, 500, 600, 800, 1000]</p>'
        st.markdown(h1, unsafe_allow_html=True)
        
        st.markdown("""

        """)
        
        st.subheader('Model validation')
        st.write('To account for overfitting, k-fold cross validation (CV) was also performed \
        during tuning. In k-fold CV, training sets were split into k number of subsets (or folds).\
        The model was then iteratively fitted for k times, where data was trained each time on\
        k-1 of the folds, while evaluated in the k$^{th}$ fold. After training, the performance on\
        each fold was summed and averaged to obtain the final validation metrics for the model.\
        In this study, a 10-fold CV (k = 10) was carried out, as 10 was the most commonly used\
        value in data validation.')
        
        st.markdown("""

        """)
        
        st.subheader('Model evaluation')   
        st.write('To assess the performance of trained models, the predicted value, _y$_{i}$_\
        was compared to the observed (test) value, _x$_{i}$_. Several evaluation metrics came\
        into play to measure the accuracy of the model. The first metric was the :blue[R$^{2}$]\
        score, also known as the coefficient of determination. The R$^{2}$ metric compared the\
        model with a baseline, to be selected via the mean of the observed data. Another metric was\
        the :blue[root mean squared error (RMSE)], which assumed that the errors follow a normal\
        distribution. In RMSE, the square root nature allowed the display of the plausible\
        magnitude of the error term regardless of its negativity.')
        
        st.markdown("""

        """)
        
    st.write('**References**')
    st.caption('[1] Larsen, B. S. (2023) Synthetic Minority Over-sampling Technique (SMOTE). Available at: https://github.com/dkbsl/matlab_smote/releases/tag/1.0 (Accessed: 31 March 2023).')
    st.caption('[2] Pedregosa, F. et al. (2011) ‘Scikit-learn: Machine Learning in Python’, Journal of Machine Learning Research, 12, pp. 2825--2830.')
    st.caption('[3] Developers, X. (2023) xgboost- Release 2.0.0-dev.')

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
