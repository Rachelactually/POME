'''
Created on Wed Oct 26 15:45:07 2022
@author: joshua and qianyee
'''

import pandas as pd
import numpy as np
import pickle
import streamlit as st
import altair as alt
from PIL import Image

filename = 'model1.sav'
pickle.dump(modelgpr1, open(filename, 'wb'))

#opening the image
image1 = Image.open('UNMClogo.jpeg')

st.set_page_config(
    page_title="POME biogas predictor",
    page_icon=image1,
    #layout="wide",
    initial_sidebar_state="expanded",
    )


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



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

col1, col2, col3 = st.columns([4,1,1])

with col1:
    st.title('POME biogas predictor')
    
with col2:
    st.write("")

with col3:
    
    #opening the image
    image = Image.open('UNMC.png')
    #displaying the image on streamlit app
    st.image(image)
    
st.caption('© 2023 Website is the creation of **Q.Y. Ong**, **X.Y. Kiew** and **Joshua Liew Y.L.** \
under the :blue[**Department of Chemical with Environmental Engineering**], **University of Nottingham Malaysia.**')
st.write('**This app predicts the biogas output from a closed system POME anaerobic digestion process.**')
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
        
        new_title = '<p style="text-align:left; color:red; font-size: 30px;"><strong>Predicting biogas components<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write('The **Gaussian Process Regressor (GPR)** model, **Random Forest (RF)** model and **Extreme Gradient Booosting (XGBoost)** model\
        are among the selected predictors for POME biogas components. The accuracy of the respective models, represented by the :blue[R$^{2}$ coefficient of\
        determination] on the prediction of the target outputs are shown in :blue[_italic_].\
        The predicted components include total biogas production, methane (CH$_{4}$), carbon dioxide (CO$_{2}$) and hydrogen sulphide (H$_{2}$S).')
        
        st.write(':blue[**Try out our predictor!**]')
        st.info('On the top left of the screen, click on **>** to specify input values.')
        #st.text_area(':blue[**Try out our predictor!**]',':blue[On the top left of the screen, click on **>** to specify input values.]')

        st.markdown("""
        
        """)
        # GPR
        new_title = '<p style="font-size: 20px;"><strong>Gaussian Process Regressor (GPR)<strong></p>'
        st.markdown(new_title, unsafe_allow_html=True)
        with st.beta_expander("Learn more about GPR here."):
            st.write('GPR is a **probabilistic model** based on non-parametric kernel models.\
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
            bootstrap replicas of the original training dataset with replacement. This resampling approach\
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
        
             
        st.write('**References**')
        st.caption('[1] Karch, J. D., Brandmaier, A. M. and Voelkle, M. C. (2020) ‘Gaussian Process Panel Modeling—Machine Learning Inspired Analysis of Longitudinal Panel Data’, Frontiers in Psychology, 11. doi: 10.3389/fpsyg.2020.00351.')
        st.caption('[2] Chen, T. and Guestrin, C. (2016) ‘XGBoost: A Scalable Tree Boosting System’, in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York, NY, USA: ACM, pp. 785–794. doi: 10.1145/2939672.2939785.')
        st.caption('[3] Breiman, L. (2001) ‘Random forests’, Machine Learning. Springer, 45(1), pp. 5–32. doi: 10.1023/A:1010933404324/METRICS.')
        
       
    
    ########################################################################
if __name__=='__main__':
    main()
    
    
########################################################################    
with tab2:
    new_title = '<p style="text-align:left; color:red; font-size: 30px;"><strong>Development of prediction models<strong></p>'
    st.markdown(new_title, unsafe_allow_html=True)   
    st.write('To develop the models, 4 stages are followed.')
        
    with st.beta_expander('**Stage 1: Scoping & data collection**'):
        st.write('Real scale industrial data of the covered lagoon POME anerobic digestor used in this study were\
        obtained from four Malaysian plants over a period of 24 months (July 2019 to June 2021).')
        st.write('The plants include (i) Lepar Hilir Palm Oil Mill, (ii) Adela Palm Oil Mill, (iii) Keratong Estate Oil Palm Mill and (iv)\
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
        library under **"gaussian_process.GaussianProcessRegressor"**
        - XGBoost was directly imported from the [**xgboost**](https://xgboost.readthedocs.io/en/stable/) library
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

########################################################################    
with tab3:
    new_title = '<p style="text-align:left; color:red; font-size: 30px;"><strong>Environmental impact of biogas repurposing<strong></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    st.subheader('Circular palm oil industry')
    st.write('Utilising POME for biogas production presents a sustainable approach for\
    reducing GHG emissions while offering economic benefits. POME biogas can be utilized \
    to meet the plant’s energy requirement, making the industry self-sufficient. Subsequently, \
    POME biogas capture offers substantial environmental benefits as it facilitates a \
    circular economy via the reuse of process waste for energy generation. POMs with biogas \
    plants generate lower carbon footprint compared to conventional mills, achieving a 76% \
    carbon reduction [1]. The concept of a circular economy in the palm oil industry is demonstrated \
    in **Fig 1**. As POME biogas has a high heating value similar to that of a commercial fuel, \
    electricity can be produced by this by-product. The application of this **_waste to wealth_** \
    concept allows energy to be recirculated back into the process, producing a :blue[**circular**] loop.')
    
    image = Image.open('Circular.png')
    st.image(image, caption='Fig 1: Circular economy of a POM with a biogas plant.')
    
    st.subheader('Life Cycle Assessment (LCA)')
     
    st.write('For a more comprehensive environmental evaluation to assess \
    the impact of this process throughout the system boundary, a **life cycle assessment (LCA)** \
    of the AD covered lagoon unit was conducted on the base case (Lepar \
    Hilir Plant) **with bioelectricity production** and another case **without \
    bioelectricity production**.')
    st.write('This LCA comprises of four key phases: (i) goal and scope definition, \
    (ii) life cycle inventory (LCI), (iii) life cycle impact assessment (LCIA), and \
    (iv) interpretation.')

    st.info('This LCA adhered to the framework outlined in the :blue[**ISO 14040/ 14044**] series.\
    The :blue[**openLCA 1.11.0**] software for windows was used for the study, and the\
    LCIA was carried out using the :blue[**ReCiPe 2016 (World H) midpoint technique**]\
    with the :blue[**EcoInvent V3.8 database**].')
                
    with st.beta_expander('**Phase (i): Goal and scope definition**'):
        st.write('The system boundary of this study focused on ‘gate-to-gate’, which considered the \
        reception of POME from the process to the completion of its anaerobic treatment. Inputs to the \
        AD system were in the form of materials and energy, while outputs were in the form of effluent, \
        solid waste and biogas emissions. Two scenarios were considered in this LCA evaluation, the first one \
        being AD without biogas repurposing (**Fig 2**), and the second one being AD with biogas repurposing \
        such as electricity generation (**Fig 3**). This allowed the comparison of environmental impact of \
        biogas repurposing to the process to be made, where bio-energy generation should offset the \
        negative impacts of the process. A functional unit of 1m3 of POME influent was used for this study.')
        
        image = Image.open('Boundary_1.png')
        st.image(image, caption='Fig 2: System boundary of a covered lagoon AD in a POME plant without electricity generation.')
        
        image = Image.open('Boundary_2.png')
        st.image(image, caption='Fig 3: System boundary of a covered lagoon AD in a POME plant with electricity generation.')
        
        st.markdown("""

        """)
    
    
    with st.beta_expander('**Phase (ii): Life cycle inventory (LCI)**'):
        
        st.write('This LCI contains all essential information for the assessment of the base \
        case and alternate case to be carried out.')
        
        st.write('**INPUT**')
        LCI_input= pd.DataFrame({"Value":[1.000, 0.694, 1.872],\
                                 "Unit":['m3','m3','kWh']},
                                index=["POME to AD (f.u.)","Biogas recirculation","Electricity (pump)"])
        LCI_input
        
        st.write('**OUTPUT**')
        LCI_output= pd.DataFrame({"Value":[0.944,0.056,19.437,12.140,6.576,0.014],\
                                 "Unit":['m3','m3','m3','m3','m3','m3'],\
                                 "Mass":['',48.657, 19.380, 7.369, 10.975, 0.022],\
                                 "Unit Mass":['','kg','kg','kg','kg','kg']},
                                index=["Raw effluent","Sludge","Raw biogas",\
                                      "CH4","CO2","H2S"])
        LCI_output
        
        st.write('**OTHER**')
        LCI_other= pd.DataFrame({"Value":[67.362,14.273,26.853,72.536],\
                                 "Unit":['kg/m3','kg/m3','kg/m3','kWh']},
                                index=["COD POME in", "COD POME out", "COD sludge out",\
                                      "Electricity generated"])
        LCI_other

    
        st.markdown("""

        """)
    
    
    
    with st.beta_expander('**Phase (iii) : Life Cycle Impact Assessment (LCIA)**'):
        st.write('In this study, the [EcoInvent](https://ecoinvent.org/) database was employed. With over 18,000 \
        LCI datasets, this database encompasses sectors including agriculture, waste treatments, \
        and water supply. For this study, certain process specific information was not included in \
        the EcoInvent database, including POME biogas and sludge streams, SO$_{2}$ emission from AD \
        process and phosphate emission due to COD and TN in POME. Hence, manual calculation was \
        conducted to obtain the important environmental impacts as a result of those parameters.')
        st.write('To conduct these calculations, emission, equivalence, and methane correction factors \
        were first obtained. Most Malaysian POMs with biogas systems compute carbon emissions\
        according to the Intergovernmental Panel on Climate Change (IPCC) criteria for CDM\
        applications, which uses the 2014 CDM electricity baseline and the 2006 IPCC Guidelines\
        for National Greenhouse Gas Inventories default emission factor values [2,3]. Methane correction \
        factors were also reviewed on a scenario basis [4].')
        st.write('The [ReCiPe 2016](https://pre-sustainability.com/legacy/download/Report_ReCiPe_2017.pdf) \
        approach was used to perform LCIA calculations. Using the ReCiPe \
        2016 (World-H) midpoint technique, only midpoints were used to compute the effect categories. \
        18 distinct effect categories at the midpoint were examined using ReCiPe 2016. In this study, \
        focus was placed on 3 categories, namely: (i) global warming potential (GWP), (ii) terrestrial \
        acidification (TAP), and (iii) freshwater eutrophication. (FEP).')
    
    
    with st.beta_expander('**Phase (iv) : LCA interpretation**'):
        
        #Create tabs
        tab21, tab22, tab23 = st.tabs(["Global warming potential (GWP)", "Acidification potential (AP)", "Eutrophication potential (EP)"])
        
        with tab21:
            st.subheader('Global warming potential (GWP)')

            col1, col2 = st.columns([3,2])

            with col1:
                source = pd.DataFrame({"Emission from POME AD":[22.112, 22.112],\
                                       "Emission from POME sludge":[9.923, 9.923],\
                                       "Emission from recirculated biogas":[0, 0.516],\
                                       "Emission from POME effluent":[-14.667, -14.667],\
                                       "Emission from electricity generation":[0, -66.072],\
                                       "Other emissions":[-1.475, -1.475]},
                                      index=["Open lagoon", "Closed lagoon"])

                st.bar_chart(source, width=300,height=400,use_container_width=False)

            with col2:
                st.metric("in kg CO$_{2}$ eq/ m$^{3}$","GWP", delta= "+389%")
                st.caption('Up to 390% improvement in GWP when biogas is captured and repurposed.')

            st.write('When biogas in repurposed in the closed lagoon, the net GWP impact is positive, \
            which implies that the implementation of biogas repurposing shows potential in decreasing\
            GHG emissions to result in a net reduction of CO$_{2}$. The conversion of biogas into \
            eneergy results in avoided CO$_{2}$ emissions, which leads to negative GWP values.\
            Electricity generated by POME biogas will be utilised by the plant and sold to the national \
            grid of Malaysia via the Feed-in Tariff (FIT) system. Consequently, the significance of capturing\
            biogas from the POME treatment system cannot be overstated as it serves as a crucial means of \
            offsetting the adverse impacts of electricity consumption.')

            #############
        
        with tab22:
            
            st.subheader('Acidification potential (AP)')

            col1, col2 = st.columns([3,2])

            with col1:
                source = pd.DataFrame({"Emission of SO2 from AD":[0.004,0.004],\
                                       "AP from software":[-0.067, -0.270]},
                                      index=["Open lagoon", "Closed lagoon"])

                st.bar_chart(source, width=300,height=400,use_container_width=False)        

            with col2:
                st.metric("in kg SO$_{2}$ eq/ m$^{3}$","AP", delta= "+323%")
                st.caption('Up to 323% improvement in AP when biogas is captured and repurposed.')

            st.write('In a closed system, there is a significant decrease in AP impacts in contrast \
            to an open system. This is consistent with the findings of [Nasution _et. al._, (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0959652618306371?via%3Dihub) \
            thereby biogas repurposing systems have a 9.5% to 19% better AP impact performance \
            than that of an open lagoon [5]. The reason is because the closed system prevents the emission \
            of SO$_{2}$ into the atmosphere, which results from the H$_{2}$S present in the biogas.')

            #############
        
        with tab23: 
            st.subheader('Eutrophication potential (EP)')

            col1, col2 = st.columns([3,2])

            with col1:
                source = pd.DataFrame({"Emission of POME COD from AD":[1.103, 1.103],\
                                       "Emission of POME TN from AD":[0.389, 0.389],
                                       "EP from software":[-0.088, -0.115]},
                                      index=["Open lagoon", "Closed lagoon"])

                st.bar_chart(source, width=300,height=400,use_container_width=False)   

            with col2:
                st.metric("in kg P eq/ m$^{3}$","EP", delta= "+1.92%")
                st.caption('Up to 2% improvement in EP when biogas is captured and repurposed.')

            st.write('There is no significant difference in EP when it comes to biogas capture in \
            this system. The magnitude of EP is largely influenced by the total nitrogen and COD in \
            the POME, which will likely not be affected regardless of whether biogas has been captured.')

    st.markdown("""

    """)
    
    st.write('**References**')
    st.caption('[1] Lim, C. and K. Biswas, W. (2019) ‘Sustainability Implications of the Incorporation of a Biogas Trapping System into a Conventional Crude Palm Oil Supply Chain’, Sustainability, 11(3), p. 792. doi: 10.3390/su11030792.')
    st.caption('[2] Eggleston, H. S. et al. (2006) IPCC guidelines for national greenhouse gas Inventories. Hayama, Japan.')
    st.caption('[3] CDM (2014) ‘Study on grid connected electricity baselines in Malaysia.’, in Malaysia: Clean Development Mechanism.')    
    st.caption('[4] Doorn, M. et al. (2006) Chapter 6-wastewater treatment and discharge. IPCC guidelines for national greenhouse gas Inventories. Hayama, Japan.')  
    st.caption('[5] Nasution, M. A. et al. (2018) ‘Comparative environmental impact evaluation of palm oil mill effluent treatment using a life cycle assessment approach: A case study based on composting and a combination for biogas technologies in North Sumatera of Indonesia’, Journal of Cleaner Production, 184, pp. 1028–1040. doi: 10.1016/j.jclepro.2018.02.299.')
    
    ########################################################################
    with tab4:
    
        st.subheader('About the project')
        st.write('Malaysia is the second-largest global palm oil producer, accounting for 100 million tonnes of annual fresh fruit bunches (FFB) harvest. \
        During palm oil production, 90% of the total FFB forms biomass waste comprising of mainly empty fruit bunches (EFB), biomass sludge, and palm oil \
        mill effluent (POME). POME is typically produced during sterilisation (36% total POME) and clarification (60% total POME). \
        5—7 tonnes of water per tonne CPO is required, where roughly half of the amount of water converted to POME.  In tandem with the rise of CPO demand, \
        the volume of POME produced also sees a significant increase. POME has emerged as the most significant pollutant generated by palm oil mills (POMs) \
        for its high levels of organic matter, oil and grease (O&G) content and substantial quantities of heavy metals. Without prior treatment, POME cannot \
        be released to the environment, as it highly exceeds the discharge limits specified within the Environmental Quality Act (EQA) 1982 by the Department \
        of Environment Malaysia.')
        st.write('In Malaysia, over 50% of POME is commonly treated via conventional ponding system, owing to the available land spaces, cost-effectiveness, \
        and low manpower requirement. However, ponding systems are often deemed redundant as it requires long retention times of up to 200 days and \
        unnecessarily large areas. During the AD process, biogas is formed. POME biogas typically contains 50—75 vol% (CH$_{4}$), 25—45 vol% (CO$_{2}$), \
        2—7 vol% water vapor, < 1 vol% oxygen and traces of H$_{2}$S. Various studies on biogas production from POME AD treatment corroborated that 1 \
        tonne of raw POME produces approximately 8.72—12.4 kg CH$_{4}$, or 28m$^{3}$ biogas.')
        st.write('Under the clean development mechanism (CDM) of the Kyoto Protocol, projects that utilize biogas generated from AD as a form of renewable \
        energy can qualify for certified emission reduction (CER) credits. CDM also presents an opportunity for Malaysia to attract foreign investments for \
        the advancement of renewable energy projects. As such, the development of POME biogas recovery projects in Malaysia were emphasized since the \
        introduction of CDM. However, as of 2019, less than 20% (92 out of 453) of POMs had followed through with biogas activities.')
        st.write('Therefore, it is imperative to keep track of the amount of biogas and GHGs that will be producced via the POME AD process.\
        Past studies have adopted different modelling techniques to predict biogas production, including kinetic-based mathematical equations \
        such as biochemical methane potential (BMP) tests, and empirical models like the Gompertz and Transfert models. However, these kinetic-based \
        mathematical models are too complex to be practical as they have too many stoichiometric coefficients and parameters to accurately reflect the \
        biochemical properties of the process. Hence, it is necessary to develop a highly predictive yet simple model for biogas production estimation.\
        This gives rise to an advancement in the application of machine learning (ML) in system biology studies. ML models are more advantageous over \
        theoretical models since they are developed with a measured dataset, which comprises of input-output data pairs for a specific system which does \
        not require any kinetic relationships between the input and output variables. The application of suitable ML models in this study allow prediction \
        of process outputs to be more accurate. ')
        st.write('The transition to IR4.0 encourages the advancement in agricultural off-site monitoring technology in the field of Internet of Things \
        (IoT), which allows data to be transferred and stored over a network. On-site POME treatment monitoring is typically conducted weekly or monthly due \
        to its associated inconvenience. Physical samples are required to perform water quality tests on its BOD, COD, O & G and other parameters. There is \
        little development on the remote monitoring of the POME AD process on the effluent and biogas quality. The development of this website allows relevant POMs\
        to carry out accurate prediction of outputs remotely. In the future, there is a potential for real-time data to be directly inputted into the model. This \
        should allow the model to repeat the training process and update its fitting accordingly.')



        st.subheader('About us')
        
        tab31, tab32 = st.tabs(["Researchers", "Ackowledgements"])

        with tab31:

            col1,col2,col3 = st.columns([1,1,1])

            with col1:

                col4,col5,col6 = st.columns([1,6,1])

                with col4:
                    st.write("")

                with col5:
                    image = Image.open('QY.png')
                    st.image(image, width=150)
                    name = '<p style= "text-align:center; font-size: 20px;"><strong>Qian Yee Ong</strong></p>'
                    st.markdown(name, unsafe_allow_html=True)

                with col6:
                    st.write("")

                title = '<p style= "text-align:center; color:blue; font-size: 15px;">Masters in Chemical with Environmental Engineering</p>'
                st.markdown(title, unsafe_allow_html=True)

                st.caption(':e-mail:: slenderqianyeeong@gmail.com')

                h1 = '<p style= "font-size: 15px;">\
                 - Model development \
                <br> - Primary website author </p>'
                st.markdown(h1, unsafe_allow_html=True)


            with col2:
                col4,col5,col6 = st.columns([1,6,1])

                with col4:
                    st.write("")

                with col5:
                    image = Image.open('Amanda.png')
                    st.image(image, width=150)
                    name = '<p style= "text-align:center; font-size: 20px;"><strong>Xin Yun Kiew</strong></p>'
                    st.markdown(name, unsafe_allow_html=True)

                with col6:
                    st.write("")

                title = '<p style= "text-align:center; color:blue; font-size: 15px;">Masters in Chemical with Environmental Engineering</p>'
                st.markdown(title, unsafe_allow_html=True)

                st.caption(':e-mail:: amandakiewxy@gmail.com')

                h1 = '<p style= "font-size: 15px;">\
                - Data pre-processing \
                <br> - Life Cycle Assessment </p>'
                st.markdown(h1, unsafe_allow_html=True)


            with col3:
                col4,col5,col6 = st.columns([1,6,1])

                with col4:
                    st.write("")

                with col5:
                    image = Image.open('Joshua.png')
                    st.image(image, width=150)
                    name = '<p style= "text-align:center; font-size: 20px;"><strong>Joshua Liew Y.L.</strong></p>'
                    st.markdown(name, unsafe_allow_html=True)

                with col6:
                    st.write("")

                title = '<p style= "text-align:center; color:blue; font-size: 15px;">Masters in Chemical Engineering</p>'
                st.markdown(title, unsafe_allow_html=True)

                st.caption(':e-mail:: joshualiew10@gmail.com')

                h1 = '<p style= "font-size: 15px;">\
                - Website deployment \
                <br> - Website author </p>'
                st.markdown(h1, unsafe_allow_html=True)



            st.markdown("""

            """)

            
        with tab32:
            
            st.caption('We would like to express out gratitude towards out academic supervisors, \
            Dr Sara Kazemi Yazdi and Dr Chen Zhi Yuan for their guidance, support and encouragement \
            throughout the research process, which helped us to refine our research objectives, \
            methodology and results. We are especially grateful to Dr Chan Yi Jing, who provided us \
            with much literature and information relevant to this research. We are also grateful to \
            the University of Nottingham Malaysia for providing us with the software support required \
            to conduct this research. We would also like to show appreciation for the engineers and \
            managers of Lepar Hilir Palm Oil Mill, Adela Palm Oil Mill, Keratong Estate Oil Palm \
            Mill and Felda Lok Heng Palm Oil Mill for providing the dataset required for this study.')
