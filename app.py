import streamlit as st
from matplotlib.gridspec import GridSpec
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score
import xgboost as xgb 
from PIL import Image
from io import StringIO
import base64
import lasio
from sklearn.ensemble import IsolationForest
import pickle
plt.style.use('seaborn-darkgrid')

########################################################################

st.markdown('## **Predict Compressional sonic (DTC) from other well logs using different machine learning algorithms: XGboost and Random Forest**') 

image = Image.open(r'geospatial.PNG')

st.image(image)

st.markdown('### DEPTH and CALI, NPHI, RDEP, RHOB, SP, GR logs are used as features to target DTC log data. ###')
st.markdown("Adjust hyper-parameters to see the model performance")
st.sidebar.header('Model selection & Hyperparameter tuning')

#########################################################################

path = r'training_data.csv'
path2 = r'test_data.csv'

df1 = pd.read_csv(path).iloc[:, 1:]
test_data = pd.read_csv(path2)
sample_rate = st.sidebar.slider('Sample every point (select larger values for faster run):', 1, 60, 30)
df = df1.iloc[::sample_rate,:]
st.sidebar.write('Data Samples:', df.shape[0])
############################################################# data prepration for keep out set
st.sidebar.markdown('### Blind well')
blind_name = st.sidebar.selectbox('', ("15/9-13", "15/9-15"))

def get_well(well_name):
	blind = None

	if well_name== '15/9-13':
		blind = test_data[test_data['WELL'] == '15/9-13']

	elif well_name == '15/9-15':
		blind = test_data[test_data['WELL'] == '15/9-15']

	return blind

blind = get_well(blind_name)
#st.sidebar.write('Blind well:', blind_name)

training_data = df

X = training_data[['CALI','DEPTH','NPHI','RDEP','RHOB','SP','GR']].values
y = training_data['DTC'].values

X_blind = blind[['CALI','DEPTH','NPHI','RDEP','RHOB','SP','GR']].values
y_blind = blind['DTC'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
############################################################## Standardization

st.sidebar.markdown('### Machine Learning Algorithm')
regression_name = st.sidebar.selectbox('', ("XGboost", "Random Forest"))


def add_parameter_ui(reg_name):
	params = dict()

	if reg_name == 'XGboost':
		n_estimators = st.sidebar.slider("n_estimators", 500, 1000, 800)
		learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.1, 0.05)
		max_depth = st.sidebar.slider("max_depth", 5, 8, 6)
		colsample_bytree = st.sidebar.slider("colsample_bytree", 0.7, 0.9, 0.8)
		subsample = st.sidebar.slider("subsample", 0.7, 0.9, 0.8)
	
		params['n_estimators'] = n_estimators
		params['learning_rate'] = learning_rate
		params['max_depth'] = max_depth
		params['colsample_bytree'] = colsample_bytree
		params['subsample'] = subsample

	elif reg_name =='Random Forest':
		max_depth = st.sidebar.slider("max_depth", 40, 60, 50)
		min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 3, 2)
		min_samples_split = st.sidebar.slider("min_samples_split", 8, 10, 9)
		n_estimators = st.sidebar.slider("n_estimators", 200,500,300)
		
		params['max_depth'] = max_depth
		params['min_samples_leaf'] = min_samples_leaf
		params['min_samples_split'] = min_samples_split
		params['n_estimators'] = n_estimators

	return params

params = add_parameter_ui(regression_name)

def get_regression(reg_name, params):
	reg = None

	if reg_name == 'XGboost':
		reg = xgb.sklearn.XGBRegressor(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
									   max_depth =params['max_depth'], colsample_bytree = params['colsample_bytree'],
									   subsample = params['subsample'])

	elif reg_name == 'Random Forest':
		reg = RandomForestRegressor(max_depth = params['max_depth'], min_samples_leaf = params['min_samples_leaf'], 
								   min_samples_split = params['min_samples_split'], n_estimators = params['n_estimators'])

	return reg


reg = get_regression(regression_name, params)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
R2 = r2_score(y_test, y_pred)

########################################################### examine models with blind data
y_pred_b = reg.predict(X_blind)
R2_b = r2_score(y_blind, y_pred_b)
blind['Pred_DTC'] =  y_pred_b
##################################################################################
Depth_b = blind.DEPTH.values
GR_b = blind.GR.values
RHOB_b = blind.RHOB.values
NPHI_b = blind.NPHI.values
DTC_b = blind.DTC.values
SP_b = blind.SP.values
CALI_b = blind.CALI.values
RDEP_b = blind.RDEP.values

fig = plt.figure(figsize=(9,12))

gs1 = GridSpec(1, 1)
ax1 = fig.add_subplot(gs1[0])

ax1.plot(DTC_b, Depth_b, color='mediumblue', alpha=.9, lw=0.6, ls= '-', label='Actual')
ax1.plot(y_pred_b, Depth_b, color='orange', alpha=.9, lw=0.7, ls= '-', label='Predicted')
ax1.set_title(blind.WELL.iloc[0], fontsize=8, color='mediumblue', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, color='0.7', dashes=(5,2,1,2) )
ax1.set_facecolor('#F8F8FF')
ax1.tick_params(axis="x", labelsize=6)
ax1.set_ylabel('Depth(m)', fontsize=9)
ax1.set_xlabel('DTC', fontsize=9)
ax1.tick_params(axis="y", labelsize=7)
ax1.legend(loc='center left')

st.pyplot(fig)

st.write(f'Regressor Model = {regression_name}')
st.write(f'R2 Score (model) =', np.round(R2,2))
st.write(f'*R2 Score (Blind) =', np.round(R2_b,2))
st.markdown('*Blind data are the two wells which were involved in the modeling process; used for prediction evaluation purposes.')


def outlier_removal(df, ind=-1):

	log_header = ['DEPTH','WELL','CALI','DEPT','NPHI','RDEP','RHOB','SP','GR']
	# major parameter percentage of outlier present with parameter contamination, nu : 3 %
	outliers_frac = 0.03
	# define outlier/anomaly detection methods to be compared
	anomaly_algorithm = [('Isolation Forest', IsolationForest(n_estimators=100, max_samples='auto',
						 contamination=outliers_frac, random_state=42))]

# detect anomaly for each features
	for i, item in enumerate(log_header[2:]):

		anomaly = anomaly_algorithm[0][1].fit_predict(df[[item]])
		df = df[anomaly==1]
		return df

@st.cache
def load_data(uploaded_file):
	if uploaded_file is not None:
		try:
			bytes_data = uploaded_file.read()
			str_io = StringIO(bytes_data.decode('Windows-1252'))
			las_file = lasio.read(str_io)
			well_data = las_file.df()
			well_data.reset_index(drop=True,inplace=True)
			well_data['DEPTH'] = las_file.index
			well_data['WELL'] = las_file.well.WELL.value
			well_data = well_data[['DEPTH','WELL','CALI','NPHI','RDEP','RHOB','SP','GR']]
			well_data['RDEP'] = np.log10(well_data['RDEP'])
			well_train_std = outlier_removal(well_data, ind=-1)

			well_train_std.dropna(inplace=True)

			
		except UnicodeDecodeError as e:
			st.error(f"error loading log.las: {e}")

	else:
		well_train_std = None

	return well_train_std

def get_table_download_link(df):
	csv = df.to_csv().encode()
	b64 = base64.b64encode(csv).decode()
	href = f'<a href="data:file/csv;base64,{b64}" download="Pred_DTC.csv" target="_blank">Download csv file</a>'
	return href

uploadedfile = st.file_uploader('Upload your las file')
if uploadedfile is not None:
	blind = load_data(uploadedfile)
	st.dataframe(blind.iloc[0:10,:])
	st.write('Shape of dataset:', blind.shape)

	if st.button('Predict DTC'):
		X_blind = blind[['CALI','DEPTH','NPHI','RDEP','RHOB','SP','GR']].values

		model = pickle.load(open('model.pkl','rb'))
		y_pred_b = model.predict(X_blind)
		blind['Pred_DTC'] =  y_pred_b
		st.dataframe(blind.iloc[0:10,:])
		
		##################################################################################
		Depth_b = blind.DEPTH.values
		fig = plt.figure(figsize=(9,12))

		gs1 = GridSpec(1, 1)
		ax1 = fig.add_subplot(gs1[0])

		ax1.plot(y_pred_b, Depth_b, color='orange', alpha=.9, lw=0.7, ls= '-', label='Predicted DTC')
		ax1.set_title(blind.WELL.iloc[0], fontsize=8, color='mediumblue', fontweight='bold')
		ax1.invert_yaxis()
		ax1.grid(True, color='0.7', dashes=(5,2,1,2) )
		ax1.set_facecolor('#F8F8FF')
		ax1.tick_params(axis="x", labelsize=6)
		ax1.set_ylabel('Depth(m)', fontsize=9)
		ax1.set_xlabel('DTC', fontsize=9)
		ax1.tick_params(axis="y", labelsize=7)
		ax1.legend(loc='center left')

		st.pyplot(fig)
		st.markdown(get_table_download_link(blind), unsafe_allow_html=True)

## Author ##
with st.beta_expander('Author'):
	st.write('Daniel Dura')