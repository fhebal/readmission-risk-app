import streamlit as st
import pandas as pd
from pandas import CategoricalDtype
from lifelines.datasets import load_rossi
from lifelines import WeibullAFTFitter, CoxPHFitter
from utils import plotter, read_config
from joblib import dump, load
import json
import matplotlib.pyplot as plt
import numpy as np
import math

# SETUP
#st.set_page_config(layout="wide")
df = load_rossi()
model = WeibullAFTFitter().fit(df, 'week', 'arrest')
cph = CoxPHFitter()
cph.fit(df, 'week', 'arrest')


DURATION = 'week'
EVENT = 'arrest'

def home_vary_survival(df, DURATION, EVENT, option):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    #plt.ylim(0, 1.01)
    #plt.xlim(0, 60)
    times = np.arange(0, 120)
    if option == "Baseline":
        kmf = KaplanMeierFitter().fit(df[DURATION], df[EVENT])
        kmf.survival_function_.plot(ax=ax)
    else:
        min_value = min(df[option].values)
        max_value = max(df[option].values)+1
        interval = math.ceil((max_value - min_value)/4)
        value_range = range(min_value, max_value, interval)
        wft = WeibullAFTFitter().fit(df, DURATION, EVENT, ancillary=True, timeline=times)
        wft.plot_partial_effects_on_outcome(option, value_range, cmap='coolwarm', ax=ax)
    st.pyplot(plt)

# DESCRIPTION TEXT

def home_survival_sidebar():
    st.sidebar.title("Readmission Risk Model")
    st.sidebar.write('''
            Survival analysis can be used to define the probability that
            an event like readmission will occur over a given a sufficient
            period of time. Reading from the graph on the right, we can
            see that at 50 days, about 75% of our member population has been
            readmitted for inpatient care''')
    st.sidebar.write('''
            Select a patient attribute from the dropdown menu below to
            explore its impact on the Baseline Survival chart''')

    # KAPLAN MEIER CURVES
    st.write("## Baseline Survival")
    drop_cols = ['week', 'arrest', 'age', 'prio']
    select_cols = ['Marital status', 'Financial status', 'Gender', 'Chronic condition', 'Insurance Status']
    select_dict = {
            'Marital status': 'mar',
            'Financial status': 'fin',
            'Gender': 'race',
            'Chronic condtion': 'wexp',
            'Insurance status': 'paro'
            }
    option = st.sidebar.selectbox('', select_cols)

    home_vary_survival(df, DURATION, EVENT, select_dict[option])


home_survival_sidebar()



# INDIVIDAL PREDICTIONS
preds = """
st.title("Individual Prediction")
st.write('''
        see that at 50 days, about 75% of our member population remains
        enrolled in a plan.''')

slider_1 = st.slider('days since last visit', 0, 100)
slider_8 = st.slider('previous hospitalizations', 0, 10)
slider_3 = st.selectbox('insurance status', ["Private", "Medicaid/Medicare"])
slider_4 = st.selectbox('gender', ['male', 'female'])
slider_5 = st.selectbox('chronic condition', ["Yes", "No"])
slider_6 = st.selectbox('marital status', ["Married", "Single"])
slider_7 = st.selectbox('immune status', ["Normal", "Immunocompromized"])
slider_2 = st.selectbox('income status', ['High income', 'Low income'])


slider_2 = slider_2 == "High income"
slider_3 = slider_3 == "Medicaid/Medicare"
slider_4 = slider_4 == "Married"
slider_5 = slider_5 == "Yes"
slider_6 = slider_6 == "Married"
slider_7 = slider_7 == "Normal"

input_vector = [[
        slider_1, slider_2, slider_3, slider_4, slider_5, slider_6, slider_7, slider_8
        ]]
cols = [x for x in df.columns if x not in [EVENT]]
prediction_output = model.predict_median(pd.DataFrame(input_vector, columns=cols))
st.write("## Weeks until readmission:", str(int(prediction_output.values[0])))

col1 = st.beta_columns(1)

"""


#plt = plotter(df, option, DURATION, EVENT, CategoricalDtype)
#KM_plot = st.pyplot(plt)

