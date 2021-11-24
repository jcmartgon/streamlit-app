import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.markdown(
    """
        <style>
            .main {
            background-color: green;
            }
        </style>
    """,
    unsafe_allow_html=True
)


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()


@st.cache
def get_data():
    taxi_data = pd.read_csv('./data/yellow_tripdata_2021-01.csv')
    return taxi_data


with header:
    st.title('Welcome to my streamlit app!')
    st.text('In this project I look into the transactions of taxis in NYC')

with dataset:
    st.header('New york city taxi dataset:')
    st.text('I found this dataset at https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page')

    taxi_data = get_data()
    st.write(taxi_data.head())

    st.subheader('Pick up location ID distribution on the NYC dataset')
    pulocation_dist = pd.DataFrame(
        taxi_data['PULocationID'].value_counts().head(50))
    st.bar_chart(pulocation_dist)

with features:
    st.header('List example:')

    st.markdown('* **Testing** lists and markdown')
    st.markdown('* **Another** list item!')

with model_training:
    st.header('Model:')
    st.text('Here the hyperparameters of the model can be changed')

    sel_col, disp_col = st.beta_columns(2)

    max_depth = sel_col.slider(
        'Select max depth of the model', min_value=10, max_value=100, value=10, step=10)

    n_estimators = sel_col.selectbox('Select number of trees: ', options=[
                                     100, 200, 300, 'No limit'], index=0)

    sel_col.text('Here is a list of features: ')
    sel_col.write(taxi_data.columns)
    input_feature = sel_col.text_input('Select a feature: ', 'PULocationID')

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(
            max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared score:')
    disp_col.write(r2_score(y, prediction))
