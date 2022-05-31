import streamlit as st
import pandas as pd

import utils as utils
import function as fn
from sklearn.model_selection import train_test_split

if 'cstate' not in st.session_state:
    st.session_state['cstate'] = 1
uploaded_file = st.file_uploader(label = "Upload source data", type=".csv")

if uploaded_file:
    st.write("### Data Preview")

    st.write("##### Data Sample")
    df = utils.load_data(uploaded_file=uploaded_file)
    n_rows = st.slider(label="Select number of rows to display", min_value=5, max_value=25, step=5)
    if(n_rows):
        st.write(utils.get_n_rows(df, n_rows))

    st.write("##### Data Infometrics")
    st.write(utils.get_info(df).astype(str))

    st.write("##### Null Infometrics")
    st.write(utils.get_null_info(df))

    st.write("#### Drop columns")
    drop_cols = st.multiselect(label = "Choose unnecessary columns to drop", options=df.columns)
    drop_button = st.button(label="Drop")
    if drop_button or int(st.session_state['cstate']) > 1:
        st.session_state['cstate'] = max(2, int(st.session_state['cstate']))
        if drop_cols:
            for col in drop_cols:
                utils.drop_column(df, col)
        st.write("##### How would you like to deal with NULL values?")
        methods = ["mean", "mode", "median", "max", "min", "drop"]
        method = st.selectbox(label="Select NULL value fixing method.", options=methods)
        null_method_button = st.button(label="Apply")
        print("before null", int(st.session_state['cstate']))
        if null_method_button or int(st.session_state['cstate']) > 2:
            st.session_state['cstate'] = max(3, int(st.session_state['cstate']))
            if method:
                st.write("NULL replacement method:", method)
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        utils.handle_missing(df, col, method)
            print("after null", int(st.session_state['cstate']))
            st.write("#### Choose Target")
            target = st.selectbox(label="Select the column name from dataframe.", options=df.columns)
            target_button = st.button(label="Fix Target")
            if target_button or int(st.session_state['cstate']) > 3:
                st.session_state['cstate'] = max(4, int(st.session_state['cstate']))
                print("after target", int(st.session_state['cstate']))
                if target:
                    st.write("Target is", target)

                numerical_features = fn.cat_features(df, 'numerical')
                categorical_features = fn.cat_features(df, 'categorical')
                st.write("#### Plot Categorical Features")
                nf = st.selectbox(label="Select the column name from dataframe.", key='cat', options=categorical_features)
                if nf:
                    st.pyplot(fig=fn.categorical_plot(df, nf, target))
                st.write("#### Plot Numerical Features")
                cf = st.selectbox(label="Select the column name from dataframe.", key='num', options=numerical_features)
                if cf:
                    st.pyplot(fig=fn.numerical_plot(df, cf, target))
                df = utils.encode(df)
                # df = fn.feature_scaling(df, target)
                X_train, X_test, y_train, y_test = fn.split_dataset(df, target)
                st.write("#### Choose the type of model")
                model_type = st.selectbox(label="Select type of model to apply", key='num', options=['Regression', 'Classification'])
                reg = ['Linear Regression', 'Linear SVR', 'Decision Tree Regressor', 'K Neighbour Regressor']
                cla = ['Logistic Regressor', 'Naive Bayes', 'K Neighbours Classifier', 'Decision Tree Classifier', 'Linear SVC']
                train_sel = None;
                if(model_type == 'Regression'):
                    train_sel = st.selectbox(label="Select model to apply", key='num', options=reg)
                else:
                    train_sel = st.selectbox(label="Select model to apply", key='num', options=cla)
                train_button = st.button(label="Train")
                st.write("Please note that this may take a long time depending on the dataset.")
                if train_button or int(st.session_state['cstate']) > 4:
                    st.session_state['cstate'] = max(5, int(st.session_state['cstate']))
                    ret = fn.model_training(X_train,y_train,X_test,y_test, train_sel)
                    for r in ret:
                        st.write(r)

    

    
