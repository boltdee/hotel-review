import streamlit as st
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import altair as alt
import zipfile
import time

# Page title
st.set_page_config(page_title='Sentiment Analysis and ML Model Building', page_icon='🤖')
st.title('🤖 Sentiment Analysis and ML Model Building')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to perform sentiment analysis and build a machine learning (ML) model in an end-to-end workflow. This includes data upload, data pre-processing, sentiment analysis, ML model building, and post-model analysis.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and 1. Select a data set, 2. Adjust the model parameters by adjusting the various slider widgets, and 3. View the sentiment analysis results and model performance.')

    st.markdown('**Under the hood**')
    st.markdown('Libraries used:')
    st.code('''- Pandas for data wrangling
- VADER for sentiment analysis
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
    ''', language='markdown')

# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)
      
    # Download example data
    @st.cache_data
    def convert_df(input_df):
        return input_df.to_csv(index=False).encode('utf-8')
    
    example_csv = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    csv = convert_df(example_csv)
    st.download_button(
        label="Download example CSV",
        data=csv,
        file_name='delaney_solubility_with_descriptors.csv',
        mime='text/csv',
    )

    # Select example data
    st.markdown('**1.2. Use example data**')
    example_data = st.toggle('Load example data')
    if example_data:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.subheader('2.1. Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('2.2. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\n', ' ', text)  # Remove new lines
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text

# Initiate the model building process
if uploaded_file or example_data:
    with st.status("Running ...", expanded=True) as status:
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
        
        # Perform sentiment analysis
        if 'text_column' in df.columns:
            df['cleaned_text'] = df['text_column'].apply(clean_text)  # Replace 'text_column' with the actual column name
            analyzer = SentimentIntensityAnalyzer()
            df['sentiment'] = df['cleaned_text'].apply(analyzer.polarity_scores)
            df['compound'] = df['sentiment'].apply(lambda score_dict: score_dict['compound'])
            df['sentiment_label'] = df['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')
        
            # Display the sentiment distribution
            st.write("### Sentiment Distribution")
            fig, ax = plt.subplots()
            df['compound'].hist(bins=20, ax=ax)
            ax.set_title('Sentiment Distribution')
            ax.set_xlabel('Sentiment Score')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

            # Display the average sentiment score
            average_sentiment = df['compound'].mean()
            st.write("### Average Sentiment Score")
            st.write(average_sentiment)

            # Prepare data for model training
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        else:
            X = df.iloc[:,:-1]
            y = df.iloc[:,-1]
            
        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)
    
        st.write("Model training ...")
        time.sleep(sleep_time)

        if parameter_max_features == 'all':
            parameter_max_features = None
            parameter_max_features_metric = X.shape[1]
        
        rf = RandomForestClassifier(
                n_estimators=parameter_n_estimators,
                max_features=parameter_max_features,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                random_state=parameter_random_state,
                criterion=parameter_criterion,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score)
        rf.fit(X_train, y_train)
        
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
            
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        rf_results = pd.DataFrame([['Random forest', train_accuracy, test_accuracy]], columns=['Method', 'Training Accuracy', 'Test Accuracy'])
        
    status.update(label="Status", state="complete", expanded=False)

    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
    
    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=
