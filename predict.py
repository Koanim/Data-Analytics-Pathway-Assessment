import streamlit as st
import joblib
import os
import pandas as pd


# Configure the Streamlit page
st.set_page_config(
    page_title='Predict Page',
    page_icon='🔍',
    layout='wide'
)


# Cache resources for efficiency
@st.cache_resource()
def load_pipeline(model_path):
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None


@st.cache_resource()
def load_encoder(encoder_path):
    try:
        encoder = joblib.load(encoder_path)
        return encoder
    except FileNotFoundError:
        st.error(f"Encoder file not found: {encoder_path}")
        return None


# Load and select the model
def select_model():
    model_options = {
        'XGBoost': './models/XGB_pipeline.joblib',
        'Gradient Boosting': './models/GB_pipeline.joblib',
        'Random Forest': './models/RF_pipeline.joblib'
    }

    col1, _ = st.columns(2)
    with col1:
        selected_model = st.selectbox('Select Model', options=list(model_options.keys()), key='selected_model')

    # Clear previous prediction if the model has changed
    if 'previous_model' in st.session_state:
        if st.session_state['previous_model'] != selected_model:
            st.session_state['prediction'] = None
            st.session_state['probability'] = None

    # Save the current model selection
    st.session_state['previous_model'] = selected_model

    model_path = model_options[selected_model]
    pipeline = load_pipeline(model_path)
    encoder = load_encoder('./models/encoder.joblib')

    if pipeline is None or encoder is None:
        st.stop()  

    return pipeline, encoder



def initialize_session_state(keys_with_defaults):
    for key, default in keys_with_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


initialize_session_state({
    'age': 30, 'age_group': 'Adult', 'job': 'unknown', 'marital': 'single',
    'education': 'unknown', 'default': 'no', 'balance': 1000, 'housing': 'no',
    'loan': 'no', 'contact': 'unknown', 'day': 15, 'month': 'jan',
    'duration': 100, 'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown',
    'prediction': None, 'probability': None
})


# Function to make predictions
def make_prediction(pipeline, encoder):
    inputs = {
        'age': st.session_state['age'],
        'age_group': st.session_state['age_group'],
        'job': st.session_state['job'],
        'marital': st.session_state['marital'],
        'education': st.session_state['education'],
        'default': st.session_state['default'],
        'balance': st.session_state['balance'],
        'housing': st.session_state['housing'],
        'loan': st.session_state['loan'],
        'contact': st.session_state['contact'],
        'day': st.session_state['day'],
        'month': st.session_state['month'],
        'duration': st.session_state['duration'],
        'campaign': st.session_state['campaign'],
        'pdays': st.session_state['pdays'],
        'previous': st.session_state['previous'],
        'poutcome': st.session_state['poutcome']
    }
    
    data = pd.DataFrame([inputs])
    
    prediction = pipeline.predict(data)[0]
    probability = pipeline.predict_proba(data)[0]
    predicted_label = encoder.inverse_transform([int(prediction)])[0]

    st.session_state['prediction'] = predicted_label
    st.session_state['probability'] = probability

    # Log prediction history
    data['prediction'] = predicted_label
    data['probability'] = probability.max()  
    data['model_used'] = st.session_state['selected_model']

    history_file = 'Data/history.csv'
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    data.to_csv(history_file, mode='a', header=not os.path.exists(history_file), index=False)

    return predicted_label, probability


# Input form for predictions
def display_form():
    pipeline, encoder = select_model()

    with st.form('input_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('### Bank Customer Records')
            st.number_input('Customer Age', key='age', min_value=18, max_value=100, value=st.session_state['age'])
            st.selectbox('Customer Age Group', options=['Youth', 'Young Adult', 'Adult', 'Senior'], key='age_group')
            st.selectbox('Customer Profession', options=[
                "admin.", "unknown", "unemployed", "management", "housemaid",
                "entrepreneur", "student", "blue-collar", "self-employed",
                "retired", "technician", "services"
            ], key='job')
            st.selectbox('Customer\'s Marital Status', options=["married", "divorced", "single"], key='marital')
            st.selectbox('Customer\'s Education Status', options=["unknown", "secondary", "primary", "tertiary"], key='education')
            st.selectbox('Does the Customer have credit in Default?', options=["yes", "no"], key='default')
            st.selectbox('Does the Customer have housing loan?', options=["yes", "no"], key='housing')
            st.selectbox('Does the Customer have personal loan?', options=["yes", "no"], key='loan')
            st.number_input('Customer Account Balance', key='balance', min_value=0, value=st.session_state['balance'])
        
        with col2:
            st.markdown('### Current and Previous Campaign Contacts')
            st.selectbox('Customer\'s Preferred Contact Mode', options=["unknown", "telephone", "cellular"], key='contact')
            st.number_input('Day of Last Contact', key='day', min_value=1, max_value=31, value=st.session_state['day'])
            st.selectbox('Month of Last Contact', options=[
                'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
                'mar', 'apr', 'sep'
            ], key='month')
            st.number_input('Duration of Last Contact (in seconds)', key='duration', min_value=0, value=st.session_state['duration'])
            st.number_input('Number of Contacts During Campaign', key='campaign', min_value=1, value=st.session_state['campaign'])
            st.number_input('Number of Days Passed After Last Contact', key='pdays', min_value=-1, value=st.session_state['pdays'])
            st.number_input('Number of Customer contacts during previous campaign', key='previous', min_value=0, value=st.session_state['previous'])
            st.selectbox('Outcome of the Previous Campaign', options=['unknown', 'failure', 'success'], key='poutcome')
        st.form_submit_button('Submit', on_click=make_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder))


# Display results
def display_results():
    final_prediction = st.session_state['prediction']
    if not final_prediction:
        st.write('### Prediction will appear here after submission.')
    else:
        col1, col2 = st.columns(2)
        with col1:
            if final_prediction == "Yes":
                st.success(f"### Subscribed? :green[Yes] - Customer is likely to subscribe.")
            else:
                st.error(f"### Subscribed? :red[No] - Customer is not likely to subscribe.")
        with col2:
            prob = st.session_state['probability']
            if final_prediction == "Yes":
                st.write(f"#### Probability: :green[{round(prob[1] * 100, 2)}%]")
            else:
                st.write(f"#### Probability: :red[{round(prob[0] * 100, 2)}%]")


# Main app structure
st.header(':rainbow-background[Will Customer Subscribe to Term Deposit?]')
display_form()
display_results()


def display_historic_predictions():
    st.subheader(":violet[Displaying historic predictions]")
    csv_path = './Data/history.csv'
    csv_exists = os.path.isfile(csv_path)

    if csv_exists:
        history = pd.read_csv(csv_path)
        
        st.dataframe(history)
    #return history

#st.header(':rainbow-background[Historic Predictions]') 
display_historic_predictions()