import streamlit as st
import joblib
import os
import pandas as pd
from threading import Lock


st.set_page_config(
    page_title='Predict Page',
    page_icon='🔍',
    layout='wide'
)


write_lock = Lock()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'Data')


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

# Function to select model
def select_model():
    model_options = {
        'XGBoost': os.path.join(MODEL_DIR, 'XGB_pipeline.joblib'),
        'Gradient Boosting': os.path.join(MODEL_DIR, 'GB_pipeline.joblib'),
        'Random Forest': os.path.join(MODEL_DIR, 'RF_pipeline.joblib')
    }

    # Dropdown to select model
    selected_model = st.selectbox(
        'Select Model',
        options=list(model_options.keys()),
        key='selected_model'
    )

    # Clear prediction if the model changes
    if 'previous_model' in st.session_state:
        if st.session_state['previous_model'] != selected_model:
            st.session_state['prediction'] = None
            st.session_state['probability'] = None

    st.session_state['previous_model'] = selected_model
    model_path = model_options[selected_model]

    pipeline = load_pipeline(model_path)
    encoder = load_encoder(os.path.join(MODEL_DIR, 'encoder.joblib'))

    if pipeline is None or encoder is None:
        st.stop()

    return pipeline, encoder

# Initialize session state
def initialize_session_state(keys_with_defaults):
    for key, default in keys_with_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

initialize_session_state({
    'age': 18, 'age_group': 'Adult', 'job': 'unknown', 'marital': 'single',
    'education': 'unknown', 'default': 'no', 'balance': 1000, 'housing': 'no',
    'loan': 'no', 'contact': 'unknown', 'day': 15, 'month': 'jan',
    'duration': 100, 'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown',
    'prediction': None, 'probability': None
})

# Function to make predictions
def make_prediction(pipeline, encoder):
    inputs = {key: st.session_state[key] for key in [
        'age', 'age_group', 'job', 'marital', 'education', 'default', 
        'balance', 'housing', 'loan', 'contact', 'day', 'month', 
        'duration', 'campaign', 'pdays', 'previous', 'poutcome'
    ]}

    if st.session_state['duration'] <= 0:
        st.warning("Duration must be greater than 0.")
        return None, None

    data = pd.DataFrame([inputs])

    try:
        # Ensure column alignment
        data = data[pipeline.feature_names_in_]
    except AttributeError:
        st.error("Pipeline feature alignment issue. Ensure feature names match.")
        return None, None

    prediction = pipeline.predict(data)[0]
    probability = pipeline.predict_proba(data)[0]

    try:
        predicted_label = encoder.inverse_transform([int(prediction)])[0]
    except Exception as e:
        st.error(f"Error in encoding prediction: {e}")
        predicted_label = "Unknown"

    st.session_state['prediction'] = predicted_label
    st.session_state['probability'] = probability

    # Log prediction history
    data['prediction'] = predicted_label
    data['probability'] = probability.max()
    data['model_used'] = st.session_state['selected_model']

    history_file = os.path.join(DATA_DIR, 'history.csv')
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with write_lock:
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
            st.selectbox('Customer Age Group', ['Youth', 'Young Adult', 'Adult', 'Senior'], key='age_group')
            st.selectbox('Customer Profession', [
                "admin.", "unknown", "unemployed", "management", "housemaid",
                "entrepreneur", "student", "blue-collar", "self-employed",
                "retired", "technician", "services"
            ], key='job')
            st.selectbox('Customer\'s Marital Status', ["married", "divorced", "single"], key='marital')
            st.selectbox('Customer\'s Education Status', ["unknown", "secondary", "primary", "tertiary"], key='education')
            st.selectbox('Does the Customer have credit in Default?', ["yes", "no"], key='default')
            st.selectbox('Does the Customer have housing loan?', ["yes", "no"], key='housing')
            st.selectbox('Does the Customer have personal loan?', ["yes", "no"], key='loan')
            st.number_input('Customer Account Balance', key='balance', min_value=0, value=st.session_state['balance'])

        with col2:
            st.markdown('### Current and Previous Campaign Contacts')
            st.selectbox('Customer\'s Preferred Contact Mode', ["unknown", "telephone", "cellular"], key='contact')
            st.number_input('Day of Last Contact', key='day', min_value=1, max_value=31, value=st.session_state['day'])
            st.selectbox('Month of Last Contact', [
                'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
                'mar', 'apr', 'sep'
            ], key='month')
            st.number_input('Duration of Last Contact (in seconds)', key='duration', min_value=0, value=st.session_state['duration'])
            st.number_input('Number of Contacts During Campaign', key='campaign', min_value=1, value=st.session_state['campaign'])
            st.number_input('Number of Days Passed After Last Contact', key='pdays', min_value=-1, value=st.session_state['pdays'])
            st.number_input('Number of Customer contacts during previous campaign', key='previous', min_value=0, value=st.session_state['previous'])
            st.selectbox('Outcome of the Previous Campaign', ['unknown', 'failure', 'success'], key='poutcome')

        st.form_submit_button('Submit', on_click=make_prediction, kwargs={'pipeline': pipeline, 'encoder': encoder})

# Display results
def display_results():
    final_prediction = st.session_state['prediction']
    if not final_prediction:
        st.write('### Prediction will appear here after submission.')
    else:
        col1, col2 = st.columns(2)
        with col1:
            if final_prediction == "Yes":
                st.success(f"### Subscribed? ✅ Yes - Likely to subscribe.")
            else:
                st.error(f"### Subscribed? ❌ No - Not likely to subscribe.")
        with col2:
            prob = st.session_state['probability']
            if final_prediction == "Yes":
                st.write(f"#### Probability: ✅ {round(prob[1] * 100, 2)}%")
            else:
                st.write(f"#### Probability: ❌ {round(prob[0] * 100, 2)}%")

# Display historic predictions
def display_historic_predictions():
    st.subheader("Historic Predictions")
    csv_path = os.path.join(DATA_DIR, 'history.csv')

    if os.path.isfile(csv_path):
        history = pd.read_csv(csv_path)
        st.dataframe(history)
    else:
        st.info("No predictions have been logged yet.")

# Main app structure
st.header("Will the Customer Subscribe to a Term Deposit?")
display_form()
display_results()
display_historic_predictions()
