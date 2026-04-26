import streamlit as st
import joblib
from src.preprocess import clean_text

st.set_page_config(page_title='Spam Classifier', page_icon='📧')
st.title('📧 Spam Email Classifier')
st.markdown('Paste any email or SMS text below to check if it is spam.')

# Load model (cached for performance)
@st.cache_resource
def load_model():
    return joblib.load('models/classifier.pkl')

model = load_model()

user_input = st.text_area('Email / SMS text', height=150,
    placeholder='e.g. Congratulations! You have won a £1000 prize...')

if st.button('Classify') and user_input.strip():
    cleaned = clean_text(user_input)
    prediction = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]

    if prediction == 1:
        st.error(f'🚨 SPAM detected — {proba[1]*100:.1f}% confidence')
    else:
        st.success(f'✅ Looks safe (Ham) — {proba[0]*100:.1f}% confidence')

    # Show top contributing words
    st.subheader('Top words in this message')
    words = cleaned.split()
    st.write(', '.join(words[:10]) if words else 'No significant words found')

# Example messages sidebar
st.sidebar.header('Try an example')
examples = {
    'Spam example': 'WINNER!! You have been selected to receive a $1000 cash prize. Call 0800-FREE now!',
    'Ham example': 'Hey, are we still meeting for lunch tomorrow at 1pm?'
}
for label, text in examples.items():
    if st.sidebar.button(label):
        st.session_state['text'] = text