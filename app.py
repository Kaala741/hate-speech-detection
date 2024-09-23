import streamlit as st
from preprocessing import preprocess_text
from model import load_model,predict
from streamlit_lottie import st_lottie
import requests


def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return new_func()
    return r.json()

def load_lottiefile(filepath:str):
    with open(filepath,encoding="utf8") as f:
        data =json.load(f)
    return data

# Load the model and tokenizer
model_path = 'bert_classifier.h5'
model, tokenizer = load_model(model_path)

lottie_coding = load_lottieurl("https://lottie.host/05d8286a-86bb-483e-a845-5715aca4241e/hrUyVR89oh.json")
lottie_coding2 = load_lottieurl("https://lottie.host/5735c629-79f5-4312-b4e4-268709e0eaab/D4OjpsayFs.json")

# Streamlit UI
st.set_page_config(layout="wide",page_title="Hate-speech-Detection",page_icon=":computer:")
st.markdown("<h1 style='text-align:center;color:white;'>Welcome To Our Website !</h1>", unsafe_allow_html=True)
#---Main Section---
with st.container():
    st.header('Automated Hate Speech Detection')
    input_text = st.text_area('Enter text to analyze', '')
    if st.button('Analyze'):
         preprocessed_text = preprocess_text(input_text)
         classification = predict(preprocessed_text, model, tokenizer)
         st.write("Classification:", classification)
st.write("---")
#---Header Section---
with st.container():
    left_column,right_column=st.columns(2)
    with left_column:
        st.header("ABOUT")
        st.write(
    """
    This website employs an automated system to detect and manage harmful content that incites hatred, violence, or discrimination against individuals or groups based on characteristics such as race, ethnicity, gender, sexual orientation, or religion. By using machine learning algorithms and natural language processing, the system analyzes user-generated comments and posts on the Reddit platform to identify and flag potentially offensive or harmful language
    """
        )
    with right_column:
        st_lottie(lottie_coding, height=400, key="coding")
# --- FORM SECTION ---
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    contact_form = """
    <form action="https://formsubmit.co/seelammanikanta777@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("form.css")
    
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st_lottie(lottie_coding2, height=300,key="anime")
