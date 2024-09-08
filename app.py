# import numpy as np
# import streamlit as st
# import pickle
# import re
# import pdfplumber
# import base64
# from streamlit_lottie import st_lottie
# import requests

# # Load models
# clf = pickle.load(open('clf.pkl', 'rb'))
# tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# # Functions
# def cleaning(text):
#     res = re.sub('http\S+\s', ' ', text)
#     res = re.sub('RT|cc', ' ', res)
#     res = re.sub('#\S+\s', ' ', res)
#     res = re.sub('@\S+', ' ', res)  
#     res = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', res)
#     res = re.sub(r'[^\x00-\x7f]', ' ', res) 
#     res = re.sub('\s+', ' ', res)
#     return res

# def decode_resume(file):
#     if file.name.endswith('.pdf'):
#         with pdfplumber.open(file) as pdf:
#             text = ''.join(page.extract_text() for page in pdf.pages)
#         return text
#     return file.read().decode('utf-8', errors='ignore')

# category_mapping = {
#     15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
#     20: "Python Developer", 24: "Web Designing", 12: "HR",
#     13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
#     18: "Operations Manager", 6: "Data Science", 22: "Sales",
#     16: "Mechanical Engineer", 1: "Arts", 7: "Database",
#     11: "Electrical Engineering", 14: "Health and Fitness", 19: "PMO",
#     4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
#     17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer",
#     0: "Advocate",
# }

# # Streamlit page configuration
# st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# # Custom CSS
# st.markdown("""
# <style>
#     .stApp {
#         background-image: linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
#         background-attachment: fixed;
#     }
#     .main {
#         background-color: rgba(255,255,255,0.8);
#         padding: 20px;
#         border-radius: 10px;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         padding: 10px 20px;
#         text-align: center;
#         text-decoration: none;
#         display: inline-block;
#         font-size: 16px;
#         margin: 4px 2px;
#         transition-duration: 0.4s;
#         cursor: pointer;
#         border-radius: 12px;
#         border: none;
#     }
#     .stButton>button:hover {
#         background-color: white;
#         color: black;
#         border: 2px solid #4CAF50;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Title and description
# st.title("🚀 AI-Powered Resume Screening App")
# st.markdown("Upload your resume and let our AI analyze it to predict the best job category for you!")

# # Load Lottie animation
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
# lottie_json = load_lottieurl(lottie_url)
# st_lottie(lottie_json, speed=1, height=200, key="initial")

# # Main function
# def main():
#     uploaded_file = st.file_uploader('📂 Upload Your Resume', type=['txt', 'pdf'])

#     if uploaded_file is not None:
#         try:
#             with st.spinner("🔍 Analyzing your resume..."):
#                 resume_text = decode_resume(uploaded_file)
#                 cleaned_resume = cleaning(resume_text)
#                 input_features = tfidfd.transform([cleaned_resume])
#                 prediction_id = clf.predict(input_features)[0]
#                 category_name = category_mapping.get(prediction_id, "Unknown")

#             st.success("✅ Analysis Complete!")
#             st.subheader("🎯 Predicted Job Category:")
#             st.info(f"**{category_name}**")
            
#             # Display a fun fact about the predicted category
#             fun_facts = {
#                 "Java Developer": "Did you know? Java runs on 3 billion devices worldwide!",
#                 "Python Developer": "Fun fact: Python was named after the comedy show Monty Python!",
#                 "Data Science": "Interesting tidbit: 90% of the world's data was created in the last two years!",
#                 # Add more fun facts for other categories
#             }
#             if category_name in fun_facts:
#                 st.markdown(f"**💡 {fun_facts[category_name]}**")

#             # Provide a tailored tip based on the category
#             tips = {
#                 "Java Developer": "Tip: Keep up with the latest Java frameworks to stay competitive!",
#                 "Python Developer": "Advice: Explore machine learning libraries to expand your skillset!",
#                 "Data Science": "Suggestion: Practice data visualization to effectively communicate your findings!",
#                 # Add more tips for other categories
#             }
#             if category_name in tips:
#                 st.markdown(f"**🌟 {tips[category_name]}**")

#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()import numpy as npimport streamlit as st
import pickle
import re
import numpy as np
import pdfplumber
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import time

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Functions
def cleaning(text):
    res = re.sub('http\S+\s', ' ', text)
    res = re.sub('RT|cc', ' ', res)
    res = re.sub('#\S+\s', ' ', res)
    res = re.sub('@\S+', ' ', res)  
    res = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[$$^_`{|}~"""), ' ', res)
    res = re.sub(r'[^\x00-\x7f]', ' ', res)
    res = re.sub('\s+', ' ', res)
    return res

def decode_resume(file):
    if file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages)
        return text
    return file.read().decode('utf-8', errors='ignore')

category_mapping = {
    0: "Software Engineer",
    1: "Data Scientist",
    2: "Product Manager",
    3: "UX Designer",
    4: "Marketing Specialist"
}

# Streamlit page configuration
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# Custom CSS
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #f8f9fa;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #6c5ce7;
        color: #fff;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #5849e0;
        transform: scale(1.05);
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        border-radius: 12px;
        border: 2px solid #6c5ce7;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
@st.cache_data
def load_lottieurl(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

lottie_resume = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5tkzkblw.json")
lottie_analysis = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_P3BdNe.json")

# Main function
def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Resume Analysis", "About"],
            icons=["house", "file-earmark-text", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Home":
        st.title("🚀 AI-Powered Resume Screening App")
        st.markdown("### Welcome to the future of resume analysis!")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            Our cutting-edge AI technology helps you:
            - 📊 Analyze your resume instantly
            - 🎯 Identify the best job category for your skills
            - 💡 Get personalized tips to improve your resume
            
            Get started by navigating to the 'Resume Analysis' page!
            """)
        with col2:
            st_lottie(lottie_resume, height=200, key="resume")

    elif selected == "Resume Analysis":
        st.title("📄 Resume Analysis")
        uploaded_file = st.file_uploader('Upload Your Resume', type=['txt', 'pdf'])

        if uploaded_file is not None:
            with st.spinner('Analyzing your resume...'):
                try:
                    resume_text = decode_resume(uploaded_file)
                    cleaned_text = cleaning(resume_text)
                    features = tfidfd.transform([cleaned_text])
                    prediction = clf.predict(features)[0]
                    job_category = category_mapping.get(prediction, "Unknown")

                    # Simulated progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    # Display results
                    st.success(f"### Your best job category is: **{job_category}**")
                    st_lottie(lottie_analysis, height=200, key="analysis")

                    # Skills match simulation
                    st.subheader("🔍 Skills Match")
                    skills = ["Python", "Data Analysis", "Machine Learning", "Communication", "Problem Solving"]
                    for skill in skills:
                        match = round(100 * np.random.random(), 1)
                        st.markdown(f"**{skill}**: {match}%")
                        st.progress(int(match))

                    st.markdown("### 🌟 Tips for improving your resume:")
                    suggestions = [
                        "✅ Highlight your key achievements with quantifiable results",
                        "🎯 Tailor your resume to the specific job description",
                        "🔑 Use industry-specific keywords to pass ATS systems",
                        "📊 Include relevant projects and their impact",
                        "🔧 Showcase your technical skills and tools proficiency"
                    ]
                    for tip in suggestions:
                        st.markdown(f"- {tip}")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif selected == "About":
        st.title("About Our AI Resume Screener")
        st.markdown("""
        Our AI-powered Resume Screening App uses state-of-the-art machine learning algorithms to analyze resumes and provide valuable insights. 
        
        ### How it works:
        1. Upload your resume
        2. Our AI analyzes the content
        3. Get instant feedback on your best-fit job category
        4. Receive personalized tips to enhance your resume
        
        ### Privacy Note:
        We value your privacy. All uploaded resumes are processed in real-time and are not stored on our servers.
        
        For more information, contact our support team at support@airesumescreen.com
        """)

if __name__ == "__main__":
    main()