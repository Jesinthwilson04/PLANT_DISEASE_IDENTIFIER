import streamlit as st
page_bg_img="""
<style>
[data-testif="stAppViewContaoner"]
{
background-color: #ffffff;
opacity: 0.8;
background-image: radial-gradient(circle at center center, #afb3f8, #ffffff), repeating-radial-gradient(circle at center center, #afb3f8, #afb3f8, 40px, transparent 80px, transparent 40px);
background-blend-mode: multiply;
    
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)
st.title("It's summer")