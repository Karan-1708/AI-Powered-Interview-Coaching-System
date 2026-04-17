import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# set_page_config must be the first Streamlit call — keep it here before any src imports
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.ui.main import main

main()
