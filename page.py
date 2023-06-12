import numpy as np
import pickle
import streamlit as st  

model = pickle.load(open('modelCb.pkl','rb'))