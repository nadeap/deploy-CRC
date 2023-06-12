import numpy as np
import pickle
import streamlit as st  

model = pickle.load(open('modelCb.pkl','rb'))

def crc_prediction(input_data):
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)

    prediction = model.predict(id_reshaped)
    print(prediction)

    if(prediction[0]==0):
        print("Credit Status: Tidak Bermasalah")
    else:
        print("Credit Status: Bermasalah")

def main():
    
    st.title('CRC PREDICTIONS')
    
    Umur = st.text_input('Umur')
    Pendapatan = st.text_input('Pendapatan')
    KepemilikanRumah = st.text_input('Number of Steps')
    
    diagnosis = ''
    
    if st.button('PREDICT'):
        diagnosis = stresslevel_prediction([Humidity, Temperature, Step_count])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()