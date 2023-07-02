import numpy as np
import pickle
import streamlit as st  


model = pickle.load(open('modelRF.pkl','rb'))

def crc_prediction(input_data):
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)
    prediction = model.predict(id_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return 'Credit Tidak Bermasalah'
    else:
        return 'Kredit Bermasalah'
    
def main():
    
    st.title('CRC PREDICTIONS')
    
    Umur = st.text_input('Umur')
    Pendapatan = st.text_input('Pendapatan')
    KepemilikanRumah = st.text_input('KepemilikanRumah')
    LamaKerja = st.text_input('LamaKerja')
    TujuanPeminjaman = st.text_input('TujuanPeminjaman')
    TingkatanPinjaman = st.text_input('TingkatanPinjaman')
    JumlahPinjaman = st.text_input('JumlahPinjaman')
    SukuBunga = st.text_input('SukuBunga')
    JumlahHistoriPeminjaman = st.text_input('JumlahHistoriPeminjaman')
    
    diagnosis = ''
    
    if st.button('PREDICT'):
        try:
            diagnosis = crc_prediction([Umur,Pendapatan,KepemilikanRumah,LamaKerja,TujuanPeminjaman,TingkatanPinjaman,
                                    JumlahPinjaman,SukuBunga,JumlahHistoriPeminjaman])
        except ValueError:
            diagnosis = 'Invalid input(s)'

    st.success(f'Credit Status: {diagnosis}')
    
if __name__=='__main__':
    main()