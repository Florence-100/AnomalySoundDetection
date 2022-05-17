# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:49:41 2022

@author: NG ZHI QING
"""

import streamlit as st
import torch
import torchaudio
import numpy as np

anomaly_threshold = 161666980

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
          
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(640, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 8)
        )
          
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 640),
            torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = AE()
model.load_state_dict(torch.load('baseline_AE'))

def convertSoundFiletoSpectrogram(file):
    waveform, sample_rate = torchaudio.load(file)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=80000)
    mel_spectrogram = transform(waveform)
    return mel_spectrogram

#write streamlit display 
title = 'Simple Streamlit Application'
st.markdown("<h1 style='font-size:65px; color:#4682B4;'>{}</h1>".format(title), unsafe_allow_html=True)

#file uploader 
uploaded_file = st.file_uploader('Please upload a sound file')

anomalyResult = 'Sound is anomalous'
normalResult = 'Sound is normal'

if uploaded_file is not None: 
    if st.button('Classify'):
        #convert sound file to melspectrogram and flatten
        melspecgram = convertSoundFiletoSpectrogram(uploaded_file)
        flatten_melspecgram = torch.flatten(melspecgram)
        
        #reconstruct sound file
        reconstructed = model(flatten_melspecgram)
        for index in range(0, len(reconstructed)):
            
            true = flatten_melspecgram.numpy()
            predicted = reconstructed[index].detach().numpy()
            anomaly_score = np.mean(np.square(true - predicted))
        
        if anomaly_score > anomaly_threshold:
            st.markdown("<p style='font-size:22px; color:#4682B4;'>{}</p>".format(anomalyResult), unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size:22px; color:#4682B4;'>{}</p>".format(normalResult), unsafe_allow_html=True)
        st.markdown("<p style='font-size:22px; color:#4682B4;'>{}</p>".format(f'Anomaly score is {anomaly_score}'), unsafe_allow_html=True)
        
