#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import time
import datetime


# In[6]:


from FinalPrediction_new import result_list
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable
import time
from pathlib import Path

from Dataset_predict import Dataset
from score import Score



print(result_list[1])
result_hr = str(result_list[1])#[33:-1]
print(result_hr)



st.title("The app of Group 10")

#The uni logo
st.markdown("""Welcome to use our tenth group of electricity price forecasting models :sunglasses:
![](https://www.prozell-cluster.de/wp-content/uploads/2017/09/tu-muenchen.png) 
""")

#The names of G10 members
if st.checkbox('Show the group member'):
    group_list1 = st.text('Donghao Song, Yingxian Li, Hongfei Yan, \nYikai Kang, Ziwei Cheng, Xiaoxuan Cai, \nHuiwen Zheng, Yuan Huang' )


st.header('YOU CAN USE THE APP FROM HERE ON')

#The brief introduction of the web app
expander = st.expander("Click to see Readme...")
expander.write("At the top of this web app, there is the logo of uni and the names of the members of our group. You can expand to view the names of the members by clicking the box on the left of Show the member.The current time in the middle will automatically read the current time when the app is executed. After our group used four different model training data and made a comparison, we finally chose to use MLP to predict the data one day later, and use LSTM to predict the data one hour later and one week later. You can click the Click to predict button at the bottom to obtain the predicted value corresponding to the current time in one hour, one day, and one week.")

#TIME display 
st.markdown("""The current local time is:""")
st.text(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

st.write(':exclamation: Please click the following button to get all the results')

#按钮出结果
result_hr = str(result_list[0])#[33:-1]
result_day= str(result_list[1])
result_we = str(result_list[2])
if st.button('Click to predict'):
     st.write((datetime.datetime.now()+datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),'\n',result_hr)
     st.write((datetime.datetime.now()+datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),'\n',result_day)
     st.write((datetime.datetime.now()+datetime.timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),'\n',result_we)
     st.balloons()






