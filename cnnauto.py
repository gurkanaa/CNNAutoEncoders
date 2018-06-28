###CNN Autoencoder for Vibration Denoising
import os
import scipy.io
import numpy as np
from keras.layers import Input,Conv1D,MaxPooling1D,UpSampling1D,Flatten,Dense
from keras.models import Model
data_path='/Users/gurkanAydemir/Documents/akademik/PhD/python/CNNAutoencoder/CNNAutoEncoders/data/'
files=os.listdir(data_path)
record_id=['K001','K002','K003','K004','K005','K006']
denoised_data={}
raw_data={}
raw_train=[]
raw_test=[]
denoised_train=[]
denoised_train=[]

for i in files:
    if 'denoised' in i:
        dummy=scipy.io.loadmat(data_path+i)
        denoised_data.update({i[9:-4]:dummy['denoised_frames']})
    elif 'raw' in i:
        dummy=scipy.io.loadmat(data_path+i)
        raw_data.update({i[4:-4]:dummy['raw_frames']})

rec=record_id[0]
for i in raw_data.keys():
    if rec in i:
        if len(raw_test)==0:
            raw_test=raw_data[i]
            denoised_test=denoised_data[i]
        else:
            raw_test=np.concatenate((raw_test,raw_data[i]),axis=1)
            denoised_test=np.concatenate((denoised_test,denoised_data[i]),axis=1)
    else:
        if len(raw_train)==0:
            raw_train=raw_data[i]
            denoised_train=denoised_data[i]
        else:
            raw_train=np.concatenate((raw_train,raw_data[i]),axis=1)
            denoised_train=np.concatenate((denoised_train,denoised_data[i]),axis=1)

#CNN model

input_1=Input(shape=(2560,1))
x = Conv1D(64, 8, activation='relu', kernel_initializer='lecun_uniform',padding='same')(input_1)
x=MaxPooling1D(2)(x)
x = Conv1D(16, 8, activation='relu', kernel_initializer='lecun_uniform',padding='same')(x)
x=MaxPooling1D(4)(x)
x = Conv1D(16, 8, activation='relu', kernel_initializer='lecun_uniform',padding='same')(x)
x=UpSampling1D(4)(x)
x = Conv1D(64, 8, activation='relu', kernel_initializer='lecun_uniform',padding='same')(x)
x=UpSampling1D(2)(x)
x = Conv1D(1,8,padding='same')(x)

autoencoder = Model(input_1, x)

print(raw_test)
raw_train=np.reshape((raw_train),(np.size(raw_train,axis=1),np.size(raw_train,axis=0),1))
denoised_train=np.reshape((denoised_train),(np.size(denoised_train,axis=1),np.size(denoised_train,axis=0),1))
raw_test=np.reshape((raw_test),(np.size(raw_test,axis=1),np.size(raw_test,axis=0),1))
denoised_test=np.reshape((denoised_test),(np.size(denoised_test,axis=1),np.size(denoised_test,axis=0),1))
#compiler
print(np.size(raw_test[0,:,0]))
autoencoder.compile(optimizer='RMSprop',loss='mean_squared_error')
autoencoder.fit(raw_train,raw_train,epochs=15,batch_size=64,shuffle=True)

x_pred=autoencoder.predict(raw_test)
print(np.sum(abs(x_pred[:,0,0])))
print(np.sum(abs(raw_test[:,0,0])))
print(np.sum(abs(raw_test[:,0,0]-x_pred[:,0,0])))
