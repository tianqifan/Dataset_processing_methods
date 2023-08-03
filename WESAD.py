'''
introduce for WESAD
website: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
Taking ECG as an example
'''

import numpy as np
import pandas as pd
import pickle
import os
from biosppy.signals import ecg,emg,eda,resp
from wfdb import processing

# define windows for splitting data
def windows(data,size):
    start=0
    while((start+size)<data.shape[0]):
        yield  int(start),int(start+size)
        start+=size
index=[2,3,4,5,6,7,8,9,10,11,13,14,15,16,17] # subjects
filepath='WESAD path'
record=['ECG','EMG','EDA','Resp']
name=record[0]
data_path='path for save Processed data'
size=7000 # 700Hz*10s
def data_pre(save path,filepath,new save path):
    # set new paths
    path=r'save path'
    print(os.path.exists(path))
    os.makedirs(r'save path/'+name+'/')
    os.makedirs(r'save path/'+name+'_split/')
    os.makedirs(r'save path/'+name+'/LABEL/')
    for i in index:
        print('start' + str(i) + 'subject')
        data = pickle.load(open(filepath + 'S' + str(i) + '/S' + str(i) + '.pkl', 'rb'), encoding='latin1')
        # get ECG
        data_ = data['signal']['chest'][name]
        label = data['label']
        if len(data_) == len(label):
            np.save('save path' + name + '/' + name + '/S' + str(i) + '_' + name + '.npy', data_)
            np.save('save path' + name + '/LABEL/S' + str(i) + '_LABEL.npy', label)
        else:
            print('error')
    for m in index:
        print('strart' + str(m) + 'subject')
        data = np.load('save path' + name + '/' + name + '/S' + str(m) + '_' + name + '.npy', allow_pickle=True)
        label = np.load('save path' + name + '/LABEL/S' + str(m) + '_LABEL.npy', allow_pickle=True)

        # label==1,2,3 
        data_1 = []
        data_2 = []
        data_3 = []

        if len(data) == len(label):
            print("correct！")
        for i in range(len(data)):

            if label[i] == 1:
                data_1.extend(data[i])
            if label[i] == 2:
                data_2.extend(data[i])
            if label[i] == 3:
                data_3.extend(data[i])

        data_1 = np.array(data_1).reshape(-1, 1)
        data_2 = np.array(data_2).reshape(-1, 1)
        data_3 = np.array(data_3).reshape(-1, 1)

        np.save("save path" + name + "/" + name + "_split/S" + str(m) + "_" + name + "1.npy", data_1)
        np.save("save path" + name + "/" + name + "_split/S" + str(m) + "_" + name + "2.npy", data_2)
        np.save("save path" + name + "/" + name + "_split/S" + str(m) + "_" + name + "3.npy", data_3)
    for j in range(1, 4):
        DATA = []
        LABEL = []
        for i in index:
            data = np.load(data_path + 'S' + str(i) + '_ECG' + str(j) + '.npy', allow_pickle=True)
            label = int(j)
            print(str(i) + ' subject ' + str(j) + 'class data shape：', data.shape, 'label is: ', j)
            print("Start filtering")
            data1 = data.squeeze()
            print("Shape before filtering：", data1.shape)
            # Default 3-45hzFIR bandpass filtering
            out = emg.emg(signal=data1, sampling_rate=700, show=False)
            filtered = out['filtered']
            print("filtered data shape:", filtered.shape)
            filtered = filtered.reshape(-1, 1)
            print("filtered data save shape:", filtered.shape)

            for (start, end) in windows(data, size):

                if (len(data[start:end] == size)):
                    if (start == 0):
                        segments = data[start:end]

                        segments = np.vstack([segments, data[start:end]])
                        labels = np.array(label)
                        labels = np.append(labels, np.array(label))
                    else:
                        segments = np.vstack([segments, data[start:end]])

                        labels = np.append(labels, np.array(label))
            DATA.extend(segments)
            LABEL.extend(labels)
        DATA = np.array(DATA).reshape(-1, size)
        LABEL = np.array(LABEL).reshape(-1, )
        np.save("new save path" + name + "/data" + str(j) + ".npy", DATA)
        np.save("new save path" + name + "/label" + str(j) + ".npy", LABEL)
        print("saved：", j, "class")
        print("DATA shape", DATA.shape, "LABEL shape", LABEL.shape)

        # cat 4 class data
    data = []
    label = []
    for i in range(1, 4):
        ecg_data = np.load("new save path" + name + "/data" + str(i) + ".npy", allow_pickle=True)
        ecg_label = np.load("new save path" + name + "/label" + str(i) + ".npy", allow_pickle=True)
        data.extend(ecg_data)
        label.extend(ecg_label)
    data = np.array(data)
    label = np.array(label)
    print("data shape", data.shape, "label shape", label.shape)
    np.save("new save path" + name + "/data7000.npy", data)
    np.save("new save path" + name + "/label10s_3classnpy", label)
    return data
def down_sample(data,targrt_freq,source_freq=256):
    data_down = []
    num=len(data)
    assert targrt_freq<=source_freq
    for i in range(len(data)):
        out,_=processing.resample_sig((data[i]),fs=source_freq,fs_target=targrt_freq)
        #out=out.reshape(-1,1)
        #print(out.shape)
        data_down.append(out)
    data_down=np.array(data_down)
    return data_down
if __name__=='__main__':
    data=data_pre(save path,filepath,new save path)
    data=dawn_sample(data,targrt_freq,source_freq=700)
