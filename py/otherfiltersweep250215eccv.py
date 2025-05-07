from __future__ import print_function
from asyncio import current_task
from cmath import polar
from operator import mod
from sqlite3 import TimestampFromTicks
from time import time
# from keras import callbacks, models
import numpy as np
# from numpy.core.numeric import load
# np.random.seed(1337)
import os

from tensorflow.keras import optimizers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow
from tensorflow import keras
from tensorflow.keras import callbacks, models
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense,Embedding,Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import os,sys,glob
import pandas as pd

from sklearn.model_selection import train_test_split
from custom_dataloader.dataloader import DataGenerator

from qkeras import *
from qkeras.utils import model_save_quantized_weights, load_qmodel

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import scipy.io as sio
from allrulefilters import *
from sklearn import metrics
# training params
learning_rate = 0.0005
batch_size = 100#1024#128
patchsize = 7
csvinputlen = patchsize * patchsize
middle = int(csvinputlen / 2)


# hidden = int(sys.argv[1])
# resize = int(sys.argv[2])
# epochs = int(sys.argv[3])
# csvfilepath = sys.argv[4]

hidden = 50#20
resize = 7
epochs = 5
trainfilepath = 'D://qmlpfpys-s//2xTrainingDataDND21train//'
testfilepath = 'D://qmlpfpys-s//2xTrainingDataDND21test//'


# networkinputlen = resize * resize

tau = 64#128#100000 # 300ms if use exp decay for preprocessing

global prefix

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    classes = ['Not Fall', 'Fall']
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='beginend', ha='beginend')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

#LossHistory, keep loss and acc
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))
 
    def loss_plot(self, prefix, loss_type, e0loss, e0acc, e0valloss, e0valacc):

        iters = range(len(self.losses[loss_type])+1)
        np.save(prefix + 'loss.npy', np.array([e0loss] + self.losses[loss_type]))
        np.save(prefix + 'acc.npy', np.array([e0acc] + self.accuracy[loss_type]))

        plt.figure()
        # acc
        plt.plot(iters, [e0acc] + self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, [e0loss] + self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, [e0valacc] + self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, [e0valloss] + self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        
        plt.savefig(prefix+'training.pdf')
        plt.clf()


import math
def preprocessingresizefortest(absTS,polarity, resize, targetEventTS, targetEventP):
    # print(features.shape)
    # print(features)
    # print(allfeatures.shape, targetEventTS.shape, targetEventP.shape)
    # absTS = allfeatures[:,:csvinputlen]
    # polarity = allfeatures[:,csvinputlen:]

    

    features = np.array(absTS)#.transpose()
    polarity = np.array(polarity)
    
    featuresdiff = np.abs([features[i,:] - targetEventTS[i] for i in range(len(features))])

    featuresdiff = (featuresdiff / 1000)
    featuresNormed = (tau - featuresdiff) * 1.0 / tau
    featuresNormed = np.clip(featuresNormed, 0, 1)
    # featuresNormed[featuresNormed > 0] = 1

    # featuresNormed = featuresdiff < tau
    # featuresNormed = featuresNormed.astype(np.int32)

    # crop
    features = featuresNormed.reshape(featuresNormed.shape[0], patchsize, patchsize)
    # beginend = int((patchsize - resize) / 2)
    # cropend = patchsize - beginend
    
    # polarity 
    center = int(patchsize/2)
    # features = features[:,beginend:cropend, beginend:cropend]
    # features = features.reshape(features.shape[0],resize * resize)
    
    polarityweight = np.array([polarity[i,:] * targetEventP[i] for i in range(len(features))])
    polarityweight = np.clip(polarityweight, 0, 1)
    print(np.unique(polarityweight))
    
    channelP = polarityweight.reshape(polarity.shape[0],patchsize, patchsize)
    channelP[:,center,center] = 0 # 
    # beginend = int((patchsize - resize) / 2)
    # cropend = patchsize - beginend
    # channelP = polarity[:,beginend:cropend, beginend:cropend]
    # channelP = channelP.reshape(channelP.shape[0], resize * resize)
    
    # channelP[features==0] = 0 # set the polarity to be 0 if the event is too old, which means the ts features are 0
    # channelP[:,int(resize*resize/2)] = 0 # ensure the beginend location has the classified event's polarity
    
    # stweight = np.multiply(features,channelP)
    # # stweight = np.multiply(stweight,weightmatrix)
    # stweight = np.sum(stweight, axis=1)
    # # features2 = np.hstack((features,channelP))
    # # print(features2.shape)
    # # return features2
    # # 只返回age 通道，减少模型参数量
    # return stweight
    return features,channelP


def preprocessing(features, targetEventTS):
    middle = int(patchsize * patchsize / 2)
    features = features.transpose()
    # normalization
    featuresdiff = features - targetEventTS
    featuresNormed = (tau - np.abs(featuresdiff)) * 1.0 / tau
    featuresNormed = np.clip(featuresNormed, 0, 1)
    # featuresNormed = np.exp(-np.abs(featuresdiff)/tau)

    featuresNormed = featuresNormed.transpose()
    return featuresNormed

import math
def initializetsandpolmap(noiseRateHz, lastTimestampUs):
    # Random random = new Random()
    if True:
        for row in range(timestampImage.shape[0]):
            for col in range(timestampImage.shape[1]):
                p = np.random.random()
                t = -noiseRateHz * math.log(1 - p)
                tUs = (int) (1000000 * t)
                timestampImage[row][col] = lastTimestampUs - tUs
        
        
        for row in range(lastPolMap.shape[0]):
            for col in range(lastPolMap.shape[1]):
                b = np.random.random()
                # arrayRow[i] = b ? 1 : -1
                if b>0.5:
                    lastPolMap[row][col] = 1
                else:
                    lastPolMap[row][col] = -1

        lastPolMap[timestampImage < 0] = 0
        timestampImage[timestampImage < 0] = 0

            
        
train = {}
val = {}



sx = 346
sy = 260
# patchsize = 7
timestampImage = np.zeros([sx,sy])
lastPolMap = np.zeros([sx,sy])

subsampleBy = 0
def getageandpolstringTI25(eventarray):
    if True:
        if True:
            if True:
                if True:
                    batchagechannel = np.zeros([eventarray.shape[0], patchsize * patchsize])
                    batchpolchannel = np.zeros([eventarray.shape[0], patchsize * patchsize])
                    # batchlabels = []
                    batchcount = 0
                    for event in eventarray:
                        # batchlabels.append(event[0]) # singalflag
                        ts = event[1] # event.timestamp
                        # type = event.getPolarity() == PolarityEvent.Polarity.Off ? -1 : 1
                        
                        x = (event[3] >> subsampleBy)
                        y = (event[2] >> subsampleBy)
                        type = event[4]
                        
                        radius = int((patchsize - 1) / 2)
                        if ((x < 0) or (x > sx) or (y < 0) or (y > sy)):
                            continue
                        
                        import time

                        absT= np.zeros([patchsize*patchsize,])#""
                        pols= np.zeros([patchsize*patchsize,])#""
                        indz = 0
                        stime1 = time.time()
                        xleftbound = max(0,x-radius)
                        xrightbound = min(sx,x+radius)
                        yleftbound = max(0,y-radius)
                        yrightbound = min(sy,y+radius)
                        # print(x,y,xleftbound,xrightbound,yleftbound,yrightbound)
                        croppedTI = timestampImage[xleftbound:xrightbound+1,yleftbound:yrightbound+1]
                        
                        # flattencroppedTI = np.reshape(croppedTI,order='F')
                        flattencroppedTI = croppedTI.flatten(order='F')
                        # print(timestampImage.shape, croppedTI.shape,flattencroppedTI.shape)

                        croppedpol = lastPolMap[xleftbound:xrightbound+1,yleftbound:yrightbound+1]
                        
                        # flattencroppedpol = np.reshape(croppedpol,order='F')
                        flattencroppedpol = croppedpol.flatten(order='F')
                        # print(lastPolMap.shape,croppedpol.shape,flattencroppedpol.shape)

                        startind = (yleftbound-(y-radius)) * radius + (xleftbound-(x-radius))
                        leng = flattencroppedpol.shape[0]
                        # print(flattencroppedpol.shape)
                        absT[startind:startind + leng,] = flattencroppedTI
                        pols[startind:startind + leng,] = flattencroppedpol
                        
                        etime1 = time.time()
                        
                        batchagechannel[batchcount] = absT
                        batchpolchannel[batchcount] = pols
                        batchcount += 1
                        timestampImage[x][y] = ts
                        lastPolMap[x][y] = type
                        

    return batchagechannel,batchpolchannel

# from tensorflow.keras.utils import to_categorical                         
def mygenerator(files):
    print('start generator')
    # while 1:
    if True:
        # print('loop generator')
        sumbatches = 0
        # random.shuffle(files)
        print(files)
        for file_ in files:
            try:
     
                df = pd.DataFrame(np.load(file_))
                df.fillna(0)
                
                zero = len(df[df.iloc[:,0] == 0])
                # random.shuffle(df.values)
                # print('read file', file_, zero, len(df)-zero)

                # batches = int(np.ceil(len(df)/batch_size))
                # sumbatches += batches
                # for i in range(0, batches):
                    # e_data = df[i*batch_size:min(len(df),i*batch_size+batch_size)].values
                if True:
                    e_data = df.values
                    if patchsize >= resize:
                        # sample = {'y': m_data[2], 'x': preprocessingresize(m_data[3:3+csvinputlen*1], resize, m_data[1], m_data[0])} # crop the TI patch according to the given size
                        y = e_data[:,0]
                        
                        # 通过参数3选择干净事件或者噪声事件
                        
                        # print(y.shape)
                        agechannel,polchannel = getageandpolstringTI25(e_data) 
                        
                        
                        # output (num, patchsize, patchsize)
                        agefeature,polfeature = preprocessingresizefortest(agechannel,polchannel, resize, e_data[:,1], e_data[:,4])
                        
                        

                        # indices = np.where(y == 1) 
                        # x1 = x[indices]
                        # np.save('2hzsignalpminus1output',x1)

                        # indices = np.where(y == 0)
                        # x0 = x[indices]
                        # np.save('2hznoisepminus1output',x0)

                    else:
                        # sample = {'y': m_data[:,4], 'x': preprocessing(m_data[:,5:5+csvinputlen*1], m_data[:,3])}
                        y = e_data[:,0]
           
                    return agefeature,polfeature,y
            except EOFError:
                print('error' + file_)
        # print(sumbatches)   



def splittraintest(csvdir, splitratio):
    
    allFiles = glob.glob(os.path.join(csvdir,'*TI25*.csv'))
    if len(allFiles) > 0:
        np_array_list = []
        for file_ in allFiles:
            print(file_)
            df = pd.read_csv(file_,usecols=[0] + [i for i in range(3,5+csvinputlen*1)], header=0) # might change if the collecting code change
            np_array_list.append(df.values)

        
        return pd.DataFrame(np.vstack(np_array_list[:int(splitratio*len(np_array_list))])), pd.DataFrame(np.vstack(np_array_list[int(splitratio*len(np_array_list)):]))


import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt

def plot_roc_curve(y_true,y_score,prefix):
    fpr,tpr,threshold = roc_curve(y_true,y_score,pos_label=1)
    auc = roc_auc_score(y_true,y_score)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('roc curve' + str(auc))
    plt.plot(fpr,tpr,color='b',linewidth=1)
    plt.plot([0,1],[0,1],'r--')
    plt.savefig(prefix + '_roccurve.pdf')
    plt.clf()



def getacc(y_true,initpredictions):
    y_pred = (initpredictions > 0.5).astype(int)
    y_true = np.reshape(y_true, [-1])
    y_pred = np.reshape(y_pred, [-1])
        

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


mtype = sys.argv[3]
# s = int(sys.argv[4])
# T = int(sys.argv[5])
# N = int(sys.argv[6])

def trainFunction(app, hz, mtype):
    
    # for bits in bitsarr:
    # if True:
    file_ = testfiles[0]
    print(file_)
    
    label_column = -1
    if 'npy' in file_:
        df = pd.DataFrame(np.load(file_))
        df.fillna(0)
        
        zero = len(df[df.iloc[:,0] == 0])
        label_column = 0
        tcol = 1
        xcol = 2
        ycol = 3
        pcol = 4
    elif 'txt' in file_:
        df = pd.read_csv(file_, skiprows=1, delimiter=' ', dtype={
            'column1': np.int64, 'column2': np.int16, 'column3': np.int16, 'column4': np.int8})
        label_column = 4
       
        tcol = 0
        xcol = 1
        ycol = 2
        pcol = 3
                    
    # df = pd.DataFrame(np.load(file_))
    # df.fillna(0)
    
    # zero = len(df[df.iloc[:,0] == 0])
    # random.shuffle(df.values)
    # print('read file', file_, zero, len(df)-zero)

    # batches = int(np.ceil(len(df)/batch_size))
    # sumbatches += batches
    # for i in range(0, batches):
        # e_data = df[i*batch_size:min(len(df),i*batch_size+batch_size)].values
    # if True:
    data = df.values
    # print(data)
    # data = data[:3000]
    print(data.shape)
    # print(data[:10])
    ytrue = data[:,label_column]
    print(np.max(data[:,xcol]))
    print(np.max(data[:,ycol]))
    info = []
    fprlist = [0]
    tprlist = [0]
    
    
    for s in [1]:
        # for T in [5,10,20,25,30,40]:
        tlist = [i for i in range(1,10,4)]+[10,15,20,25,30,40,50]
        # tlist = tlist[::-1]
        for T in tlist: # for eccv stcf
        # for T in range(100,500,100):
            # for N in range(1,5):
            if True:
                N = 1
                ypred = []
                
                    
                if mtype == 'baf':
                    # data= data[data[:,1].argsort()] #sort elements by timestamp x,y,p,t
                    # print(data.shape, data[:,3])
                    temp=np.zeros((sx,sy))
                    real=[]
                    
                    
                    countbaf = 0
                    for d in data:
                    #   if True:
                        countbaf += 1
                        # filterres = stcffilter(d, temp, s, T, N)
                        # if filterres:
                        #     real.append(d)
                        # update(int(d[0]),int(d[1]),d[3],s,temp)
                        
                        updatebaf(int(d[2]),int(d[3]),d[1],s,temp)
                        if d[1]-temp[int(d[2])][int(d[3])]<T*1000:
                        # if True:
                            # real.append(d)
                            ypred.append(1)
                        else:
                            ypred.append(0)
                            
                    # filtered_events_numpy=np.stack(real, axis=0 )
                    # print(countbaf, filtered_events_numpy.shape)
                elif mtype == 'stcf':

                    


                    print('stcf')
                    # slayerSNN.io.encodeNpSpikes("./content/temp/ibmadvimg{}.npy".format(idx), advTD) # can be deleted
                    # data= np.load('./content/temp/ibmadvimg{}.npy'.format(idx)) # can be deleted
                    # data = advTD
                    # data= data[data[:,3].argsort()] #sort elements by timestamp x,y,p,t
                    # print(data.shape, data[:,3])

                    temp=np.zeros((sx,sy))
                    real=[]
                    for d in data:
                        if True:
                            filterres = stcffilter(int(d[xcol]),int(d[ycol]),int(d[pcol]),d[tcol], temp, s, T, N)
                            if filterres:
                                ypred.append(1) # eccv 0 is signal
                            else:
                                ypred.append(0) # eccv 1 is noise
                            # if filterres:
                            #     ypred.append(1)
                            # else:
                            #     ypred.append(0)
                            update(int(d[xcol]),int(d[ycol]),d[tcol],s,temp)
                        
                    #   update(int(d[0]),int(d[1]),d[3],s,temp)
                    #   if d[3]-temp[int(d[0])][int(d[1])]<T:
                    #     real.append(d)
                    # filtered_events_numpy=np.stack(real, axis=0 )
                elif mtype == 'fwf':
                    print('fwf')
                    # slayerSNN.io.encodeNpSpikes("./content/temp/ibmadvimg{}.npy".format(idx), advTD) # can be deleted
                    # data= np.load('./content/temp/ibmadvimg{}.npy'.format(idx)) # can be deleted
                    # data = advTD
                    # data= data[data[:,3].argsort()] #sort elements by timestamp x,y,p,t
                    # print(data.shape, data[:,3])

                    temp=np.zeros((sx,sy))
                    L = T
                    D = N
                    xarr = np.zeros(L)
                    yarr = np.zeros(L)
                    
                    real=[]
                    for d in data:
                    
                        filterres = fwffilter(d, xarr,yarr,L,D)
                        if filterres:
                            real.append(d)
                        update(int(d[0]),int(d[1]),d[3],s,temp)
                        
                    #   update(int(d[0]),int(d[1]),d[3],s,temp)
                    #   if d[3]-temp[int(d[0])][int(d[1])]<T:
                    #     real.append(d)
                    filtered_events_numpy=np.stack(real, axis=0 )
                elif mtype == 'dwf':
                    print('dwf')
                    # slayerSNN.io.encodeNpSpikes("./content/temp/ibmadvimg{}.npy".format(idx), advTD) # can be deleted
                    # data= np.load('./content/temp/ibmadvimg{}.npy'.format(idx)) # can be deleted
                    # data = advTD
                    # data= data[data[:,3].argsort()] #sort elements by timestamp x,y,p,t
                    # print(data.shape, data[:,3])

                    temp=np.zeros((sx,sy))
                    real=[]
                    
                    L = T
                    D = N
                    xarr = np.zeros(L)
                    yarr = np.zeros(L)
                    count = 0
                    for d in data:
                        
                        
                        filterres = dwffilter(d, xarr,yarr, L,D, count)
                        count += 1
                        if filterres:
                            # real.append(d)
                            ypred.append(1)
                        else:
                            ypred.append(0)
                        # update(int(d[0]),int(d[1]),d[3],s,temp)
                        
                    #   update(int(d[0]),int(d[1]),d[3],s,temp)
                    #   if d[3]-temp[int(d[0])][int(d[1])]<T:
                    #     real.append(d)
                    # filtered_events_numpy=np.stack(real, axis=0 )
                
                    
                # np.save(app + hz+mtype+'gtoutput',np.array(ypred))     
                
                # np.save(app + hz+mtype+'predoutput'+str(s)+'_'+str(T)+'_'+str(N),np.array(ypred))     
                
                # ypred = np.load(filename)
                # mtype = filename.split(hz)[1].split('pred')[0]
                # parameters = filename.split('_')
                # s = int(parameters[0][-1])
                # T = int(parameters[1])
                # N = int(parameters[2][0])
                print(mtype,s,T,N)
                
                # if label_column == 4:
                #     ytrue = 1 - ytrue
                tn, fp, fn, tp = confusion_matrix(1-ytrue,ypred).ravel()
            
                
                # cleandis = np.array(cleandis)
                # noisedis = np.array(noisedis)
                # # print('-----------'+applicaiton+'------------')
                # print(np.mean(cleandis), np.min(cleandis), np.max(cleandis), np.median(cleandis))
                # print(np.mean(noisedis), np.min(noisedis), np.max(noisedis), np.median(noisedis))
                # print(cleanarr.shape, noisearr.shape)
                # print('fpr', fp, tn,'tpr',  tp, fn)
                # continue
                fpr = fp/(tn+fp)
                tpr = tp/(tp + fn)
                fprlist.append(fpr)
                tprlist.append(tpr)
                nerr = tn/(tn+fp)
                verr = fn/(tp + fn)
                print('fpr', fp/(tn+fp))
                print('tpr', tp/(tp + fn))
                print('nerr', tn/(tn+fp))
                print('verr', fn/(tp + fn))

                acc = (tp + tn) / (tp+fp+fn+tn)
                print('acc', acc)
                print('total',tp+fp+fn+tn )
                if fp > 0:
                    snr = 10 * np.log10((tp/fp))
                else:
                    snr = 1000000000
                print('snr', snr)
                info.append([s,T,N,fpr,tpr,nerr,verr,acc,snr])
    fprlist.append(1)
    tprlist.append(1)
    from sklearn import metrics
    aucvalue = metrics.auc(fprlist,tprlist)
    print(hz,mtype,'auc', aucvalue)
    roundauc = '%.3f' % aucvalue
    np.save(app+hz+mtype + 'sweeppara' + str(roundauc), np.array(info))
    
    import csv
    fields = ['s','T', 'N', 'fpr','tpr','nerr','verr','acc','snr']
    with open(app+hz+mtype + 'sweeppara' + str(roundauc)+'eccvreverselabel.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # writing the fields
        # csvwriter.writerow(fields)
    
        # writing the data rows
        csvwriter.writerows(info)
        # agefeature,polfeature,label = mygenerator(testfiles)
        
        # if middle == 'wap':
        #     weightmatrix = np.zeros([patchsize,patchsize])
        #     center = int(patchsize*patchsize/2)
        #     for xind in range(patchsize):
        #         for yind in range(patchsize):
        #             weightmatrix[xind,yind] = math.exp(-(math.pow(xind-center,2)+math.pow(yind-center,2))/(2*center*center))
            
            

        # for resize in range(3,patchsize+1,2):
        #     beginend = int((patchsize - resize) / 2)
        #     cropend = patchsize - beginend
        #     agesub = agefeature[:,beginend:cropend,beginend:cropend].reshape(agefeature.shape[0],resize*resize)
        #     polsub = polfeature[:,beginend:cropend,beginend:cropend].reshape(polfeature.shape[0],resize*resize)
        #     if middle == 'wap':
        #         weightmatrix1 = weightmatrix[beginend:cropend,beginend:cropend].reshape(resize*resize)
        #         weightmatrix2 = np.array(list(weightmatrix1)*label.shape[0])
        #         weightmatrix3 = weightmatrix2.reshape(label.shape[0], weightmatrix1.shape[0])
            
        #     # print(weightmatrix3.shape)
    
        #     sumfeature = np.multiply(agesub,polsub)
        #     if middle == 'wap':
        #         sumfeature = np.multiply(sumfeature,weightmatrix3)
        #     sumfeature = np.sum(sumfeature,axis=1)
        # indices = np.where(ypred == 1) 
        # x1 = data[indices]
        # np.save(app + hz+'signal'+mtype+'output'+str(resize),x1)

        # indices = np.where(ypred == 0)
        # x0 = data[indices]
        # np.save(app + hz+'noise'+mtype+'output'+str(resize),x0)
        
        # print(x1.shape,x0.shape)
        # plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
        # plt.hist(x1, bins=50)
        # plt.gca().set(title='signal Frequency Histogram', ylabel='Frequency')
        # plt.savefig(app + hz+'signal'+mtype+'output'+str(resize)+'.pdf')
        # plt.cla()
        # plt.hist(x0, bins=50)

        # plt.gca().set(title='noise Frequency Histogram', ylabel='Frequency')
        # plt.savefig(app + hz+'noise'+mtype+'output'+str(resize)+'.pdf')


        
flag = sys.argv[1]
hz = sys.argv[2]
# middle = sys.argv[3]

if flag == 'dri':
    trainfiles = ['driandrealshotnoiseevents.npy']
    # testfiles = ['driandrealshotnoiseevents.npy']

    testfiles = ['driving_'+hz+'_merge_1s_0n_tiaozheng.npy']
    testfiles = ['D:\\BaiduNetdiskDownload\\ECCV2024_datasets\\AUC_test\\'+hz+'\\driving_mix_result.txt']
    trainbatches = 62000
    testbatches = 62000

if flag == 'hotel':
    trainfiles = ['hotelandrealshotnoiseevents1.npy']
    testfiles = ['hotelandrealshotnoiseevents1.npy']
    trainbatches = 31000
    testbatches = 31000
    
if flag == 'campus':
    trainfiles = ['driandrealshotnoiseevents.npy']

    testfiles = ['campus1_'+hz+'.npy']
    
    trainbatches = 30000
    testbatches = 30000

# testbatches = int(sys.argv[2])

# lastts = np.load(testfiles[0])[0,1]
# print('lastts', lastts)


# initializetsandpolmap(5,lastts)

wkfiles1 = []
# use float model
# bits = [32]#+[i for i in range(2,9,2)]

# middle = 'single'

# use quantized model
bits = [3]#+[i for i in range(2,9,2)]

# middle = 'qsingle'
for repeat in range(1):
    trainFunction(flag, hz,mtype)

