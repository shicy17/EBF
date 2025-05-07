import numpy as np
import os
import sys


# 计算马氏距离

app = sys.argv[1]
hidden = 50

# hz = sys.argv[2]
weights = sys.argv[2]

# threshold = 15
info = []

def calculatemetrics(threshold,cleanarr,noisearr):

    # train
    
    
    print(np.mean(cleanarr), np.mean(noisearr))

    # print(cleanarr.shape, noisearr.shape)

    # # trainlen = int(sys.argv[3])
    # noisetrainarr = noisearr[:trainlen,:] # trainlen就是用于训练的事件数据量，可变
    # noisetestarr = noisearr[10000:,:] 
    # # 用于test的干净事件数据，因为所有train events数量最大的是100000，为了保证用不同训练数据的测试公平性，测试的干净数据从100000之后开始
    # # 计算干净矩阵均值
    # y = np.mean(noisetrainarr,axis=0) 
    # # 计算协方矩阵
    # cov = np.cov(noisetrainarr.T)

    # # 计算协方差矩阵的逆矩阵
    # inv_cov = np.linalg.inv(cov)
    
    # 测试阶段
    cleandis = []
    noisedis = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # 先判断noise
    for x in cleanarr:
        # mahalanobis_distance = np.sqrt(np.dot(np.dot((x - y).T, inv_cov), (x - y)))
        # cleandis.append(mahalanobis_distance)
        if x > threshold:
            tp += 1 # noise 
        else:
            fn += 1 # noise but pass
        # print(mahalanobis_distance)
        
    # 再判断没有被用于训练的干净数据
    for x in noisearr:
        # mahalanobis_distance = np.sqrt(np.dot(np.dot((x - y).T, inv_cov), (x - y)))
        # noisedis.append(mahalanobis_distance)
        if x < threshold:
            tn += 1 # real
        else:
            fp += 1

        # print(mahalanobis_distance)
        
    return tp,fp,tn,fn

# 消融实验，换参数，对不同的数据可能最好的参数会不同
# for output in [10,30,50]:
#     for thr in range(10,40,5):
#         for trainlen in [1000,5000,10000,50000,100000]:

# for output in [20]:
    
#     # if True:
#     for trainlen in [10000]:

from sklearn import metrics
for nhz in [1]:
    
    
# for nhz in [1]:
    hz = str(nhz) + 'hz'
    if app == 'dri':
        trainfiles = ['driandrealshotnoiseevents.npy']
        # testfiles = ['driandrealshotnoiseevents.npy']

        testfiles = ['driving_'+hz+'_merge_1s_0n_pianzhi_tiaozheng.npy']

        
        

    if app == 'campus':
        trainfiles = ['driandrealshotnoiseevents.npy']

        testfiles = ['campus1_'+hz+'.npy']
        # testfiles = ['buaa1_' + hz + '.npy']
        
        
        
    if app == 'hotel':
        trainfiles = ['hotelandrealshotnoiseevents1.npy']
        testfiles = ['hotelandrealshotnoiseevents1.npy']
    
    # ours
    # ytrue = np.load(testfiles[0], allow_pickle=True)[:3000000,0].astype(np.int32)
    
    # edformer data
    data = np.load('ebf1231retestours' + hz + 'predandtrue.npy', allow_pickle=True)  
    # data = np.load('ebf1231retesteccv' + hz + 'predandtrue.npy', allow_pickle=True)     
       
    initpredictions = data[:,0]
    ytrue = data[:,1]
    # ytrue = np.load(testfiles[0], allow_pickle=True)[:3000000,4] == 'signal'
    # for resize in range(3,8,2):
    for resize in [5]:
        # cleanarr = np.load('0416'+app + hz+'signal'+weights+str(resize)+'.npy')[:3000000]
        # noisearr = np.load('0416'+app + hz+'noise'+weights+str(resize)+'.npy')[:3000000]
        # print('0416'+app + hz+'noise'+weights+str(resize))
        # initpredictions = np.load('0417ebf'+app + hz +weights+str(resize)+'.npy')[:3000000]
        
        # ours
        # initpredictions = np.load('0611ebf'+app + hz +weights+str(resize)+'.npy')[:3000000]
        print(np.mean(initpredictions), np.median(initpredictions))

        fprlist = [1]
        tprlist = [1]
        print('***********resize************',resize)
        if True:
            for thr in [i for i in range(1,10,2)]+[i for i in range(10,150,20)]:
            # for thr in [1,2,3,4,5,6,7,8,11,13]:
                thr = thr*1.0/10
            
                print('thr',thr)
                
                ypred = (initpredictions > thr).astype(np.int32)
                # print(ytrue.shape, ypred.shape)
                # print(ytrue.dtype, ypred.dtype)
                tn, fp, fn, tp = metrics.confusion_matrix(ytrue,ypred).ravel()
                # tp,fp,tn,fn = calculatemetrics(thr,cleanarr,noisearr)
                
                
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
                if fp>0 and (tp/fp) > 0:
                    snr = 10 * np.log10((tp/fp))
                else:
                    snr = 0
                print('snr', snr)
                info.append([resize,thr,fpr,tpr,nerr,verr,acc,snr])
                # info[thr] = perinfo

            fprlist.append(0)
            tprlist.append(0)
            from sklearn import metrics
            aucvalue = metrics.auc(fprlist,tprlist)
            print('resize', resize, 'aucvalue', aucvalue)
# 把数据存储到csv中

import csv
fields = ['resize','thr', 'fpr','tpr','nerr','verr','acc','snr']
# with open(app +weights+'5hzebf0422.csv', 'w', newline='') as csvfile:
with open(app +weights+'5hzebf0611.csv', 'w', newline='') as csvfile:

    csvwriter = csv.writer(csvfile)
    # writing the fields
    # csvwriter.writerow(fields)
 
    # writing the data rows
    csvwriter.writerows(info)
    
