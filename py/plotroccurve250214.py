import csv  
from matplotlib import pyplot as plt  
from datetime import datetime 
import matplotlib
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import numpy as np

import os,sys

from numpy.core.fromnumeric import sort

rootpath= 'D:\\code\\qmlpf-main\\qmlpf-main\\250219plotauc' # adjusted according to your path
# npyfolder = sys.argv[1] # the folder where the npy data are
files = os.listdir(rootpath)
# application = sys.argv[2]
# targetfolder = sys.argv[2] # the folder for storing result images
# application = sys.argv[3] # incicating which dataset/application is used
# # if len(sys.argv) < 3:
# #     return "Parameters not enough. three parameters needed: npyfolder, targetfolder, application"
# filtername = sys.argv[4]
# plotflag = sys.argv[5]

# if not os.path.exists(targetfolder):
#     os.makedirs(rootpath+targetfolder)

methods = []

thrs = []
tprs = []
tnrs = []
brs = []
fprs = []
aucs = []

colors = [['black'],['blue'],['green'],['purple'],['cyan'],['red'],['darkorange'],['plum'],['goldenrod'],['darkblue'],['darkgreen'],['darkred'],['grey'],['yellow'],['olive'],['pink'],['brown']]

matplotlib.rcParams.update({'font.size': 14})
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from plot_new_curve import getauc

fig=plt.figure() 
cc = 0 

app = sys.argv[1]
hz = 'hz'
# filn = sys.argv[2]
import pandas as pd
from sklearn import metrics

print(rootpath)
print(files)
for filename in files:
    # print(filename)
    if app in filename and hz in filename and '.csv' in filename:# and application in filename:# and filtername in filename:
        print(filename)
        filn = filename.split(hz)[1].split('f')[0] + 'f'
        print(filn)
        # arr = np.load(npyfolder + '/' + filename)
        data = pd.read_csv(rootpath+'\\'+filename,header=None).values
        print(data.shape)
        # continue
        # data = data[1:].astype(np.float32)
        # datadict = {}
        # datadict[3] = data[:8]
        # datadict[5] = data[8:16]
        # datadict[7] = data[16:24]
        fprlist = []
        tprlist = []
        if filn == 'qmlpf' or filn == 'ebf':
            fprlist.append(1)
            tprlist.append(1)
            data = data[:,2:4]
        
        elif filn == 'baf' or filn == 'stcf':
            fprlist.append(0)
            tprlist.append(0)
            data = data[:,3:5]
            
        elif filn == 'edf':
            data = data[:,2:4]

        # data3 = data[:8,:]
        
        # if 'thr' in filename:
        # print(arr.shape)
        # continue
        # leng = arr.shape[0]
        # halflen = int(leng/2)
        # y_true = arr[:halflen].astype(np.int)
        # y_score = arr[halflen:]#.astype(np.int)
        # print(y_true.shape,y_score.shape,y_true.dtype)
        # print()

        # print(filename)
        # print(np.unique(y_score).shape)
        # continue
        
        curcolor = colors[cc][0]
        ls = 'solid'
        # if '2x' in filename:
        #     ls = 'solid'
        # for idx in range(3,8,2):
        #     tmpdata = datadict[idx]
        if True:
            method = filn
            
            # fprlist = [1]
            # tprlist = [1]
            fprlist+= list(data[:,0])
            tprlist+= list(data[:,1])
            # fprlist.append(0)
            # tprlist.append(0)
        
        
        
        
            if filn == 'qmlpf' or filn == 'ebf':
                fprlist.append(0)
                tprlist.append(0)

            elif filn == 'baf' or filn == 'stcf':
                fprlist.append(1)
                tprlist.append(1)
                
                
            print(fprlist,tprlist)
            
            
            aucvalue = metrics.auc(fprlist,tprlist)
            print('aucvalue', aucvalue)
            roundauc = '%.3f' % aucvalue
            
            if filn == 'baf':
                
                ls = '-.'
                
            elif filn == 'stcf':
                
                ls = '-o'
                
            elif filn == 'edf':
                
                ls = '--'
                
            elif filn == 'ebf':
                
                ls = '-*'

            else:
                ls = '1'
            

            plt.plot(fprlist,tprlist,ls,label=method.upper()+', AUC='+str(roundauc),color=curcolor)

        
        # plt.plot([0,1],[0,1],'r--')

        cc += 1

plt.legend(loc='lower right')
 
# plt.title('Driving')
plt.plot([(0,0),(1,1)],'r--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

ax=plt.gca()

tick_spacing = 0.05
xtick_spacing = 0.1
ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

x1, y1 = 0,1
x2, y2 = -0.05, 1.1

plt.plot([x1, x2],[y1,y2], "*", color='r')

plt.annotate(r'$Ideal Point$', xy=(x1,y1),xycoords='data',xytext=(x2,y2),textcoords='data',
             fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.3'))


plt.savefig(app+hz+  'AllROC250219.pdf') 
 
