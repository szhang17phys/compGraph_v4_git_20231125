# utlities
import os
import re
import sys
import time

import tensorflow as tf

import pickle as pk
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D # draw 3d figure---

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from pylab import *
from networks import *

#-----------------------------------
pos_X = []
pos_Y = []
pos_Z = []
#-----------------------------------
XA = []
#Added by Shu, 20231123---
#Here opCh refers to all opch of protodunevd_v4---
opCh = []

large = 23
med   = 17
small = 9
fontax = {'family': 'sans-serif',
          'color':  'black',
          'weight': 'bold',
          'size': 18,
          }
          
fontlg =  {'family': 'sans-serif',
           'weight': 'normal',
           'size': 17,
          }    
params = {'axes.titlesize': 10,
          'axes.labelsize': 9,
          'figure.titlesize': 21,
          'figure.figsize': (9, 6),
          'figure.dpi': 200,
          'lines.linewidth':  3,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          }
plt.style.use('seaborn-paper')
plt.rcParams.update(params)

#========================================
def mkdir(dir="image/"):
       if not os.path.exists(dir):
           print('make directory '+str(dir))
           os.makedirs(dir)

#This is about png generated automatically after training, Shu, 20231123---
def save_plot(fake, true, nbin, xlabel, ylabel, name):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1,1,1)

    ax.bar(nbin,height=true,color='blue',width=1, alpha=0.5, label='Geant4')
    ax.grid(b=True)
    
    plt.scatter(nbin, fake, color='red', s=5, label='CompGraph Module')
    plt.legend()
    plt.savefig(name + '-lin.png', dpi=200)
    plt.yscale('log')
    plt.legend()
    plt.savefig(name + '-log.png', dpi=200)    
    plt.clf()
    plt.close()
           
def savescatter(labelX, labelY, ylim, name):
    plt.legend(loc='upper left', prop=fontlg)
    plt.grid(color='0.9', linestyle='-.', linewidth=0.5)
    
    plt.xlabel(labelX, fontdict=fontax)
    plt.ylabel(labelY, fontdict=fontax)
    
    plt.ylim(0, ylim)    
    plt.savefig(name+'.png')
    plt.clf()
    
def savehist(hist, range, labelX, labelY, title, name, w):        
    if w:
        plt.rcParams['figure.figsize'] = (12, 9)
    else: 
        plt.rcParams['figure.figsize'] = (12, 9)
    plt.rcParams['figure.figsize'] = (12, 9)
    
    mu            = np.mean(hist)
    sigma         = np.std(hist)
    bins          = 200
    fig, ax       = plt.subplots()
    h1d, nbins, _ = ax.hist(hist, bins=bins, range=range, histtype='step', color='navy', linewidth=2)

    x     = (nbins[1:] + nbins[:-1])/2
    stats =  fit_gaussian(x, h1d)

#---To test the sigma------------------------
    Weights    = h1d/np.sum(h1d)
    Ini_mu     = np.average(x, weights = Weights)
    Ini_sigma  = np.sqrt(np.average(x**2, weights = Weights) - Ini_mu**2)
    #print("mu= ", mu, ", sigma = ", sigma, ";;;;; Ini_mu = ", Ini_mu, ", Ini_sigma = ", Ini_sigma)
#----------------------------------------------


    sigus  = int(bins*(stats[1]+abs(stats[2]+1))/2)
    sigls  = int(bins*(stats[1]-abs(stats[2]+1))/2)
    sigs   = sum(h1d[sigls:sigus])
    
    sigup  = int(bins*(stats[1]+1.1)/2)
    siglp  = int(bins*(stats[1]+0.9)/2)
    sigp   = sum(h1d[siglp:sigup])

    tots   = sum(h1d[0:bins])

   #added by Shuaixiang (Shu), Mar 29, 2023---
    sigup15 = int(bins*(stats[1]+1.15)/2)
    siglp15 = int(bins*(stats[1]+0.85)/2)
    sigp15  = sum(h1d[siglp15:sigup15])

    stat10 = sum(h1d[90:110])
    stat15 = sum(h1d[85:115])
    stat20 = sum(h1d[80:120])
    stat25 = sum(h1d[75:125])    

    print("\n\n--------------- ", title, " ---------------")
    print('Total vertice: '+str(tots))
    print('(Fitting) Vertice in 1 sigma: '+str(sigs)+', '+str(format(sigs/tots, '.3f')))
    print('(Fitting) Vertice in 10%: '+str(sigp)+', '+str(format(sigp/tots, '.3f'))+
          ', outside 10%: '+str(format(1-sigp/tots, '.3f')))
    print('(Fitting) Vertexs in 15%: '+str(sigp15)+', '+str(format(sigp15/tots, '.3f'))+
          ', outside 15%: '+str(format(1-sigp15/tots, '.3f')))

    print('Raw    :  mu = '+str(format(mu, '.4f'))+', sigma = '+str(format(sigma, '.4f')))
    print('Histo  :  mu = '+str(format(Ini_mu, '.4f'))+', sigma = '+str(format(Ini_sigma, '.4f')))
    print('Fitting:  mu = '+str(format(stats[1], '.4f'))+',  sigma = '+str(format(stats[2], '.4f')))

    print('\nInside [-0.10, 0.10]: '+ str(stat10))
    print('Inside [-0.10, 0.10]: '+str(format(stat10/tots, '.4f'))+
          ';   Outside: '+str(tots - stat10)+',  '+str(format(1-stat10/tots, '.4f')))
    print('Inside [-0.15, 0.15]: '+ str(stat15))
    print('Inside [-0.15, 0.15]: '+str(format(stat15/tots, '.4f'))+
          ';   Outside: '+str(tots - stat15)+',  '+str(format(1-stat15/tots, '.4f')))
    print('Inside [-0.20, 0.20]: '+ str(stat20))
    print('Inside [-0.20, 0.20]: '+str(format(stat20/tots, '.4f'))+
          ';   Outside: '+str(tots - stat20)+',  '+str(format(1-stat20/tots, '.4f')))
    print('Inside [-0.25, 0.25]: '+ str(stat25))
    print('Inside [-0.25, 0.25]: '+str(format(stat25/tots, '.4f'))+
          ';   Outside: '+str(tots - stat25)+',  '+str(format(1-stat25/tots, '.4f')))


    x_int = np.linspace(nbins[0], nbins[-1], 1000)
    ax.plot(x_int, gaussian(x_int, stats[0], stats[1], stats[2]), color='red', 
            linestyle='-.', linewidth=1.5, label='$\mu=%.3f,\ \sigma=%.3f$' %(stats[1], abs(stats[2])))
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
#    ax.legend(loc='upper left', handles=handles, labels=labels, prop=fontlg)

    plt.legend(['$\mu$ = {}, SD = {}'.format(format(Ini_mu, '.3f'), format(Ini_sigma, '.3f')), 
                '$\mu$ = {}, $\sigma$ = {}'.format(format(stats[1], '.3f'), format(stats[2], '.3f'))], 
                loc='upper left', prop={'size':17})

    plt.xlabel(labelX, fontdict=fontax)
    plt.ylabel(labelY, fontdict=fontax)

#    plt.yscale('log')
#    plt.ylim(1, 400000)

    plt.title(title, fontdict={'weight':'normal','size':35})
    plt.grid(color='0.9', linestyle='-.', linewidth=0.6)
    #plt.savefig(name+'.pdf')
    plt.savefig(name+'.png')
    plt.clf()
    plt.close()
    
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gaussian_grad(x, a, x0, sigma):
    exp_arg = -(x - x0)**2 / (2 * sigma**2)
    exp     = np.exp(exp_arg)    
    f       = a * exp
    
    grad_a      = exp
    grad_x0     = (x - x0) / (sigma**2) * f
    grad_sigma  = (x - x0)**2 / (sigma**3) * f
    
    return np.vstack([grad_a, grad_x0, grad_sigma]).T

def fit_gaussian(x, hist):
    # NOTE: had to normalize since average sometimes fails due to numerical errors.
    weights    = hist/np.sum(hist)
    ini_a      = np.max(hist)
    ini_mu     = np.average(x, weights = weights)
    ini_sigma  = np.sqrt(np.average(x**2, weights = weights) - ini_mu**2)
    
    ini        = [ini_a, ini_mu, ini_sigma]
    popt, _    = curve_fit(gaussian, xdata=x, ydata=hist, p0=ini,
                           bounds = [[0, x[0], 0], [np.inf, x[-1], np.inf]],
                           jac = gaussian_grad, max_nfev = 10000)
    return popt


def get_data(path, nfile, dim_pos, dim_pdr):
    dataset = []
    files = [f for f in os.listdir(path)]
    print('Processing ' + str(len(files) if nfile == -1 or nfile > len(files) else nfile) + ' files...')
    for i,f in enumerate(files):
        if i == nfile:
            break
        datafile = os.path.join(path, f)
        datatmp  = []
        with open(datafile, 'rb') as ft:
            datatmp = pk.load(ft)
            dataset.extend(datatmp)
    n_vec = len(dataset)
    print('Dataset loaded, dataset length: '+str(n_vec))
    
    inputs  = np.zeros(shape=(n_vec, dim_pos))
    outputs = np.zeros(shape=(n_vec, dim_pdr))
    for i in range(0, n_vec):
        event = dataset[i] 
        inputs[i,0] = event['x']
        inputs[i,1] = event['y']
        inputs[i,2] = event['z']
        outputs[i]  = event['image'].reshape(dim_pdr)
    return inputs, outputs
#==========================================







#===========================================
#Sugggested by Mu, for module 0, 20230126---
def eval_model_dunevd_16op(pos, pdr, pre, evlpath):                
    print('Behavior testing for Module 0 PDS...')            
    cut_x  = (pos[:,0] > 235) & (pos[:,0] < 245)
    cut_y  = (pos[:,1] > 120) & (pos[:,1] < 130)
    coor_z = pos[:,2][cut_x & cut_y]
    true_z = pdr[cut_x & cut_y]
    emul_z = pre[cut_x & cut_y]
    #scan along y direction---
#    cut_z  = (pos[:,2] > -250) & (pos[:,2] < 550)
#    cut_x  = (pos[:,0] > -425) & (pos[:,0] < 364)
    cut_z  = (pos[:,2] > 240) & (pos[:,2] < 250)
    cut_x  = (pos[:,0] > 140) & (pos[:,0] < 150)
    coor_y = pos[:,1][cut_z & cut_x]
    true_y = pdr[cut_z & cut_x]
    emul_y = pre[cut_z & cut_x]
    
    cut_y  = (pos[:,1] > 125) & (pos[:,1] < 130)
    cut_z  = (pos[:,2] > 145) & (pos[:,2] < 150)
    coor_x = pos[:,0][cut_y & cut_z]            
    true_x = pdr[cut_y & cut_z]    
    emul_x = pre[cut_y & cut_z]
    
    num_x = len(coor_x)        
    num_y = len(coor_y)
    num_z = len(coor_z)
    #---Z SCAN---------------
#    print('Scan Z with ' + str(num_z) + ' points.')
    #To test------
#    print("Output of coor_z:")
#    print(coor_z)
#    print("========================================")
    opch   = ['PD 0', 'PD 1',  'PD 2',  'PD 3',  'PD 4',  'PD 5',  'PD 6',    'PD 7', 'PD 8', 'PD 9', 'PD 10', 'PD 11', 'PD 12', 'PD 13', 'PD 14', 'PD 15']
#    colors = ['blue',  'cyan',   'black',  'green',  'indigo', 'greenyellow', 'magenta', 'darkorchid', 'red', 'tan', 'pink', 'lime', 'navy', 'violet', 'crimson', 'grey']
    colors = ['blue',  'red',   'black',  'green',  'blue', 'red', 'black', 'green', 'blue', 'red', 'black', 'green', 'blue', 'red', 'black', 'green']
              
    num_op   = len(opch)          
    true_z_s = np.zeros(shape=(num_z, num_op))    
    emul_z_s = np.zeros(shape=(num_z, num_op))

    for index, op in zip(range(0, num_op), range(0, 4)):#change range---
        true_z_s[:,index] = true_z[:,op]
        emul_z_s[:,index] = emul_z[:,op]
        
    ylim = 0.003#change ylim---
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, 4):#change range---; change true_z_s[; i-4/8/12]---
        plt.scatter(coor_z, true_z_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'true_z_s')
    
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, 4):#change range--- change emul_z_s[;,i-4/8/12]---
        plt.scatter(coor_z, emul_z_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', linestyle='solid', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'emul_z_s')


    #---Y SCAN------------------
#    print("\n------Y Scan-------------------------------------------")
#    print('Scan Y with ' + str(num_y) + ' points.')
    #test------
#    print("Output of coor_y:")
#    print(coor_y)
#    print("Length of coor_y : ", len(coor_y))


    coor_y2 = []
#    true_y2 = [][16]
#    emul_y2 = [][16]
    for num in range(0, num_y):
        if pos[num,0]>140 and pos[num,0]<150 and pos[num,2]>240 and pos[num,2]<250:
            coor_y2.append(coor_y[num])

    true_y2 = np.zeros(shape=(len(coor_y2), num_op))
    emul_y2 = np.zeros(shape=(len(coor_y2), num_op))
    i = 0
    for num in range(0, num_y):
        if pos[num,0]>140 and pos[num,0]<150 and pos[num,2]>240 and pos[num,2]<250:
            for op in range(0, num_op):
                true_y2[i, op] = true_y[num, op]
                emul_y2[i, op] = emul_y[num, op]
            i = i+1        

#    print("Length of coor_y2: ", len(coor_y2))
#    print("\nShape of emul_z: ", shape())
#    print("Shape of emul_y:  ", emul_y.shape)
#    print("Shape of emul_y2: ", emul_y2.shape)
#    print("-------------------------------------------------------")

    num_op   = len(opch)
    true_y_s = np.zeros(shape=(len(coor_y), num_op))
    emul_y_s = np.zeros(shape=(len(coor_y), num_op))

    for index, op in zip(range(0, num_op), range(0, 16)):#change range---
        true_y_s[:,index] = true_y[:,op]
        emul_y_s[:,index] = emul_y[:,op]
        
    ylim = 0.005#change limit---
    for i in range(0, 4):#change range---
        plt.scatter(coor_y, true_y_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'true_y_s')    
    
    for i in range(0, 4):#change range---
        plt.scatter(coor_y, emul_y_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'emul_y_s')


    #---X SCAN-----------------
#    print('Scan X with ' + str(num_x) + ' points.')
    true_x_s = np.zeros(shape=(num_x, num_op))
    emul_x_s = np.zeros(shape=(num_x, num_op))
    
    for index, op in zip(range(0, num_op), range(0, 4)):#change range---
        true_x_s[:,index] = true_x[:,op]
        emul_x_s[:,index] = emul_x[:,op]
        
    ylim = 0.0035#change limit---
    for i in range(0, 4):#change range---
        plt.scatter(coor_x, true_x_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'true_x_s')
    
    for i in range(0, 4):#change range---
        plt.scatter(coor_x, emul_x_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'emul_x_s')
#=============================================






#=============================================
#For protodunevd_v4, 40 opch---
#Written by Shu, 20231123---
def eval_model_protodunevd_v4(pos, pdr, pre, evlpath):                
    print('Behavior testing for protoDUNE-VD (v4) PDS...')
    #scan along z direction---            
    cut_x  = (pos[:,0] > -375) & (pos[:,0] < 415)
    cut_y  = (pos[:,1] > -427.4) & (pos[:,1] < 427.4)
    coor_z = pos[:,2][cut_x & cut_y]
    true_z = pdr[cut_x & cut_y]
    emul_z = pre[cut_x & cut_y]

    #scan along y direction---
    cut_z  = (pos[:,2] > -277.75) & (pos[:,2] < 577.05)
    cut_x  = (pos[:,0] > -375) & (pos[:,0] < 415)
    coor_y = pos[:,1][cut_z & cut_x]
    true_y = pdr[cut_z & cut_x]
    emul_y = pre[cut_z & cut_x]
    
    #scan along x direction---
    cut_y  = (pos[:,1] > -427.4) & (pos[:,1] < 427.4)
    cut_z  = (pos[:,2] > -277.75) & (pos[:,2] < 577.05)
    coor_x = pos[:,0][cut_y & cut_z]            
    true_x = pdr[cut_y & cut_z]    
    emul_x = pre[cut_y & cut_z]
    
    num_z = len(coor_z)       
    num_y = len(coor_y)
    num_x = len(coor_x)

    opch = ['PD 0', 'PD 1', 'PD 2', 'PD 3', 'PD 4', 'PD 5', 'PD 6', 'PD 7', 'PD 8', 'PD 9', 'PD 10', 'PD 11', 'PD 12', 'PD 13', 'PD 14', 'PD 15', 'PD 16', 'PD 17', 'PD 18', 'PD 19', 'PD 20', 'PD 21', 'PD 22', 'PD 23', 'PD 24', 'PD 25', 'PD 26', 'PD 27', 'PD 28', 'PD 29', 'PD 30', 'PD 31', 'PD 32', 'PD 33', 'PD 34', 'PD 35', 'PD 36', 'PD 37', 'PD 38', 'PD 39']
#    colors = ['blue',  'cyan',   'black',  'green',  'indigo', 'greenyellow', 'magenta', 'darkorchid', 'red', 'tan', 'pink', 'lime', 'navy', 'violet', 'crimson', 'grey']
    colors = ['blue', 'red', 'black', 'green', 'blueviolet', 'blue', 'red', 'black', 'green', 'blueviolet', 'blue', 'red', 'black', 'green', 'blueviolet', 'blue', 'red', 'black', 'green', 'blueviolet', 'blue', 'red', 'black', 'green', 'blueviolet', 'blue', 'red', 'black', 'green', 'blueviolet', 'blue', 'red', 'black', 'green', 'blueviolet', 'blue', 'red', 'black', 'green', 'blueviolet']
              
    num_op   = len(opch) 

    #---Z SCAN-----------------------------------------------------   
    print('\n\nScan Z with ' + str(num_z) + ' points.')
    #To test------
    print("Output of coor_z:")
    print(coor_z)
    print("=================================")
        
    true_z_s = np.zeros(shape=(num_z, num_op))    
    emul_z_s = np.zeros(shape=(num_z, num_op))

    for index, op in zip(range(0, num_op), range(0, 4)):#change range---
        true_z_s[:,index] = true_z[:,op]
        emul_z_s[:,index] = emul_z[:,op]
        
    ylim = 0.003#change ylim---
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, 4):#change range---; change true_z_s[; i-4/8/12]---
        plt.scatter(coor_z, true_z_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'true_z_s')
    
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, 4):#change range--- change emul_z_s[;,i-4/8/12]---
        plt.scatter(coor_z, emul_z_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', linestyle='solid', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'emul_z_s')


    #---Y SCAN-----------------------------------------------------
    print('\n\nScan Y with ' + str(num_y) + ' points.')
    #test------
    print("Output of coor_y:")
    print(coor_y)
    print("Length of coor_y : ", len(coor_y))
    print("=================================")

    coor_y2 = []
    for num in range(0, num_y):
        if pos[num,0]>-375 and pos[num,0]<415 and pos[num,2]>-277.75 and pos[num,2]<577.05:
            coor_y2.append(coor_y[num])

    true_y2 = np.zeros(shape=(len(coor_y2), num_op))
    emul_y2 = np.zeros(shape=(len(coor_y2), num_op))
    i = 0
    for num in range(0, num_y):
        if pos[num,0]>-375 and pos[num,0]<415 and pos[num,2]>-277.75 and pos[num,2]<577.05:
            for op in range(0, num_op):
                true_y2[i, op] = true_y[num, op]
                emul_y2[i, op] = emul_y[num, op]
            i = i+1        

#    print("Length of coor_y2: ", len(coor_y2))
#    print("\nShape of emul_z: ", shape())
#    print("Shape of emul_y:  ", emul_y.shape)
#    print("Shape of emul_y2: ", emul_y2.shape)
#    print("-------------------------------------------------------")

    num_op   = len(opch)
    true_y_s = np.zeros(shape=(len(coor_y), num_op))
    emul_y_s = np.zeros(shape=(len(coor_y), num_op))

    for index, op in zip(range(0, num_op), range(0, 40)):#change range---
        true_y_s[:,index] = true_y[:,op]
        emul_y_s[:,index] = emul_y[:,op]
        
    ylim = 0.005#change limit---
    for i in range(0, 4):#change range---
        plt.scatter(coor_y, true_y_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'true_y_s')    
    
    for i in range(0, 4):#change range---
        plt.scatter(coor_y, emul_y_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'emul_y_s')


    #---X SCAN-----------------------------------------------------
    print('\n\nScan X with ' + str(num_x) + ' points.')
    print("Output of coor_x:")
    print(coor_x)
    print("Length of coor_x : ", len(coor_x))
    print("=================================")
    true_x_s = np.zeros(shape=(num_x, num_op))
    emul_x_s = np.zeros(shape=(num_x, num_op))
    
    for index, op in zip(range(0, num_op), range(0, 4)):#change range---
        true_x_s[:,index] = true_x[:,op]
        emul_x_s[:,index] = emul_x[:,op]
        
    ylim = 0.0035#change limit---
    for i in range(0, 4):#change range---
        plt.scatter(coor_x, true_x_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'true_x_s')
    
    for i in range(0, 4):#change range---
        plt.scatter(coor_x, emul_x_s[:,i], marker='.', s=75, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'emul_x_s')
#====================================================










        
def eval(pos, pdr, mtier, modpath, evlpath):
    dim_pdr = pdr.shape[1]
    
    if dim_pdr == 90:
        if mtier == 0:
            print('Loading ProtoDUNE t0 net...')
            model = model_protodunev7_t0(dim_pdr)
        elif mtier == 1:
            print('Loading ProtoDUNE t1 net...')
            model = model_protodunev7_t1(dim_pdr)
        elif mtier == 2:
            print('Loading ProtoDUNE t2 net...')
            model = model_protodunev7_t2(dim_pdr)
        elif mtier == 3:
            print('Loading ProtoDUNE t3 net...')
            model = model_protodunev7_t3(dim_pdr)
    elif dim_pdr == 480:
        if mtier == 0:
            print('Loading DUNE t0 net...')
            model = model_dune10kv4_t0(dim_pdr)
        elif mtier == 1:
            print('Loading DUNE t1 net...')
            model = model_dune10kv4_t1(dim_pdr)
        elif mtier == 2:
            print('Loading DUNE t2 net...')
            model = model_dune10kv4_t2(dim_pdr)
        elif mtier == 3:
            print('Loading DUNE t3 net...')
            model = model_dune10kv4_t3(dim_pdr)
    elif dim_pdr == 168:
        if mtier == 0:
            print('Loading VD t0 net...')
            model = model_dunevd_t0(dim_pdr)
        if mtier == 1:
            print('Loading VD t1 net...')
            model = model_dunevd_t1(dim_pdr)
    elif dim_pdr == 160:
        if mtier == 0:
            print('Loading ProtoDUNEHD t0 net...')
            model = model_protodunehd_t0(dim_pdr)
    #Suggested by Mu, for module 0, 20230125---
    elif dim_pdr == 16:
        if mtier == 0:
            print('Loading module 0 16op net...')
            model = model_dunevd_16op(dim_pdr)
    #Added by Shu, for protodunevd_v4 40 opch, 20231124---
    elif dim_pdr == 40:
        if mtier == 0:
            print('Loading protodunevd_v4 40 opch net...')
            model = model_protodunevd_v4(dim_pdr)

 
    weight = modpath+'best_model.h5'
    if os.path.isfile(weight):
        print('Loading weights...')
        model.load_weights(weight)
    else:
        print('Err: no weight found!')
        return
        
    print('Predicting...')
    tstart = time.time()
    pre = model.predict({'pos_x': pos[:,0], 'pos_y': pos[:,1], 'pos_z': pos[:,2]})
    print('\n')
    print( '\nFinish evaluation in '+str(time.time()-tstart)+'s.')
    
    if dim_pdr == 90:
        eval_protodune(pos, pdr, pre, evlpath)
    elif dim_pdr == 480:
        eval_dune(pos, pdr, pre, evlpath)
    elif dim_pdr == 168:
        eval_dunevd(pos, pdr, pre, evlpath)
    elif dim_pdr == 160:
        eval_protodunehd(pos, pdr, pre, evlpath)
    #Suggested by Mu, for module 0, 20230126---
    elif dim_pdr == 16:
        eval_model_dunevd_16op(pos, pdr, pre, evlpath)
    #Added by Shu, for protodunevd_v4 40 opch, 20231125---
    elif dim_pdr == 40:
        eval_model_protodunevd_v4(pos, pdr, pre, evlpath)


##===Case of sum up all XArapucas=======================================
#    print('\n--------------------Scan along X-------------------------')
#    print('Intensity and resolution evaluating...')            
#    pre = pre.sum(axis=1)#it means suming up all pds; YES, refer to Mu's paper---
#    pdr = pdr.sum(axis=1)
#    
#    cut = (pre != 0) & (pdr != 0)
#    #Shu: 400 will not work---
#    x_list = [100, 200, 300, 400, 500]
#    for i in range(len(x_list)):
#        w = True
#        if i == 0:#here range of random abs(x) is [0. 100]---
#            low_x = np.absolute(pos[:,0]) >  0#the x position of the vertex---
#            upp_x = np.absolute(pos[:,0]) <= x_list[i]
#            title = '0<|x|<%d' %(x_list[i])
#        elif i == (len(x_list)-1):#here range of random x is [0, x_list[i-1]]---
#            low_x = np.absolute(pos[:,0]) >  0
#            upp_x = np.absolute(pos[:,0]) <= x_list[i]
#            #title = 'All (%d<|x|<%d)' %(0, x_list[i])
#            title = 'Vertex of Whole Space'
#            w     = False
#        else:#here range of random x is [x_list[i-1], x_list[i]]---
#            low_x = np.absolute(pos[:,0]) <= x_list[i]
#            upp_x = np.absolute(pos[:,0]) >  x_list[i-1]
#            title = '%d<|x|<%d' %(x_list[i-1], x_list[i])
#            
#        image_diff = pre[cut & low_x & upp_x] - pdr[cut & low_x & upp_x]
#        image_true = pdr[cut & low_x & upp_x]
#        visib_diff = np.divide(image_diff, image_true)
#        
#
#        savehist(visib_diff, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'X_intensity-'+str(x_list[i]), w) 
#        
#        print("\nBig biased values:")
#        Bias = 0
#        for diff in visib_diff:
#            if abs(diff) > 1:
#                print(str(format(diff, '.3f')))
#                Bias += 1
#        print("# of abs((emul-simu)/simu)>1 : ", Bias)
#
#    print('---------------------------------------------------------')
##        print("Length of image_diff: ", len(image_diff))
##        print("iamge_diff: \n",image_diff)
##---------------------------------------------------------------------
#
#
##---Scan along y and z axis-------------------------------------------
#    print('\n\n\n------------------- Scan along Y------------------------')
#    print('Intensity and resolution evaluating...')            
#    
#    cut = (pre != 0) & (pdr != 0)
#    #Shu: 400 will not work---
#    y_list = [100, 200, 300, 400, 500]
#    for i in range(len(y_list)):
#        w = True
#        if i == 0:#here range of random y is [0, 100]---
#            low_y = np.absolute(pos[:,1]) >  0#the y position of the vertey---
#            upp_y = np.absolute(pos[:,1]) <= y_list[i]
#            title = '0<|y|<%d' %(y_list[i])
#        elif i == (len(y_list)-1):#here range of random y is [0, y_list[i-1]]---
#            low_y = np.absolute(pos[:,1]) >  1
#            upp_y = np.absolute(pos[:,1]) <= y_list[i]
#            title = 'All (%d<|y|<%d)' %(0, y_list[i])
#            w     = False
#        else:#here range of random y is [y_list[i-1], y_list[i]]---
#            low_y = np.absolute(pos[:,1]) <= y_list[i]
#            upp_y = np.absolute(pos[:,1]) >  y_list[i-1]
#            title = '%d<|y|<%d' %(y_list[i-1], y_list[i])
#            
#        image_diff = pre[cut & low_y & upp_y] - pdr[cut & low_y & upp_y]
#        image_true = pdr[cut & low_y & upp_y]
#        visib_diff = np.divide(image_diff, image_true)
#        
#        savehist(visib_diff, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'Y_intensity-'+str(y_list[i]), w)
# 
#        print("\nBig biased values:")
#        Bias = 0
#        for diff in visib_diff:
#            if abs(diff) > 1:
#                print(str(format(diff, '.3f')))
#                Bias += 1
#        print("# of abs((emul-simu)/simu)>1 : ", Bias)
#
#    print('---------------------------------------------------------')
#
#    print('\n\n\n------------------- Scan along Z------------------------')
#    print('Intensity and resolution evaluating...')           
#    #Shu: 630 will not work---
#    z_list = [-270, -180, -90, 0, 90, 180, 270, 360, 450, 540, 630]
#    for i in range(len(z_list)):
#        w = True
#        if i == 0:#here range of random z is [-280, -270]---
#            low_z = pos[:,2] >  -280#the z position of the vertex---
#            upp_z = pos[:,2] <= z_list[i]
#            title = '-280<z<%d' %(z_list[i])
#        elif i == (len(z_list)-1):#here range of random z is [1, z_list[i-1]]---
#            low_z = pos[:,2] >  -280
#            upp_z = pos[:,2] <= z_list[i]
#            title = 'All (-280<z<%d)' %(z_list[i])
#            w     = False
#        else:#here range of random z is [z_list[i-1], z_list[i]]---
#            low_z = pos[:,2] <= z_list[i]
#            upp_z = pos[:,2] >  z_list[i-1]
#            title = '%d<z<%d' %(z_list[i-1], z_list[i])
#            
#        image_diff = pre[cut & low_z & upp_z] - pdr[cut & low_z & upp_z]
#        image_true = pdr[cut & low_z & upp_z]
#        visib_diff = np.divide(image_diff, image_true)
#        
#        savehist(visib_diff, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'Z_intensity-'+str(z_list[i]), w)
#
#        print("\nBig biased values:")
#        Bias = 0
#        for diff in visib_diff:
#            if abs(diff) > 1:
#                print(str(format(diff, '.3f')))
#                Bias += 1
#        print("# of abs((emul-simu)/simu)>1 : ", Bias)
#
#    print('---------------------------------------------------------')
#
##=====================================================================




#This need further modifications, Shu, 20231124---
##===Test vertex of large (emul-simu)/simu=============================
#    print("Total points: ", len(visib_diff))
#    for num in range(0, len(visib_diff)):
##        if visib_diff[num] < -0.25:
#        if num % 200 == 0: 
#            pos_X.append(pos[num, 0])
#            pos_Y.append(pos[num, 1])
#            pos_Z.append(pos[num, 2])

    #y and x---
#    plt.scatter(pos_Y, pos_X, c='red', s=1)
#    plt.xlabel('Y')
#    plt.ylabel('X')
#    plt.grid()
#    plt.savefig('dis_Y_X.png', dpi=200)
#    #y and z---
#    plt.scatter(pos_Y, pos_Z, c='red', s=1)
#    plt.xlabel('Y')
#    plt.ylabel('Z')
#    plt.grid()
#    plt.savefig('dis_Y_Z.png', dpi=200)
#    #x and z---
#    plt.scatter(pos_X, pos_Z, c='red', s=1)
#    plt.xlabel('X')
#    plt.ylabel('Z')
#    plt.grid()
#    plt.savefig('dis_X_Z.png', dpi=200)
#    #x, y and z---
#    figxyz = plt.figure()
#    ax = Axes3D(figxyz)
#    ax.scatter(pos_X, pos_Y, pos_Z)    
#    plt.savefig('dis_X_Y_Z.png', dpi=200)

    #keep x, y, z in  txt files---
#    with open('x_position.txt', 'w') as filehandle:
#        for listitem in pos_X:
#            filehandle.write('%d\n' % listitem)

#    with open('y_position.txt', 'w') as filehandle:
#        for listitem in pos_Y:
#            filehandle.write('%d\n' % listitem)
#
#    with open('z_position.txt', 'w') as filehandle:
#        for listitem in pos_Z:
#            filehandle.write('%d\n' % listitem)

#    print("Length of pos_X: ", len(pos_X))
#    print("pos[:, 0] ", pos[:, 0])
#    print("pos[:, 1] ", pos[:, 1])
#    print("pos[:, 2] ", pos[:, 2])
#    print("---------------------------------------------------")

#=================================================================






#===test single Optical Channel==================================
    #This is in fact the whole space---
    cut_z = (pos[:,2] > -277.75) & (pos[:,2] < 577.05)
    cut_y = (pos[:,1] > -427.4) & (pos[:,1] < 427.4)
    coor_x = pos[:,0][cut_z & cut_y]
    true_x = pdr[cut_z & cut_y]
    emul_x = pre[cut_z & cut_y]
   
    num_x = len(coor_x)
    true_x_s = np.zeros(shape=(num_x, 40))
    emul_x_s = np.zeros(shape=(num_x, 40))

    for index, op in zip(range(0, 40), range(0, 40)):
        true_x_s[:,index] = true_x[:,op]
        emul_x_s[:,index] = emul_x[:,op]
    
#    for label in range (0, 40):
    label = 38 #opCh label---
    hist = np.divide((emul_x_s[:, label] - true_x_s[:, label]), true_x_s[:, label])
    title = 'Response of OpCh %d' %(label)
    fileName = 'OpCh_%d' %(label)
    #The range here is (-1, 1), absolute values larger than 1 is discarded!
    savehist(hist, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, fileName, True)

    print("\nBig biased values:")
    Bias = 0
    for diff in hist:
        if abs(diff) > 1:
            print(str(format(diff, '.3f')))
            Bias += 1
    print("# of abs((emul-simu)/simu)>1 : ", Bias)

    #for num in range(0, len(hist)):
    #    if np.absolute(pos[num,0])<300 and np.absolute(pos[num,1])<300 and pos[num,2]>0 and pos[num,2]<300:
    #        if num % 1500 == 0:
    #            print('Value: ', format(hist[num], '.2f'), "; Pos: (%d, %d, %d)"%(pos[num,0],pos[num,1],pos[num,2]))
    #        opCh.append(hist[num])
    #savehist(opCh, (-100, 100), '(Emul-Simu)/Simu', 'Counts', title, fileName, True)

    print("\nTotal points (Original)   : ", len(hist))
    #print("Total points (Constrained): ", len(opCh))
#================================================================




#To test single XArapuca-------------------------------------------
#This time explore the averaged performance for certain track------
#    cut_z = (pos[:,0] > 140) & (pos[:,0] < 150)
#    cut_y = (pos[:,1] > -5) & (pos[:,1] < 5)
#    coor_x = pos[:,2][cut_z & cut_y]
#    true_x = pdr[cut_z & cut_y]
#    emul_x = pre[cut_z & cut_y]
   
#    num_x = len(coor_x)
#    true_x_s = np.zeros(shape=(num_x, 16))
#    emul_x_s = np.zeros(shape=(num_x, 16))

#    for index, op in zip(range(0, 16), range(0, 16)):
#        true_x_s[:,index] = true_x[:,op]
#        emul_x_s[:,index] = emul_x[:,op]

#    print("\n------Test single XArapuca-------------------")
#    print("Scan along x direction:")
#    print("Size of num_x: ", len(coor_x))
#    list_coor_x = list(coor_x)
#    print("X coordinate: \n", list_coor_x)

#    print("\nShape of true_x:   ", true_x.shape)
#    print("Shape of true_x_s: ", true_x_s.shape)

#    SingleXA_sum_emul = []
#    SingleXA_sum_true = []
#    SingleXA_diff = []
#    SingleXA_sum_emul = emul_x_s.sum(axis=0)*1000000
#    SingleXA_sum_emul = np.floor(SingleXA_sum_emul)
#    SingleXA_sum_true = true_x_s.sum(axis=0)*1000000
#    SingleXA_sum_true = np.floor(SingleXA_sum_true)
#    for num in range(len(SingleXA_sum_emul)):
#        value = (SingleXA_sum_emul[num] - SingleXA_sum_true[num])/SingleXA_sum_true[num]
#        SingleXA_diff.append(float(format(value, '.4f')))

#    list_emul = list(SingleXA_sum_emul)
#    list_simu = list(SingleXA_sum_true)
#    list_diff = list(SingleXA_diff)

#    print("\nSize of SingleXA_sum_emul: ", len(SingleXA_sum_emul))
#    print("SingleXA_sum_emul:\n", list_emul)
#    print("SingleXA_sum_true:\n", list_simu)  
#    print("SingleXA_diff:\n", list_diff)  


#    label = 7 #XA label---
#    hist = np.divide((emul_x_s[:, label] - true_x_s[:, label]), true_x_s[:, label])
#    title = 'Response of XArapuca %d' %(label)
#    fileName = 'XA_%d' %(label)
#    savehist(hist, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, fileName, True)
            
#    for num in range(0, len(hist)):
#        if np.absolute(pos[num,0])<300 and np.absolute(pos[num,1])<300 and pos[num,2]>0 and pos[num,2]<300:
#            if num % 1500 == 0:
#                print('Value: ', format(hist[num], '.2f'), "; Pos: (%d, %d, %d)"%(pos[num,0],pos[num,1],pos[num,2]))
#            XA.append(hist[num])
#    savehist(XA, (-100, 100), '(Emul-Simu)/Simu', 'Counts', title, fileName, True)

#    print("\nTotal points (Original)   : ", len(hist))
#    print("Total points (Constrained): ", len(XA))
#---------------------------------------------------------












#Not so useful--------------------------------------------
def freezemodel(modpath):
    fname = modpath+'best_model.h5' #name of the saved model
    K.set_learning_phase(0)         #this line must be executed before loading Keras model.
    
    print('Loading model from file: '+fname)    
    model = load_model(fname, compile=False)
    print(model.outputs)
    print(model.inputs)
    
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    
    tf.train.write_graph(frozen_graph, modpath, 'graph.pb', as_text=False)

def debug(dim_pdr, mtier, opt):    
    print('TensorFlow version: ' + tf.version.VERSION)
    
    if dim_pdr == 90:
        if mtier == 0:
            print('Loading ProtoDUNE t0 net...')
            model = model_protodunev7_t0(dim_pdr)
        elif mtier == 1:
            print('Loading ProtoDUNE t1 net...')
            model = model_protodunev7_t1(dim_pdr)
        elif mtier == 2:
            print('Loading ProtoDUNE t2 net...')
            model = model_protodunev7_t2(dim_pdr)
        elif mtier == 3:
            print('Loading ProtoDUNE t3 net...')
            model = model_protodunev7_t3(dim_pdr)
    elif dim_pdr == 480:
        if mtier == 0:
            print('Loading DUNE t0 net...')
            model = model_dune10kv4_t0(dim_pdr)
        elif mtier == 1:
            print('Loading DUNE t1 net...')
            model = model_dune10kv4_t1(dim_pdr)
        elif mtier == 2:
            print('Loading DUNE t2 net...')
            model = model_dune10kv4_t2(dim_pdr)
        elif mtier == 3:
            print('Loading DUNE t3 net...')
            model = model_dune10kv4_t3(dim_pdr)
    elif dim_pdr == 168:
        if mtier == 0:
            print('Loading VD t0 net...')
            model = model_dunevd_t0(dim_pdr)
        if mtier == 1:
            print('Loading VD t1 net...')
            model = model_dunevd_t1(dim_pdr)
    elif dim_pdr == 160:
        if mtier == 0:
            print('Loading ProtoDUNEHD t0 net...')
            model = model_protodunehd_t0(dim_pdr)        
    #Suggested by Mu, for module 0, 20230125---
    elif dim_pdr == 16:
        if mtier == 0:
            print('Loading module 0 16op net...')
            model = model_dunevd_16op(dim_pdr) 
    #Added by Shu, for protodunevd_v4 40 opch, 20231124---
    elif dim_pdr == 40:
        if mtier == 0:
            print('Loading protodunevd_v4 40op net...')
            model = model_protodunevd_v4(dim_pdr)            

    if opt == 'SGD':
        optimizer = SGD(momentum=0.9)
    else:
        optimizer = Adam()
        
    model.compile(optimizer=optimizer, loss=vkld_loss, metrics=['mape', 'mae'])

