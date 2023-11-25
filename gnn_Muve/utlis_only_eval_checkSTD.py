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
pos_X2 = []
pos_Y2 = []
pos_Z2 = []
#-----------------------------------
XA = []

large = 23
med   = 17
small = 9
fontax = {'family': 'sans-serif',
          'color':  'black',
          'weight': 'bold',
          'size': 10,
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
          'xtick.labelsize': 9,
          'ytick.labelsize': 9,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          }
plt.style.use('seaborn-paper')
plt.rcParams.update(params)
#['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight',
# 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette',
# 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel',
#'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']

def mkdir(dir="image/"):
       if not os.path.exists(dir):
           print('make directory '+str(dir))
           os.makedirs(dir)

def save_plot(fake, true, nbin, xlabel, ylable, name):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1,1,1)

    ax.bar(nbin,height=true,color='blue',width=1, alpha=0.5, label='true')
    ax.grid(b=True)
    
    plt.scatter(nbin, fake, color='red', s=3, label='fake')
    plt.legend()
    plt.savefig(name + '-lin.png', dpi=200)
    plt.yscale('log')
    plt.legend()
    plt.savefig(name + '-log.png', dpi=200)    
    plt.clf()
    plt.close()
           
def savescatter(xlable, ylable, ylim, name):
    plt.legend(loc='upper left', prop=fontlg)
    plt.grid(color='0.9', linestyle='-.', linewidth=0.5)
    
    plt.xlabel(xlable, fontdict=fontax)
    plt.ylabel(ylable, fontdict=fontax)
    
    plt.ylim(0, ylim)    
    plt.savefig(name+'.png')
    plt.clf()
    
def savehist(hist, range, xlabel, ylable, title, name, w):        
    if w:
        plt.rcParams['figure.figsize'] = (12, 9)
    else:
        plt.rcParams['figure.figsize'] = (12, 9)
    plt.rcParams['figure.figsize'] = (12, 9)
    
    mu            = np.mean(hist)
    sigma         = np.std(hist)
    bins          = 400 #remark: when draw the [-100,100] histo, even though errors, still work!!!
    fig, ax       = plt.subplots()
    h1d, nbins, _ = ax.hist(hist, bins=bins, range=range, histtype='step', color='navy', linewidth=2)
    
    x     = (nbins[1:] + nbins[:-1])/2
    stats =  fit_gaussian(x, h1d)
    
#---To test the sigma------------------------
    Weights    = h1d/np.sum(h1d)
    Ini_mu     = np.average(x, weights = Weights)
    Ini_sigma  = np.sqrt(np.average(x**2, weights = Weights) - Ini_mu**2)
#    print("mu= ", mu, ", sigma = ", sigma, ";;;;; Ini_mu = ", Ini_mu, ", Ini_sigma = ", Ini_sigma)
#----------------------------------------------



    sigus  = int(bins*(stats[1]+abs(stats[2]+1))/2)
    sigls  = int(bins*(stats[1]-abs(stats[2]+1))/2)
    sigs   = sum(h1d[sigls:sigus])
    
    sigup  = int(bins*(stats[1]+1.1)/2)
    siglp  = int(bins*(stats[1]+0.9)/2)
    sigp   = sum(h1d[siglp:sigup])

    tots   = sum(h1d[1:bins])

   #added by Shuaixiang (Shu), Mar 29, 2023---
    sigup15 = int(bins*(stats[1]+1.15)/2)
    siglp15 = int(bins*(stats[1]+0.85)/2)
    sigp15  = sum(h1d[siglp15:sigup15])

    stat_point1 = sum(h1d[19990:20010])
    stat_point15 = sum(h1d[19985:20015])
    stat_point20 = sum(h1d[19980:20020])
    stat_point50 = sum(h1d[19950:20050])
    stat_1 = sum(h1d[19900:20100])
    stat_5 = sum(h1d[19500:20500])
    stat_10 = sum(h1d[19000:21000])
    stat_50 = sum(h1d[15000:25000])
    stat_100 = sum(h1d[10000:30000])        

    print("--------------- ", title, " ---------------")
    print('Total vertexs: '+str(tots))
#    print('(Fitting) Vertexs in 1 sigma: '+str(sigs)+', '+str(format(sigs/tots, '.3f')))
#    print('(Fitting) Vertexs in 10%: '+str(sigp)+', '+str(format(sigp/tots, '.3f'))+
#          ', outside 10%: '+str(format(1-sigp/tots, '.3f')))
#    print('(Fitting) Vertexs in 15%: '+str(sigp15)+', '+str(format(sigp15/tots, '.3f'))+
#          ', outside 15%: '+str(format(1-sigp15/tots, '.3f')))

    print('Raw::     Mean: '+str(format(mu, '.4f'))+'; Sigma: '+str(format(sigma, '.4f')))
    print('Hist::    Mean: '+str(format(Ini_mu, '.4f'))+'; Sigma: '+str(format(Ini_sigma, '.4f')))
    print('Fitting:: Mean: '+str(format(stats[1], '.4f'))+'; Sigma: '+str(format(stats[2], '.4f')))

    print('\nInside [-0.1, 0.1]  : '+str(format(stat_point1/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_point1)+',  '+str(format(1-stat_point1/tots, '.4f')))
    print('Inside [-0.15, 0.15]: '+str(format(stat_point15/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_point15)+',  '+str(format(1-stat_point15/tots, '.4f')))
    print('Inside [-0.20, 0.20]: '+str(format(stat_point20/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_point20)+',  '+str(format(1-stat_point20/tots, '.4f')))
    print('Inside [-0.50, 0.50]: '+str(format(stat_point50/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_point50)+',  '+str(format(1-stat_point50/tots, '.4f')))
    print('Inside [-1, 1]:       '+str(format(stat_1/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_1)+',  '+str(format(1-stat_1/tots, '.4f')))
    print('Inside [-5, 5]:       '+str(format(stat_5/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_5)+',  '+str(format(1-stat_5/tots, '.4f')))
    print('Inside [-10, 10]:     '+str(format(stat_10/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_10)+',  '+str(format(1-stat_10/tots, '.4f')))
    print('Inside [-50, 50]:     '+str(format(stat_50/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_50)+',  '+str(format(1-stat_50/tots, '.4f')))
    print('Inside [-100, 100]:   '+str(format(stat_100/tots, '.4f'))+
          ';   Outside: '+str(tots - stat_100)+',  '+str(format(1-stat_100/tots, '.4f')))




    x_int = np.linspace(nbins[0], nbins[-1], 1000)
    ax.plot(x_int, gaussian(x_int, stats[0], stats[1], stats[2]), color='red', 
            linestyle='-.', linewidth=1.5, label='$\mu=%.3f,\ \sigma=%.3f$' %(stats[1], abs(stats[2])))
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
#    ax.legend(loc='upper left', handles=handles, labels=labels, prop=fontlg)

    plt.legend(['$\mu$ = {}, SD = {}'.format(format(Ini_mu, '.3f'), format(Ini_sigma, '.3f')), 
                '$\mu$ = {}, $\sigma$ = {}'.format(format(stats[1], '.3f'), format(stats[2], '.3f'))], 
                loc='upper left', prop={'size':30})

    plt.xlabel(xlabel, fontdict=fontax)
    plt.ylabel(ylable, fontdict=fontax)

    plt.yscale('log')
    plt.ylim(1, 200000)

#    plt.text(-1.0, 1200, "Raw data:")
#    plt.text(-1.0, 1000, "$\mu$ = %.3f"%(mu))
#    plt.text(-1.0, 800, "$\sigma$ = %.3f"%(sigma))

    plt.title(title, fontdict={'weight':'normal','size':30})
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
    
def eval_protodune(pos, pdr, pre, evlpath):                
    print('Behavior testing for protoDUNE PDS...')
    
    cut_x  = (pos[:,0] > 130) & (pos[:,0] < 170)
    cut_y  = (pos[:,1] > 390) & (pos[:,1] < 450)
    
    coor_z = pos[:,2][cut_x & cut_y]
    true_z = pdr[cut_x & cut_y]
    emul_z = pre[cut_x & cut_y]
    
    cut_z  = (pos[:,2] > 330) & (pos[:,2] < 360)
    cut_x  = (pos[:,0] > 130) & (pos[:,0] < 170)

    coor_y = pos[:,1][cut_z & cut_x]
    true_y = pdr[cut_z & cut_x]
    emul_y = pre[cut_z & cut_x]
    
    cut_y  = (pos[:,1] > 390) & (pos[:,1] < 450)
    cut_z  = (pos[:,2] > 330) & (pos[:,2] < 360)
    coor_x = pos[:,0][cut_y & cut_z]            
    true_x = pdr[cut_y & cut_z]    
    emul_x = pre[cut_y & cut_z]
    
    num_x = len(coor_x)        
    num_y = len(coor_y)
    num_z = len(coor_z)
    
    print('Scan Z with ' + str(num_z) + ' points.')
    opch_f = ['PD 24', 'PD 44', 'PD 43', 'PD 42', 'PD 41', 'PD 40', 'PD 39', 'PD 38', 'PD 37',
              'PD 36', 'PD 35', 'PD 34', 'PD 33', 'PD 32', 'PD 31', 'PD 30', 'PD 29', 'PD 05']
    opch_s = ['PD 24', 'PD sum 44-29',   'PD 05']
    colors = ['black', 'navy',  'red',  'green', 'indigo', 'pink',  'orange', 'magenta', 'purple',
              'brown', 'violet','cyan', 'tomato', 'aquamarine', 'maroon', 'orchid', 'turquoise', 'chocolate']              
    true_z_f = np.zeros(shape=(num_z, 18))
    true_z_s = np.zeros(shape=(num_z, 3))
    
    emul_z_f = np.zeros(shape=(num_z, 18))
    emul_z_s = np.zeros(shape=(num_z, 3))

    true_z_f[:,0] = true_z[:,24]
    emul_z_f[:,0] = emul_z[:,24]
    for index, op in zip(range(1, 17), range(44, 28, -1)):
        true_z_f[:,index] = true_z[:,op]
        emul_z_f[:,index] = emul_z[:,op]
    true_z_f[:,17] = true_z[:,5]
    emul_z_f[:,17] = emul_z[:,5]
    
    true_z_s[:,0] = true_z[:,24]
    emul_z_s[:,0] = emul_z[:,24]
    sum_true_z = np.zeros(shape=(num_z, 1))
    sum_emul_z = np.zeros(shape=(num_z, 1))
    for op in range(44, 28, -1):
        sum_true_z[:,0] = sum_true_z[:,0] + true_z[:,op]
        sum_emul_z[:,0] = sum_emul_z[:,0] + emul_z[:,op]
    true_z_s[:,1] = sum_true_z[:,0]
    emul_z_s[:,1] = sum_emul_z[:,0]
    true_z_s[:,2] = true_z[:,5]
    emul_z_s[:,2] = emul_z[:,5]
    
    ylim = 0.004
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, 3):
        plt.scatter(coor_z, true_z_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch_s[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'true_z_s')
    
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, 3):
        plt.scatter(coor_z, emul_z_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch_s[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'emul_z_s')
    
    ylim = 0.0025
    for i in range(1, 17, 6):
        plt.scatter(coor_z, true_z_f[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch_f[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'true_z_ar')
    
    for i in range(1, 17, 6):
        plt.scatter(coor_z, emul_z_f[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch_f[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'emul_z_ar')
          
    print('Scan Y with ' + str(num_y) + ' points.')
    opch_s = ['PD 10', 'PD 11', 'PD 12', 'PD 13', 'PD 14', 'PD sum 44-29', 'PD 15', 'PD 16', 'PD 17', 'PD 18']
    colors = ['red',   'blue',  'black', 'indigo', 'navy', 'magenta',     'purple', 'pink',  'orange', 'brown',
              'aquamarine', 'cyan', 'turquoise', 'tomato', 'orchid', 'chocolate', 'violet',  'maroon']
    
    true_y_s = np.zeros(shape=(num_y, 10))
    for index, op in zip(range(0, 5), range(10, 15)):
        true_y_s[:,index] = true_y[:,op]
    sum_true_y = np.zeros(shape=(num_y, 1))
    for op in range(44, 28, -1):
        sum_true_y[:,0] = sum_true_y[:,0] + true_y[:,op]
    true_y_s[:,5] = sum_true_y[:,0]
    for index, op in zip(range(6, 10), range(15, 19)):
        true_y_s[:,index] = true_y[:,op]
            
    emul_y_s = np.zeros(shape=(num_y, 10))
    for index, op in zip(range(0, 5), range(10, 15)):
        emul_y_s[:,index] = emul_y[:,op]
    sum_emul_y = np.zeros(shape=(num_y, 1))
    for op in range(44, 28, -1):
        sum_emul_y[:,0] = sum_emul_y[:,0] + emul_y[:,op]
    emul_y_s[:,5] = sum_emul_y[:,0]
    for index, op in zip(range(6, 10), range(15, 19)):
        emul_y_s[:,index] = emul_y[:,op]
        
    ylim = 0.01
    for i in range(0, 10, 4):
        plt.scatter(coor_y, true_y_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch_s[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'true_y_s')
            
    for i in range(0, 10, 4):
        plt.scatter(coor_y, emul_y_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch_s[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'emul_y_s')
    
    print('Scan X with ' + str(num_x) + ' points.')
    true_x_s = np.zeros(shape=(num_x, 10))
    for index, op in zip(range(0, 5), range(10, 15)):
        true_x_s[:,index] = true_x[:,op]
    sum_true_x = np.zeros(shape=(num_x, 1))
    for op in range(44, 28, -1):
        sum_true_x[:,0] = sum_true_x[:,0] + 0
    true_x_s[:,5] = sum_true_x[:,0]
    for index, op in zip(range(6, 10), range(15, 19)):
        true_x_s[:,index] = true_x[:,op]
            
    emul_x_s = np.zeros(shape=(num_x, 10))
    for index, op in zip(range(0, 5), range(10, 15)):
        emul_x_s[:,index] = emul_x[:,op]
    sum_emul_x = np.zeros(shape=(num_x, 1))
    for op in range(44, 28, -1):
        sum_emul_x[:,0] = sum_emul_x[:,0] + 0
    emul_x_s[:,5] = sum_emul_x[:,0]
    for index, op in zip(range(6, 10), range(15, 19)):
        emul_x_s[:,index] = emul_x[:,op]
        
    ylim = 0.02
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, 10, 4):
        plt.scatter(coor_x, true_x_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch_s[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'true_x_s')
    
    plt.locator_params(axis="y", nbins=5)        
    for i in range(0, 10, 4):
        plt.scatter(coor_x, emul_x_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch_s[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'emul_x_s')

def eval_dune(pos, pdr, pre, evlpath):                
    print('Behavior testing for DUNE PDS...')            
    cut_x  = (pos[:,0] > 130) & (pos[:,0] < 170)
    cut_y  = (pos[:,1] > 270) & (pos[:,1] < 330)
    coor_z = pos[:,2][cut_x & cut_y]
    true_z = pdr[cut_x & cut_y]
    emul_z = pre[cut_x & cut_y]
    
    cut_z  = (pos[:,2] > 760) & (pos[:,1] < 800)
    cut_x  = (pos[:,0] > 130) & (pos[:,0] < 170)   
    coor_y = pos[:,1][cut_z & cut_x]
    true_y = pdr[cut_z & cut_x]
    emul_y = pre[cut_z & cut_x]
    
    cut_y  = (pos[:,1] > 270) & (pos[:,1] < 330)
    cut_z  = (pos[:,2] > 760) & (pos[:,1] < 800)    
    coor_x = pos[:,0][cut_y & cut_z]            
    true_x = pdr[cut_y & cut_z]    
    emul_x = pre[cut_y & cut_z]
    
    num_x = len(coor_x)        
    num_y = len(coor_y)
    num_z = len(coor_z)
    
    print('Scan Z with ' + str(num_z) + ' points.')
    opch   = ['PD 005', 'PD 015', 'PD 025', 'PD 035', 'PD 045', 'PD 055', 'PD 065', 'PD 075',  'PD 085',
              'PD 095', 'PD 105', 'PD 115', 'PD 125', 'PD 135', 'PD 145', 'PD 155', 'PD 165',  'PD 175',
              'PD 185', 'PD 195', 'PD 205', 'PD 215', 'PD 225', 'PD 235']
    colors = ['blue',   'cyan',   'black',  'green',  'indigo', 'magenta', 'pink',  'violet',   'purple',
              'red',    'plum',   'bisque', 'peru',   'tomato', 'navy',   'chocolate','orange', 'maroon',
              'black',  'lime',   'orchid',   'gold',   'olive',  'turquoise']
              
    num_op   = len(opch)          
    true_z_s = np.zeros(shape=(num_z, num_op))    
    emul_z_s = np.zeros(shape=(num_z, num_op))

    for index, op in zip(range(0, num_op), range(5, 245, 10)):
        true_z_s[:,index] = true_z[:,op]
        emul_z_s[:,index] = emul_z[:,op]
        
    ylim = 0.004
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, num_op, 9):
        plt.scatter(coor_z, true_z_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'true_z_s')
    
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, num_op, 9):
        plt.scatter(coor_z, emul_z_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'emul_z_s')
    
    print('Scan Y with ' + str(num_y) + ' points.')
    opch   = ['PD 100', 'PD 101', 'PD 102', 'PD 103', 'PD 104', 'PD 105', 'PD 106',
              'PD 107', 'PD 108', 'PD 109', 'PD 340', 'PD 341', 'PD 342', 'PD 343',
              'PD 344', 'PD 345', 'PD 346', 'PD 347', 'PD 348', 'PD 349']
    colors = ['green',  'orange', 'black',  'cyan',  'indigo', 'magenta', 'pink',
              'navy',   'purple', 'red',    'plum',   'bisque', 'peru',   'tomato',
              'red',    'maroon', 'orange', 'brown',  'lime',   'orchid',   'gold',   'olive',  'turquoise']
            
    num_op   = len(opch)
    true_y_s = np.zeros(shape=(num_y, num_op))
    emul_y_s = np.zeros(shape=(num_y, num_op))

    for index, op in zip(range(0, 10), range(100, 110)):
        true_y_s[:,index] = true_y[:,op]
        emul_y_s[:,index] = emul_y[:,op]
        
    for index, op in zip(range(10, 20), range(340, 350)):
        true_y_s[:,index] = true_y[:,op]
        emul_y_s[:,index] = emul_y[:,op]

    ylim = 0.005
    for i in range(0, num_op, 7):
        plt.scatter(coor_y, true_y_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'true_y_s')    
    
    for i in range(0, num_op, 7):
        plt.scatter(coor_y, emul_y_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'emul_y_s')
    
    print('Scan X with ' + str(num_x) + ' points.')
    true_x_s = np.zeros(shape=(num_x, num_op))
    emul_x_s = np.zeros(shape=(num_x, num_op))
    
    for index, op in zip(range(0, 10), range(100, 110)):
        true_x_s[:,index] = true_x[:,op]
        emul_x_s[:,index] = emul_x[:,op]
    for index, op in zip(range(10, 20), range(340, 350)):
        true_x_s[:,index] = true_x[:,op]
        emul_x_s[:,index] = emul_x[:,op]
        
    ylim = 0.002
    for i in range(0, num_op, 7):
        plt.scatter(coor_x, true_x_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'true_x_s')
    
    for i in range(0, num_op, 7):
        plt.scatter(coor_x, emul_x_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'emul_x_s')

def eval_dunevd(pos, pdr, pre, evlpath):                
    print('Behavior testing for DUNEVD PDS...')            
    cut_x  = (pos[:,0] > 130) & (pos[:,0] < 170)
    cut_y  = (pos[:,1] > 390) & (pos[:,1] < 450)
    
    coor_z = pos[:,2][cut_x & cut_y]
    true_z = pdr[cut_x & cut_y]
    emul_z = pre[cut_x & cut_y]
    
    cut_z  = (pos[:,2] > 330) & (pos[:,2] < 360)
    cut_x  = (pos[:,0] > 130) & (pos[:,0] < 170)

    coor_y = pos[:,1][cut_z & cut_x]
    true_y = pdr[cut_z & cut_x]
    emul_y = pre[cut_z & cut_x]
    
    cut_y  = (pos[:,1] > 390) & (pos[:,1] < 450)
    cut_z  = (pos[:,2] > 330) & (pos[:,2] < 360)
    coor_x = pos[:,0][cut_y & cut_z]            
    true_x = pdr[cut_y & cut_z]    
    emul_x = pre[cut_y & cut_z]
    
    num_x = len(coor_x)        
    num_y = len(coor_y)
    num_z = len(coor_z)
    
    print('Scan Z with ' + str(num_z) + ' points.')
    opch   = ['PD 028', 'PD 029', 'PD 030', 'PD 031', 'PD 032', 'PD 033', 'PD 034', 'PD 035',  'PD 036',
              'PD 037', 'PD 038', 'PD 039', 'PD 040', 'PD 041']
    colors = ['blue',   'cyan',   'black',  'green',  'indigo', 'magenta', 'green',  'violet',   'purple',
              'red',    'plum',   'bisque', 'red',   'tomato', 'navy',   'chocolate','orange', 'maroon',
              'black',  'lime',   'orchid', 'gold',   'olive',  'turquoise', 'pink',  'violet',   'purple',
              'pink']
              
    num_op   = len(opch)          
    true_z_s = np.zeros(shape=(num_z, num_op))    
    emul_z_s = np.zeros(shape=(num_z, num_op))

    for index, op in zip(range(0, num_op), range(28, 41, 1)):
        true_z_s[:,index] = true_z[:,op]
        emul_z_s[:,index] = emul_z[:,op]
        
    ylim = 0.00015
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, num_op, 6):
        plt.scatter(coor_z, true_z_s[:,i], marker='.', s=500, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'true_z_s')
    
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, num_op, 6):
        plt.scatter(coor_z, emul_z_s[:,i], marker='.', s=500, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'emul_z_s')
    
    print('Scan Y with ' + str(num_y) + ' points.')
    opch   = ['PD 104', 'PD 105', 'PD 106', 'PD 107', 'PD 108', 'PD 109', 'PD 110',
              'PD 111', 'PD 112', 'PD 113', 'PD 114', 'PD 115', 'PD 116', 'PD 117',
              'PD 118', 'PD 119']
    colors = ['red',  'orange', 'black',  'cyan',  'indigo', 'magenta', 'blue',
              'navy',   'purple', 'red',    'plum',   'bisque', 'green',   'tomato',
              'red',    'maroon', 'orange', 'brown',  'lime',   'red',   'gold',   'olive',  'turquoise']
            
    num_op   = len(opch)
    true_y_s = np.zeros(shape=(num_y, num_op))
    emul_y_s = np.zeros(shape=(num_y, num_op))

    for index, op in zip(range(0, num_op), range(104, 117, 1)):
        true_y_s[:,index] = true_y[:,op]
        emul_y_s[:,index] = emul_y[:,op]
        
    ylim = 0.0015
    for i in range(0, num_op, 6):
        plt.scatter(coor_y, true_y_s[:,i], marker='.', s=500, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'tre_y_s')    
    
    for i in range(0, num_op, 6):
        plt.scatter(coor_y, emul_y_s[:,i], marker='.', s=500, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'emul_y_s')
    
    print('Scan X with ' + str(num_x) + ' points.')
    opch   = ['PD 006', 'PD 020', 'PD 034', 'PD 048']
    colors = ['red',    'green',  'blue',   'brown',  'indigo', 'magenta', 'pink',
              'navy',   'purple', 'red',    'plum',   'bisque', 'peru',   'tomato',
              'red',    'maroon', 'orange', 'brown',  'lime',   'orchid',   'gold',   'olive',  'turquoise']
            
    num_op   = len(opch)
    true_x_s = np.zeros(shape=(num_x, num_op))
    emul_x_s = np.zeros(shape=(num_x, num_op))
    
    for index, op in zip(range(0, num_op), range(6, 48, 14)):
        true_x_s[:,index] = true_x[:,op]
        emul_x_s[:,index] = emul_x[:,op]
        
    ylim = 0.00012
    for i in range(0, num_op, 1):
        plt.scatter(coor_x, true_x_s[:,i], marker='.', s=500, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'true_x_s')
    
    for i in range(0, num_op, 1):
        plt.scatter(coor_x, emul_x_s[:,i], marker='.', s=500, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'emul_x_s')


def eval_protodunehd(pos, pdr, pre, evlpath):                
    print('Behavior testing for ProtoDUNEHD PDS...')            
    cut_x  = (pos[:,0] > 100) & (pos[:,0] < 180)
    cut_y  = (pos[:,1] > 250) & (pos[:,1] < 270)
    coor_z = pos[:,2][cut_x & cut_y]
    true_z = pdr[cut_x & cut_y]
    emul_z = pre[cut_x & cut_y]
    
    cut_z  = (pos[:,2] > 190) & (pos[:,1] < 200)
    cut_x  = (pos[:,0] > 135) & (pos[:,0] < 140)
    coor_y = pos[:,1][cut_z & cut_x]
    true_y = pdr[cut_z & cut_x]
    emul_y = pre[cut_z & cut_x]
    
    cut_y  = (pos[:,1] > 200) & (pos[:,1] < 210)
    cut_z  = (pos[:,2] > 200) & (pos[:,1] < 210)
    coor_x = pos[:,0][cut_y & cut_z]            
    true_x = pdr[cut_y & cut_z]    
    emul_x = pre[cut_y & cut_z]
    
    num_x = len(coor_x)        
    num_y = len(coor_y)
    num_z = len(coor_z)
    
    print('Scan Z with ' + str(num_z) + ' points.')
    opch   = ['PD 19', 'PD 25',  'PD 31',  'PD 37',  'PD 43',  'PD 49',  'PD 55',    'PD 61']
    colors = ['blue',  'cyan',   'black',  'green',  'indigo', 'pink',   'magenta',  'violet']
              
    num_op   = len(opch)          
    true_z_s = np.zeros(shape=(num_z, num_op))    
    emul_z_s = np.zeros(shape=(num_z, num_op))

    for index, op in zip(range(0, num_op), range(19, 67, 6)):
        true_z_s[:,index] = true_z[:,op]
        emul_z_s[:,index] = emul_z[:,op]
        
    ylim = 0.002
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, num_op, 3):
        plt.scatter(coor_z, true_z_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'true_z_s')
    
    plt.locator_params(axis="y", nbins=5)
    for i in range(0, num_op, 3):
        plt.scatter(coor_z, emul_z_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Z [cm]', 'Visibilities', ylim, evlpath+'emul_z_s')
    
    print('Scan Y with ' + str(num_y) + ' points.')
    opch   = ['PD 16', 'PD 17',  'PD 18',  'PD 19',  'PD 20',    'PD 21']
    colors = ['blue',  'cyan',   'green',  'black',  'magenta',  'pink',   'violet']
            
    num_op   = len(opch)
    true_y_s = np.zeros(shape=(num_y, num_op))
    emul_y_s = np.zeros(shape=(num_y, num_op))

    for index, op in zip(range(0, num_op), range(16, 22)):
        true_y_s[:,index] = true_y[:,op]
        emul_y_s[:,index] = emul_y[:,op]
        
    ylim = 0.001
    for i in range(0, num_op, 2):
        plt.scatter(coor_y, true_y_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'true_y_s')    
    
    for i in range(0, num_op, 2):
        plt.scatter(coor_y, emul_y_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('Y [cm]', 'Visibilities', ylim, evlpath+'emul_y_s')
    
    print('Scan X with ' + str(num_x) + ' points.')
    true_x_s = np.zeros(shape=(num_x, num_op))
    emul_x_s = np.zeros(shape=(num_x, num_op))
    
    for index, op in zip(range(0, num_op), range(16, 22)):
        true_x_s[:,index] = true_x[:,op]
        emul_x_s[:,index] = emul_x[:,op]
        
    ylim = 0.005
    for i in range(0, num_op, 2):
        plt.scatter(coor_x, true_x_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'true_x_s')
    
    for i in range(0, num_op, 2):
        plt.scatter(coor_x, emul_x_s[:,i], marker='.', s=100, alpha=0.5, cmap='viridis', color=colors[i], label=opch[i])
    savescatter('X [cm]', 'Visibilities', ylim, evlpath+'emul_x_s')


#Sugggested by Mu, for module 0, 20230126---
def eval_model_dunevd_16op(pos, pdr, pre, evlpath):                
    print('Behavior testing for Module 0 PDS...')            
    cut_x  = (pos[:,0] > 235) & (pos[:,0] < 245)
    cut_y  = (pos[:,1] > 120) & (pos[:,1] < 130)
    coor_z = pos[:,2][cut_x & cut_y]
    true_z = pdr[cut_x & cut_y]
    emul_z = pre[cut_x & cut_y]
    
    cut_z  = (pos[:,2] > 145) & (pos[:,1] < 150)
    cut_x  = (pos[:,0] > 235) & (pos[:,0] < 240)
    coor_y = pos[:,1][cut_z & cut_x]
    true_y = pdr[cut_z & cut_x]
    emul_y = pre[cut_z & cut_x]
    
    cut_y  = (pos[:,1] > 125) & (pos[:,1] < 130)
    cut_z  = (pos[:,2] > 145) & (pos[:,1] < 150)
    coor_x = pos[:,0][cut_y & cut_z]            
    true_x = pdr[cut_y & cut_z]    
    emul_x = pre[cut_y & cut_z]
    
    num_x = len(coor_x)        
    num_y = len(coor_y)
    num_z = len(coor_z)
    #---Z SCAN---------------
    print('Scan Z with ' + str(num_z) + ' points.')
    #To test------
    print("Output of coor_z:")
    print(coor_z)
    print("========================================")
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
    print('Scan Y with ' + str(num_y) + ' points.')
    #test------
    print("Output of coor_y:")
    print(coor_y)
    print("=========================================")
#    opch   = ['PD 16', 'PD 17',  'PD 18',  'PD 19',  'PD 20',    'PD 21']
#    colors = ['blue',  'cyan',   'green',  'black',  'magenta',  'pink',   'violet']
            
    num_op   = len(opch)
    true_y_s = np.zeros(shape=(num_y, num_op))
    emul_y_s = np.zeros(shape=(num_y, num_op))

    for index, op in zip(range(0, num_op), range(0, 4)):#change range---
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
    print('Scan X with ' + str(num_x) + ' points.')
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
#----------------------------------------------------------------------------------------------------










def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        
        input_graph_def = graph.as_graph_def() # Graph -> GraphDef ProtoBuf
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
    
def lr_scheduler(epoch, lr):
    p = 1000
    d = 0.997
    if (epoch<p) or (epoch%p==0):
        lr = 2e-4
    else:
        lr = lr * d
    return lr
    
def train(pos, pdr, mtier, epochs, batchsize, modpath, opt):
    dim_pdr = pdr.shape[1]
    
    #load different models according to the number of optical channels
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

    if opt == 'SGD':
        optimizer = SGD(momentum=0.9)
    else:
        optimizer = Adam()
        
    model.compile(optimizer=optimizer, loss=vkld_loss, metrics=['mape', 'mae'])

    #if there is existing weight file, load it.
    weight = modpath+'best_model.h5'
    if os.path.isfile(weight):
        model.load_weights(weight)    
        
    checkpoints = [ModelCheckpoint(weight, monitor='loss', verbose=0, save_best_only=True, mode='auto', period=10),
                   LearningRateScheduler(lr_scheduler),
                   EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)
                  ]
                  
    #start model training
    ftrain, ftest, ptrain, ptest=train_test_split(pos, pdr, test_size=0.15)
    model.fit({'pos_x': ftrain[:,0], 'pos_y': ftrain[:,1], 'pos_z': ftrain[:,2]}, ptrain,
               validation_data=({'pos_x': ftest[:,0], 'pos_y': ftest[:,1], 'pos_z': ftest[:,2]}, ptest),
               epochs=epochs, batch_size=batchsize, callbacks=checkpoints, verbose=2, shuffle=True)
               
    #export trained model in SavedModel format for C++ API
    tf.saved_model.save(model, modpath)
    
    #initial evaluation of the training performance
    model.load_weights(weight)
    ypred = model.predict({'pos_x': ftest[:,0], 'pos_y': ftest[:,1], 'pos_z': ftest[:,2]})
    n_test = len(ftest)
    for i in range(0, 5):
        x_ax = range(len(ptest[i]))
        save_plot(ypred[i], ptest[i], x_ax, 'Op Channel', 'visibility', modpath+'/eval_'+str(i))
        
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


##---Case of sum up all XArapucas-------------------------------------------------------------
#    print('\n--------------------Scan along X-------------------------')
#    print('Intensity and resolution evaluating...')            
#    pre = pre.sum(axis=1)#it means suming up all pds; YES, refer to Mu's paper---
#    pdr = pdr.sum(axis=1)
#    
#    cut = (pre != 0) & (pdr != 0)
#    #Shu: 400 will not work---
#    x_list = [50, 100, 150, 200, 250, 300, 350, 400]
#    for i in range(len(x_list)):
#        w = True
#        if i == 0:#here range of random x is [0. 50]---
#            low_x = np.absolute(pos[:,0]) >  1#the x position of the vertex---
#            upp_x = np.absolute(pos[:,0]) <= x_list[i]
#            title = '1<|x|<%d' %(x_list[i])
#        elif i == (len(x_list)-1):#here range of random x is [1, x_list[i-1]]---
#            low_x = np.absolute(pos[:,0]) >  1
#            upp_x = np.absolute(pos[:,0]) <= x_list[i-1]
#            title = '%d<|x|<%d' %(1, x_list[i-1])
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
#        savehist(visib_diff, (-200, 200), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'X_intensity-'+str(x_list[i]), w) 
#    print('--------------------Scan along X-------------------------')
##        print("Length of image_diff: ", len(image_diff))
##        print("iamge_diff: \n",image_diff)
##---------------------------------------------------------------------
#
#
#
##---Scan along y and z axis-------------------------------------------
#    print('\n------------------- Scan along Y------------------------')
#    print('Intensity and resolution evaluating...')            
#    
#    cut = (pre != 0) & (pdr != 0)
#    #Shu: 400 will not work---
#    y_list = [60, 120, 180, 240, 300, 360, 420, 480]
#    for i in range(len(y_list)):
#        w = True
#        if i == 0:#here range of random y is [0. 60]---
#            low_y = np.absolute(pos[:,1]) >  1#the y position of the vertey---
#            upp_y = np.absolute(pos[:,1]) <= y_list[i]
#            title = '1<|y|<%d' %(y_list[i])
#        elif i == (len(y_list)-1):#here range of random y is [1, y_list[i-1]]---
#            low_y = np.absolute(pos[:,1]) >  1
#            upp_y = np.absolute(pos[:,1]) <= y_list[i-1]
#            title = '%d<|y|<%d' %(1, y_list[i-1])
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
#        savehist(visib_diff, (-200, 200), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'Y_intensity-'+str(y_list[i]), w)
#    print('--------------------Scan along Y-------------------------')
#
#
##Range of absolute z------
#    print('\n------------------- Scan along Z------------------------')
#    print('Intensity and resolution evaluating...')           
#    #Shu: 630 will not work---
#    z_list = [90, 180, 270, 360, 450, 540, 630]
#    for i in range(len(z_list)):
#        w = True
#        if i == 0:#---
#            low_z = pos[:,2] >  0#the z position of the vertex---
#            upp_z = pos[:,2] <= z_list[i]
#            title = '0<|z|<%d' %(z_list[i])
#        elif i == (len(z_list)-1):#here range of random z is [1, z_list[i-1]]---
#            low_z = pos[:,2] >  0
#            upp_z = pos[:,2] <= z_list[i-1]
#            title = '0<|z|<%d' %(z_list[i-1])
#            w     = False
#        else:#here range of random z is [z_list[i-1], z_list[i]]---
#            low_z = pos[:,2] <= z_list[i]
#            upp_z = pos[:,2] >  z_list[i-1]
#            title = '%d<|z|<%d' %(z_list[i-1], z_list[i])
#            
#        image_diff = pre[cut & low_z & upp_z] - pdr[cut & low_z & upp_z]
#        image_true = pdr[cut & low_z & upp_z]
#        visib_diff = np.divide(image_diff, image_true)
#        
#        savehist(visib_diff, (-100, 100), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'Z_intensity-'+str(z_list[i]), w)
#    print('--------------------Scan along Z-------------------------')
#
#
#
#
#    print('\n------------------- Scan along Z------------------------')
#    print('Intensity and resolution evaluating...')           
#    #Shu: 630 will not work---
#    z_list = [-180, -90, 0, 90, 180, 270, 360, 450, 540, 630]
#    for i in range(len(z_list)):
#        w = True
#        if i == 0:#here range of random z is [-270, -180]---
#            low_z = pos[:,2] >  -270#the z position of the vertex---
#            upp_z = pos[:,2] <= z_list[i]
#            title = '-270<z<%d' %(z_list[i])
#        elif i == (len(z_list)-1):#here range of random z is [1, z_list[i-1]]---
#            low_z = pos[:,2] >  -270
#            upp_z = pos[:,2] <= z_list[i-1]
#            title = '-270<z<%d' %(z_list[i-1])
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
##        savehist(visib_diff, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'Z_intensity-'+str(z_list[i]), w)
#    print('--------------------Scan along Z-------------------------')

#---------------------------------------------------------------------





#----------Check range x[250, 300] and y[360, 420]-------------------
    pre = pre.sum(axis=1)#it means suming up all pds---
    pdr = pdr.sum(axis=1)    
    cut = (pre != 0) & (pdr != 0)

    #for range x[250, 300]------
    print('---------------X range: [250, 350]---------------')
    low_x = np.absolute(pos[:, 0]) > 0
    upp_x = np.absolute(pos[:, 0]) <= 350
    image_diff = pre[cut & low_x & upp_x] - pdr[cut & low_x & upp_x]
    image_true = pdr[cut & low_x & upp_x]
    visib_diff = np.divide(image_diff, image_true)
        
    print("Total points: ", len(visib_diff))

    title = '250<|x|<300'
    savehist(visib_diff, (-200, 200), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'Z_intensity-300.png', 'true')

    for num in range(0, len(visib_diff)):
#        if np.absolute(visib_diff[num] ) > 10:
        if np.absolute(pos[num,0])>250 and np.absolute(pos[num,0])<350 and np.absolute(visib_diff[num])>5:
            print('Value: ', format(visib_diff[num], '.2f'), "; Pos: (%d, %d, %d)"%(pos[num,0],pos[num,1],pos[num,2]))
            pos_X.append(pos[num, 0])
            pos_Y.append(pos[num, 1])
            pos_Z.append(pos[num, 2])
  
    #keep x, y, z in  txt files------
    with open('x_position.txt', 'w') as filehandle:
        for listitem in pos_X:
            filehandle.write('%d\n' % listitem)
    with open('y_position.txt', 'w') as filehandle:
        for listitem in pos_Y:
            filehandle.write('%d\n' % listitem)
    with open('z_position.txt', 'w') as filehandle:
        for listitem in pos_Z:
            filehandle.write('%d\n' % listitem)
    print("---------------------------------------------------")


    print('\n---------------Y range: [320, 420]---------------')
    low_y = np.absolute(pos[:,1]) > 0
    upp_y = np.absolute(pos[:,1]) <=  420
    image_diff = pre[cut & low_y & upp_y] - pdr[cut & low_y & upp_y]
    image_true = pdr[cut & low_y & upp_y]
    visib_diff = np.divide(image_diff, image_true)
        
    print("Total points: ", len(visib_diff))
    for num in range(0, len(visib_diff)):
#        if np.absolute(visib_diff[num] ) > 10:
        if np.absolute(pos[num,1])>320 and np.absolute(pos[num,1])<420 and np.absolute(visib_diff[num] ) > 5:
            print('Value: ', format(visib_diff[num], '.2f'), "; Pos: (%d, %d, %d)"%(pos[num,0],pos[num,1],pos[num,2]))
            pos_X2.append(pos[num, 0])
            pos_Y2.append(pos[num, 1])
            pos_Z2.append(pos[num, 2])
    #keep x, y, z in  txt files------
    with open('x2_position.txt', 'w') as filehandle:
        for listitem in pos_X2:
            filehandle.write('%d\n' % listitem)
    with open('y2_position.txt', 'w') as filehandle:
        for listitem in pos_Y2:
            filehandle.write('%d\n' % listitem)
    with open('z2_position.txt', 'w') as filehandle:
        for listitem in pos_Z2:
            filehandle.write('%d\n' % listitem)
    print("---------------------------------------------------")

#--------------------------------------------------------------------






#-----------------------------------------------------------------------
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

#--------------------------------------------------------------------------------




##To test single XArapuca-----------------------------------
#    cut_z = (pos[:,2] > -250) & (pos[:,2] < 550)
#    cut_y = (pos[:,1] > -400) & (pos[:,0] < 400)
#    coor_x = pos[:,0][cut_z & cut_y]
#    true_x = pdr[cut_z & cut_y]
#    emul_x = pre[cut_z & cut_y]
#   
#    num_x = len(coor_x)
#    true_x_s = np.zeros(shape=(num_x, 16))
#    emul_x_s = np.zeros(shape=(num_x, 16))
#
#    for index, op in zip(range(0, 16), range(0, 16)):
#        true_x_s[:,index] = true_x[:,op]
#        emul_x_s[:,index] = emul_x[:,op]
#
##===================================
##    for i in range(0, 16):
##        hist = np.divide((emul_z_s[:, i] - true_z_s[:, i]), true_z_s[:, i])
##        title = 'Response of XArapuca %d' %(i)
##        fileName = 'XA_%d' %(i)
##        savehist(hist, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, fileName, True)
##===================================
#
#    label = 0 #XA label---
#    hist = np.divide((emul_x_s[:, label] - true_x_s[:, label]), true_x_s[:, label])
#    title = 'Response of XArapuca %d' %(label)
#    fileName = 'XA_%d' %(label)
#    savehist(hist, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, fileName, True)
##---------------------------------------------------------








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

    if opt == 'SGD':
        optimizer = SGD(momentum=0.9)
    else:
        optimizer = Adam()
        
    model.compile(optimizer=optimizer, loss=vkld_loss, metrics=['mape', 'mae'])

