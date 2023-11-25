#1D Generative Neural Network for Photon Detection Fast Simulation (network for multi-input single-output regression)

#This file was created by Muve (Fermilab), most of the codes were written by him. Great work and pioneer---
#I (Shuaixiang (Shu) will make some modifications and comments, to satisfy my interests---
#Nov 22, 2023---


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Dense, concatenate, Multiply, Lambda, Flatten, BatchNormalization, PReLU, ReLU
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np

def outer_product(inputs):
    x, y = inputs
    batchSize = K.shape(x)[0]    
    outerProduct = x[:,:, np.newaxis] * y[:, np.newaxis,:]
    #outerProduct = K.reshape(outerProduct, (batchSize, -1))    
    #outerProduct = tf.tensordot(x, y, axes=0)
    #outerProduct = tf.reshape(outerProduct, [batchSize, dimX*dimY])
    return outerProduct
    
def vkld_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    diff   = (y_true-y_pred)
    loss   = K.abs(K.sum(diff*K.log(y_pred/y_true), axis=-1))
    return loss

#Network Architecture
#Input: three scalars (Position of scintillation)
#Output: one vecotr (photon detector response)
def model_protodunev7_t0(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
        
    feat_npl = Dense(1)(pos_x)
    feat_npl = BatchNormalization(momentum=0.9)(feat_npl)
    feat_npl = ReLU()(feat_npl)
    
    feat_ppl = Dense(1)(pos_x)
    feat_ppl = BatchNormalization(momentum=0.9)(feat_ppl)
    feat_ppl = ReLU()(feat_ppl)
    
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(3)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_bar = Lambda(outer_product)([feat_row, feat_col])
    feat_bar = Flatten()(feat_bar)
    
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(29)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    
    feat_plt = concatenate([pos_y, pos_z])
    
    feat_npa = Dense(4)(feat_plt)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    
    feat_ppa = Dense(4)(feat_plt)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    
    feat_hl1 = concatenate([feat_bar, feat_npa])
    feat_hl1 = Multiply()([feat_hl1, feat_npl])    
    
    feat_hl2 = concatenate([feat_ppa, feat_bar])
    feat_hl2 = Multiply()([feat_hl2, feat_ppl])
    
    feat_con = concatenate([feat_hl1, feat_hl2])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='protodunev7_model')  
    
    model.summary()
    #plot_model(model, to_file='./model_protodunev7_t0.png', show_shapes=True)
    return model    

def model_protodunev7_t1(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
        
    feat_npl = Dense(1)(pos_x)
    feat_npl = BatchNormalization(momentum=0.9)(feat_npl)
    feat_npl = ReLU()(feat_npl)
    
    feat_ppl = Dense(1)(pos_x)
    feat_ppl = BatchNormalization(momentum=0.9)(feat_ppl)
    feat_ppl = ReLU()(feat_ppl)
    
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(3)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_bar = Lambda(outer_product)([feat_row, feat_col])
    feat_bar = Flatten()(feat_bar)
    
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(29)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    
    feat_plt = concatenate([pos_y, pos_z])
    
    feat_npa = Dense(4)(feat_plt)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    
    feat_ppa = Dense(4)(feat_plt)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    
    feat_hl1 = concatenate([feat_bar, feat_npa])
    feat_hl1 = Multiply()([feat_hl1, feat_npl])    
    
    feat_hl2 = concatenate([feat_ppa, feat_bar])
    feat_hl2 = Multiply()([feat_hl2, feat_ppl])
    
    feat_con = concatenate([feat_hl1, feat_hl2])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='protodunev7_model')  
    
    model.summary()
    #plot_model(model, to_file='./model_protodunev7_t1.png', show_shapes=True)
    return model    
    
def model_protodunev7_t2(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
        
    feat_npl = Dense(1)(pos_x)
    feat_npl = BatchNormalization(momentum=0.9)(feat_npl)
    feat_npl = ReLU()(feat_npl)
    
    feat_ppl = Dense(1)(pos_x)
    feat_ppl = BatchNormalization(momentum=0.9)(feat_ppl)
    feat_ppl = ReLU()(feat_ppl)
    
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(3)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_bar = Lambda(outer_product)([feat_row, feat_col])
    feat_bar = Flatten()(feat_bar)

    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(29)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    
    feat_plt = concatenate([pos_y, pos_z])
    
    feat_npa = Dense(4)(feat_plt)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    
    feat_ppa = Dense(4)(feat_plt)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    
    feat_hl1 = concatenate([feat_bar, feat_npa])
    feat_hl1 = Multiply()([feat_hl1, feat_npl])    
    
    feat_hl2 = concatenate([feat_ppa, feat_bar])
    feat_hl2 = Multiply()([feat_hl2, feat_ppl])
    
    feat_con = concatenate([feat_hl1, feat_hl2])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='protodunev7_model')  
    
    model.summary()
    #plot_model(model, to_file='./model_protodunev7_t2.png', show_shapes=True)
    return model    

def model_protodunev7_t3(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
        
    feat_npl = Dense(1)(pos_x)
    feat_npl = BatchNormalization(momentum=0.9)(feat_npl)
    feat_npl = ReLU()(feat_npl)
    
    feat_ppl = Dense(1)(pos_x)
    feat_ppl = BatchNormalization(momentum=0.9)(feat_ppl)
    feat_ppl = ReLU()(feat_ppl)
    
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(3)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_bar = Lambda(outer_product)([feat_row, feat_col])
    feat_bar = Flatten()(feat_bar)
    
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    feat_bar = Dense(29)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    
    feat_plt = concatenate([pos_y, pos_z])
    
    feat_npa = Dense(4)(feat_plt)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    feat_npa = Dense(16)(feat_npa)
    feat_npa = BatchNormalization(momentum=0.9)(feat_npa)
    feat_npa = ReLU()(feat_npa)
    
    feat_ppa = Dense(4)(feat_plt)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    feat_ppa = Dense(16)(feat_ppa)
    feat_ppa = BatchNormalization(momentum=0.9)(feat_ppa)
    feat_ppa = ReLU()(feat_ppa)
    
    feat_hl1 = concatenate([feat_bar, feat_npa])
    feat_hl1 = Multiply()([feat_hl1, feat_npl])    
    
    feat_hl2 = concatenate([feat_ppa, feat_bar])
    feat_hl2 = Multiply()([feat_hl2, feat_ppl])
    
    feat_con = concatenate([feat_hl1, feat_hl2])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='protodunev7_model')  
    
    model.summary()
    #plot_model(model, to_file='./model_protodunev7_t3.png', show_shapes=True)
    return model    
    
def model_dune10kv4_t0(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
    
    feat_int = Dense(1)(pos_x)
    feat_int = BatchNormalization(momentum=0.9)(feat_int)
    feat_int = ReLU()(feat_int)
        
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(12)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_cov = Lambda(outer_product)([feat_row, feat_col])
    feat_cov = Flatten()(feat_cov)
    feat_cov = Multiply()([feat_cov, feat_int])
    
    feat_cov = Dense(240)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_cov)
    model    = Model(inputs=input_layer, outputs=pdr, name='dune10k_wide_model')
    
    model.summary()
    #plot_model(model, to_file='./model_dune10kv4_wide.png', show_shapes=True)
    return model
    
def model_dune10kv4_t1(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
    
    feat_int = Dense(1)(pos_x)
    feat_int = BatchNormalization(momentum=0.9)(feat_int)
    feat_int = ReLU()(feat_int)
        
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(12)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_cov = Lambda(outer_product)([feat_row, feat_col])
    feat_cov = Flatten()(feat_cov)
    feat_cov = Multiply()([feat_cov, feat_int])
    
    feat_cov = Dense(240)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    
    feat_cov = Dense(dim_pdr)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_cov)
    model    = Model(inputs=input_layer, outputs=pdr, name='dune10k_wide_model')
    
    model.summary()
    #plot_model(model, to_file='./model_dune10kv4_t1.png', show_shapes=True)
    return model

def model_dune10kv4_t2(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
    
    feat_int = Dense(1)(pos_x)
    feat_int = BatchNormalization(momentum=0.9)(feat_int)
    feat_int = ReLU()(feat_int)
        
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(12)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_cov = Lambda(outer_product)([feat_row, feat_col])
    feat_cov = Flatten()(feat_cov)
    feat_cov = Multiply()([feat_cov, feat_int])
    
    feat_cov = Dense(240)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)

    feat_cov = Dense(dim_pdr)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    feat_cov = Dense(dim_pdr)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_cov)
    model    = Model(inputs=input_layer, outputs=pdr, name='dune10k_wide_model')
    
    model.summary()
    #plot_model(model, to_file='./model_dune10kv4_t2.png', show_shapes=True)
    return model
    
def model_dune10kv4_t3(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
    
    feat_int = Dense(1)(pos_x)
    feat_int = BatchNormalization(momentum=0.9)(feat_int)
    feat_int = ReLU()(feat_int)
        
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(12)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_cov = Lambda(outer_product)([feat_row, feat_col])
    feat_cov = Flatten()(feat_cov)
    feat_cov = Multiply()([feat_cov, feat_int])
    
    feat_cov = Dense(240)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)

    feat_cov = Dense(dim_pdr)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    feat_cov = Dense(dim_pdr)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    feat_cov = Dense(dim_pdr)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_cov)
    model    = Model(inputs=input_layer, outputs=pdr, name='dune10k_wide_model')
    
    model.summary()
    #plot_model(model, to_file='./model_dune10kv4_t3.png', show_shapes=True)
    return model

def model_dunevd_t0(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]

    # photon detection array on cathode
    feat_int_cathode = Dense(1)(pos_x)
    feat_int_cathode = BatchNormalization(momentum=0.9)(feat_int_cathode)
    feat_int_cathode = ReLU()(feat_int_cathode)
        
    feat_row_cathode = Dense(8)(pos_y)
    feat_row_cathode = BatchNormalization(momentum=0.9)(feat_row_cathode)
    feat_row_cathode = ReLU()(feat_row_cathode)
    
    feat_col_cathode = Dense(7)(pos_z)
    feat_col_cathode = BatchNormalization(momentum=0.9)(feat_col_cathode)
    feat_col_cathode = ReLU()(feat_col_cathode)
    
    feat_cov_cathode = Lambda(outer_product)([feat_row_cathode, feat_col_cathode])
    feat_cov_cathode = Flatten()(feat_cov_cathode)
    feat_cov_cathode = Multiply()([feat_cov_cathode, feat_int_cathode])
    
    feat_cov_cathode = Dense(112)(feat_cov_cathode)
    feat_cov_cathode = BatchNormalization(momentum=0.9)(feat_cov_cathode)
    feat_cov_cathode = ReLU()(feat_cov_cathode)

    # photon detection array on side
    feat_int_side = Dense(1)(pos_y)
    feat_int_side = BatchNormalization(momentum=0.9)(feat_int_side)
    feat_int_side = ReLU()(feat_int_side)
        
    feat_row_side = Dense(4)(pos_x)
    feat_row_side = BatchNormalization(momentum=0.9)(feat_row_side)
    feat_row_side = ReLU()(feat_row_side)
    
    feat_col_side = Dense(7)(pos_z)
    feat_col_side = BatchNormalization(momentum=0.9)(feat_col_side)
    feat_col_side = ReLU()(feat_col_side)
    
    feat_cov_side = Lambda(outer_product)([feat_row_side, feat_col_side])
    feat_cov_side = Flatten()(feat_cov_side)
    feat_cov_side = Multiply()([feat_cov_side, feat_int_side])
    
    feat_cov_side = Dense(56)(feat_cov_side)        
    feat_cov_side = BatchNormalization(momentum=0.9)(feat_cov_side)
    feat_cov_side = ReLU()(feat_cov_side)

    feat_con = concatenate([feat_cov_side, feat_cov_cathode])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='dunevd_model')
    
    model.summary()
    #plot_model(model, to_file='./dunevd_t0.png', show_shapes=True)
    return model

def model_dunevd_t1(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]

    # photon detection array on cathode
    feat_int_cathode = Dense(1)(pos_x)
    feat_int_cathode = BatchNormalization(momentum=0.9)(feat_int_cathode)
    feat_int_cathode = ReLU()(feat_int_cathode)
        
    feat_row_cathode = Dense(8)(pos_y)
    feat_row_cathode = BatchNormalization(momentum=0.9)(feat_row_cathode)
    feat_row_cathode = ReLU()(feat_row_cathode)
    
    feat_col_cathode = Dense(7)(pos_z)
    feat_col_cathode = BatchNormalization(momentum=0.9)(feat_col_cathode)
    feat_col_cathode = ReLU()(feat_col_cathode)
    
    feat_cov_cathode = Lambda(outer_product)([feat_row_cathode, feat_col_cathode])
    feat_cov_cathode = Flatten()(feat_cov_cathode)
    feat_cov_cathode = Multiply()([feat_cov_cathode, feat_int_cathode])

    feat_cov_cathode = Dense(56)(feat_cov_cathode)
    feat_cov_cathode = BatchNormalization(momentum=0.9)(feat_cov_cathode)
    feat_cov_cathode = ReLU()(feat_cov_cathode)
    
    feat_cov_cathode = Dense(112)(feat_cov_cathode)
    feat_cov_cathode = BatchNormalization(momentum=0.9)(feat_cov_cathode)
    feat_cov_cathode = ReLU()(feat_cov_cathode)

    # photon detection array on side
    feat_int_side = Dense(1)(pos_y)
    feat_int_side = BatchNormalization(momentum=0.9)(feat_int_side)
    feat_int_side = ReLU()(feat_int_side)
        
    feat_row_side = Dense(4)(pos_x)
    feat_row_side = BatchNormalization(momentum=0.9)(feat_row_side)
    feat_row_side = ReLU()(feat_row_side)
    
    feat_col_side = Dense(7)(pos_z)
    feat_col_side = BatchNormalization(momentum=0.9)(feat_col_side)
    feat_col_side = ReLU()(feat_col_side)
    
    feat_cov_side = Lambda(outer_product)([feat_row_side, feat_col_side])
    feat_cov_side = Flatten()(feat_cov_side)
    feat_cov_side = Multiply()([feat_cov_side, feat_int_side])
    
    feat_cov_side = Dense(56)(feat_cov_side)        
    feat_cov_side = BatchNormalization(momentum=0.9)(feat_cov_side)
    feat_cov_side = ReLU()(feat_cov_side)

    feat_cov_side = Dense(56)(feat_cov_side)        
    feat_cov_side = BatchNormalization(momentum=0.9)(feat_cov_side)
    feat_cov_side = ReLU()(feat_cov_side)
    
    feat_con = concatenate([feat_cov_side, feat_cov_cathode])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='dunevd_model')
    
    model.summary()
    #plot_model(model, to_file='./dunevd_t1.png', show_shapes=True)
    return model
    
def model_protodunehd_t0(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
    
    feat_npl = Dense(1)(pos_x)
    feat_npl = BatchNormalization(momentum=0.9)(feat_npl)
    feat_npl = ReLU()(feat_npl)
    
    feat_ppl = Dense(1)(pos_x)
    feat_ppl = BatchNormalization(momentum=0.9)(feat_ppl)
    feat_ppl = ReLU()(feat_ppl)
        
    feat_row = Dense(8)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(10)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_cov = Lambda(outer_product)([feat_row, feat_col])
    feat_cov = Flatten()(feat_cov)
    
    feat_hl1 = Multiply()([feat_cov, feat_npl])
    feat_hl1 = Dense(80)(feat_hl1)
    feat_hl1 = BatchNormalization(momentum=0.9)(feat_hl1)
    feat_hl1 = ReLU()(feat_hl1)
    
    feat_hl2 = Multiply()([feat_cov, feat_ppl])
    feat_hl2 = Dense(80)(feat_hl2)
    feat_hl2 = BatchNormalization(momentum=0.9)(feat_hl2)
    feat_hl2 = ReLU()(feat_hl2)
    
    feat_con = concatenate([feat_hl1, feat_hl2])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='protodunehd_model')
    
    model.summary()
    #plot_model(model, to_file='./protodunehd_t0.png', show_shapes=True)
    return model



#For module 0, suggested by Mu, 20230125---
def model_dunevd_16op(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]

    # for channel 0 - 3
    feat_int_1 = Dense(1)(pos_z)#intensity layer---
    feat_int_1 = BatchNormalization(momentum=0.9)(feat_int_1)
    feat_int_1 = ReLU()(feat_int_1)
        
    feat_row_1 = Dense(2)(pos_y)
    feat_row_1 = BatchNormalization(momentum=0.9)(feat_row_1)
    feat_row_1 = ReLU()(feat_row_1)
    
    feat_col_1 = Dense(2)(pos_x)#intensity layer---
    feat_col_1 = BatchNormalization(momentum=0.9)(feat_col_1)
    feat_col_1 = ReLU()(feat_col_1)
    
    feat_cov_1 = Lambda(outer_product)([feat_row_1, feat_col_1])
    feat_cov_1 = Flatten()(feat_cov_1)
    feat_cov_1 = Multiply()([feat_cov_1, feat_int_1])

    feat_cov_1 = Dense(750)(feat_cov_1)        
    feat_cov_1 = BatchNormalization(momentum=0.9)(feat_cov_1)
    feat_cov_1 = ReLU()(feat_cov_1)

    # for channel 4 - 11
    feat_int_2 = Dense(1)(pos_x)
    feat_int_2 = BatchNormalization(momentum=0.9)(feat_int_2)
    feat_int_2 = ReLU()(feat_int_2)
        
    feat_row_2 = Dense(2)(pos_y)
    feat_row_2 = BatchNormalization(momentum=0.9)(feat_row_2)
    feat_row_2 = ReLU()(feat_row_2)
    
    feat_col_2 = Dense(4)(pos_z)
    feat_col_2 = BatchNormalization(momentum=0.9)(feat_col_2)
    feat_col_2 = ReLU()(feat_col_2)
    
    feat_cov_2 = Lambda(outer_product)([feat_row_2, feat_col_2])
    feat_cov_2 = Flatten()(feat_cov_2)
    feat_cov_2 = Multiply()([feat_cov_2, feat_int_2])

    feat_cov_2 = Dense(1500)(feat_cov_2)        
    feat_cov_2 = BatchNormalization(momentum=0.9)(feat_cov_2)
    feat_cov_2 = ReLU()(feat_cov_2)

    # for channel 12 - 15
    feat_int_3 = Dense(1)(pos_z)
    feat_int_3 = BatchNormalization(momentum=0.9)(feat_int_3)
    feat_int_3 = ReLU()(feat_int_3)
        
    feat_row_3 = Dense(2)(pos_y)
    feat_row_3 = BatchNormalization(momentum=0.9)(feat_row_3)
    feat_row_3 = ReLU()(feat_row_3)
    
    feat_col_3 = Dense(2)(pos_x)
    feat_col_3 = BatchNormalization(momentum=0.9)(feat_col_3)
    feat_col_3 = ReLU()(feat_col_3)
    
    feat_cov_3 = Lambda(outer_product)([feat_row_3, feat_col_3])
    feat_cov_3 = Flatten()(feat_cov_3)
    feat_cov_3 = Multiply()([feat_cov_3, feat_int_3])

    feat_cov_3 = Dense(750)(feat_cov_3)
    feat_cov_3 = BatchNormalization(momentum=0.9)(feat_cov_3)
    feat_cov_3 = ReLU()(feat_cov_3)

    # combine the three---
    feat_con = concatenate([feat_cov_1, feat_cov_2, feat_cov_3])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)

    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='dunevd_16op_model')
    
    model.summary()
    #plot_model(model, to_file='./dunevd_16op.png', show_shapes=True)
    return model



#===============================================================
#For protoDUNE-VD v4 geometry, containing 40 optical channels---
#Written by Shu, 20231122---
def model_protodunevd_v4(dim_pdr):#dim_pdr: num of opchannels---
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]


    #Part 1: channel 0 - 3
    feat_int_1 = Dense(1)(pos_z)#intensity layer---
    feat_int_1 = BatchNormalization(momentum=0.9)(feat_int_1)
    feat_int_1 = ReLU()(feat_int_1)
        
    feat_row_1 = Dense(2)(pos_y)
    feat_row_1 = BatchNormalization(momentum=0.9)(feat_row_1)
    feat_row_1 = ReLU()(feat_row_1)
    
    feat_col_1 = Dense(2)(pos_x)#intensity layer---
    feat_col_1 = BatchNormalization(momentum=0.9)(feat_col_1)
    feat_col_1 = ReLU()(feat_col_1)
    
    feat_cov_1 = Lambda(outer_product)([feat_row_1, feat_col_1])
    feat_cov_1 = Flatten()(feat_cov_1)
    feat_cov_1 = Multiply()([feat_cov_1, feat_int_1])

    feat_cov_1 = Dense(400)(feat_cov_1)        
    feat_cov_1 = BatchNormalization(momentum=0.9)(feat_cov_1)
    feat_cov_1 = ReLU()(feat_cov_1)


    #Part 2: channel 4 - 11
    feat_int_2 = Dense(1)(pos_x)
    feat_int_2 = BatchNormalization(momentum=0.9)(feat_int_2)
    feat_int_2 = ReLU()(feat_int_2)
        
    feat_row_2 = Dense(2)(pos_y)
    feat_row_2 = BatchNormalization(momentum=0.9)(feat_row_2)
    feat_row_2 = ReLU()(feat_row_2)
    
    feat_col_2 = Dense(4)(pos_z)
    feat_col_2 = BatchNormalization(momentum=0.9)(feat_col_2)
    feat_col_2 = ReLU()(feat_col_2)
    
    feat_cov_2 = Lambda(outer_product)([feat_row_2, feat_col_2])
    feat_cov_2 = Flatten()(feat_cov_2)
    feat_cov_2 = Multiply()([feat_cov_2, feat_int_2])

    feat_cov_2 = Dense(1000)(feat_cov_2)        
    feat_cov_2 = BatchNormalization(momentum=0.9)(feat_cov_2)
    feat_cov_2 = ReLU()(feat_cov_2)


    #Part 3: channel 12 - 15
    feat_int_3 = Dense(1)(pos_z)
    feat_int_3 = BatchNormalization(momentum=0.9)(feat_int_3)
    feat_int_3 = ReLU()(feat_int_3)
        
    feat_row_3 = Dense(2)(pos_y)
    feat_row_3 = BatchNormalization(momentum=0.9)(feat_row_3)
    feat_row_3 = ReLU()(feat_row_3)
    
    feat_col_3 = Dense(2)(pos_x)
    feat_col_3 = BatchNormalization(momentum=0.9)(feat_col_3)
    feat_col_3 = ReLU()(feat_col_3)
    
    feat_cov_3 = Lambda(outer_product)([feat_row_3, feat_col_3])
    feat_cov_3 = Flatten()(feat_cov_3)
    feat_cov_3 = Multiply()([feat_cov_3, feat_int_3])

    feat_cov_3 = Dense(200)(feat_cov_3)
    feat_cov_3 = BatchNormalization(momentum=0.9)(feat_cov_3)
    feat_cov_3 = ReLU()(feat_cov_3)


    #Part 4: channel 16 - 17
    feat_int_4 = Dense(1)(pos_z)
    feat_int_4 = BatchNormalization(momentum=0.9)(feat_int_4)
    feat_int_4 = ReLU()(feat_int_4)
        
    feat_row_4 = Dense(2)(pos_y)
    feat_row_4 = BatchNormalization(momentum=0.9)(feat_row_4)
    feat_row_4 = ReLU()(feat_row_4)
    
    feat_col_4 = Dense(2)(pos_x)
    feat_col_4 = BatchNormalization(momentum=0.9)(feat_col_4)
    feat_col_4 = ReLU()(feat_col_4)
    
    feat_cov_4 = Lambda(outer_product)([feat_row_4, feat_col_4])
    feat_cov_4 = Flatten()(feat_cov_4)
    feat_cov_4 = Multiply()([feat_cov_4, feat_int_4])

    feat_cov_4 = Dense(200)(feat_cov_4)
    feat_cov_4 = BatchNormalization(momentum=0.9)(feat_cov_4)
    feat_cov_4 = ReLU()(feat_cov_4)


    #Part 5: channel 18 - 21
    feat_int_5 = Dense(1)(pos_z)
    feat_int_5 = BatchNormalization(momentum=0.9)(feat_int_5)
    feat_int_5 = ReLU()(feat_int_5)
        
    feat_row_5 = Dense(2)(pos_y)
    feat_row_5 = BatchNormalization(momentum=0.9)(feat_row_5)
    feat_row_5 = ReLU()(feat_row_5)
    
    feat_col_5 = Dense(2)(pos_x)
    feat_col_5 = BatchNormalization(momentum=0.9)(feat_col_5)
    feat_col_5 = ReLU()(feat_col_5)
    
    feat_cov_5 = Lambda(outer_product)([feat_row_5, feat_col_5])
    feat_cov_5 = Flatten()(feat_cov_5)
    feat_cov_5 = Multiply()([feat_cov_5, feat_int_5])

    feat_cov_5 = Dense(200)(feat_cov_5)
    feat_cov_5 = BatchNormalization(momentum=0.9)(feat_cov_5)
    feat_cov_5 = ReLU()(feat_cov_5)


    #Part 6: channel 22 - 23
    feat_int_6 = Dense(1)(pos_z)
    feat_int_6 = BatchNormalization(momentum=0.9)(feat_int_6)
    feat_int_6 = ReLU()(feat_int_6)
        
    feat_row_6 = Dense(2)(pos_y)
    feat_row_6 = BatchNormalization(momentum=0.9)(feat_row_6)
    feat_row_6 = ReLU()(feat_row_6)
    
    feat_col_6 = Dense(2)(pos_x)
    feat_col_6 = BatchNormalization(momentum=0.9)(feat_col_6)
    feat_col_6 = ReLU()(feat_col_6)
    
    feat_cov_6 = Lambda(outer_product)([feat_row_6, feat_col_6])
    feat_cov_6 = Flatten()(feat_cov_6)
    feat_cov_6 = Multiply()([feat_cov_6, feat_int_6])

    feat_cov_6 = Dense(200)(feat_cov_6)
    feat_cov_6 = BatchNormalization(momentum=0.9)(feat_cov_6)
    feat_cov_6 = ReLU()(feat_cov_6)


    #Part 7: channel 24 - 29
    feat_int_7 = Dense(1)(pos_z)
    feat_int_7 = BatchNormalization(momentum=0.9)(feat_int_7)
    feat_int_7 = ReLU()(feat_int_7)
        
    feat_row_7 = Dense(2)(pos_y)
    feat_row_7 = BatchNormalization(momentum=0.9)(feat_row_7)
    feat_row_7 = ReLU()(feat_row_7)
    
    feat_col_7 = Dense(2)(pos_x)
    feat_col_7 = BatchNormalization(momentum=0.9)(feat_col_7)
    feat_col_7 = ReLU()(feat_col_7)
    
    feat_cov_7 = Lambda(outer_product)([feat_row_7, feat_col_7])
    feat_cov_7 = Flatten()(feat_cov_7)
    feat_cov_7 = Multiply()([feat_cov_7, feat_int_7])

    feat_cov_7 = Dense(400)(feat_cov_7)
    feat_cov_7 = BatchNormalization(momentum=0.9)(feat_cov_7)
    feat_cov_7 = ReLU()(feat_cov_7)


    #Part 8: channel 30 - 33
    feat_int_8 = Dense(1)(pos_z)
    feat_int_8 = BatchNormalization(momentum=0.9)(feat_int_8)
    feat_int_8 = ReLU()(feat_int_8)
        
    feat_row_8 = Dense(2)(pos_y)
    feat_row_8 = BatchNormalization(momentum=0.9)(feat_row_8)
    feat_row_8 = ReLU()(feat_row_8)
    
    feat_col_8 = Dense(2)(pos_x)
    feat_col_8 = BatchNormalization(momentum=0.9)(feat_col_8)
    feat_col_8 = ReLU()(feat_col_8)
    
    feat_cov_8 = Lambda(outer_product)([feat_row_8, feat_col_8])
    feat_cov_8 = Flatten()(feat_cov_8)
    feat_cov_8 = Multiply()([feat_cov_8, feat_int_8])

    feat_cov_8 = Dense(200)(feat_cov_8)
    feat_cov_8 = BatchNormalization(momentum=0.9)(feat_cov_8)
    feat_cov_8 = ReLU()(feat_cov_8)

    #Part 9: channel 34 - 39
    feat_int_9 = Dense(1)(pos_z)
    feat_int_9 = BatchNormalization(momentum=0.9)(feat_int_9)
    feat_int_9 = ReLU()(feat_int_9)
        
    feat_row_9 = Dense(2)(pos_y)
    feat_row_9 = BatchNormalization(momentum=0.9)(feat_row_9)
    feat_row_9 = ReLU()(feat_row_9)
    
    feat_col_9 = Dense(2)(pos_x)
    feat_col_9 = BatchNormalization(momentum=0.9)(feat_col_9)
    feat_col_9 = ReLU()(feat_col_9)
    
    feat_cov_9 = Lambda(outer_product)([feat_row_9, feat_col_9])
    feat_cov_9 = Flatten()(feat_cov_9)
    feat_cov_9 = Multiply()([feat_cov_9, feat_int_9])

    feat_cov_9 = Dense(400)(feat_cov_9)
    feat_cov_9 = BatchNormalization(momentum=0.9)(feat_cov_9)
    feat_cov_9 = ReLU()(feat_cov_9)


    #combine the nine blocks---
    feat_con = concatenate([feat_cov_1, feat_cov_2, feat_cov_3, feat_cov_4, feat_cov_5, feat_cov_6, feat_cov_7, feat_cov_8, feat_cov_9])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)

    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='protodunevd_v4_model')
    
    model.summary()
    #plot_model(model, to_file='./protodunevd_v4.png', show_shapes=True)
    return model





