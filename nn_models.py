import numpy as np
import tensorflow as tf
import pickle
import pickle
import pandas as pd

from random import randint
import datetime
import colorlover as cl
from IPython.display import SVG
from IPython.display import HTML
from keras.utils.vis_utils import model_to_dot

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import norm
from keras import objectives

import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.merge import *
from keras.utils import to_categorical

from utils import *

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tqdm import TQDMNotebookCallback
from keras import regularizers
from keras import backend as K
from keras import metrics

def build_lstm_c_vae(original_dim,
                   timesteps,
                   batch_size = 20,
                   latent_dim = 2,
                   intermediate_dim = 32,
                   activ='relu',
                   optim=Adam(lr=0.0005),
                   epsilon_std=1.,
                   time_dim=0,
                   day_dim=0,
                   comp=True):
    

    input_dim = original_dim
    
    n_epoch = 100
    n_d = day_dim
    n_t = time_dim
    concat_day = not (n_d == 0)
    concat_time = not (n_t == 0)
    n_epoch = 100

    X = Input(shape=(timesteps, input_dim,), name='Input')
    day = Input(shape=(timesteps, day_dim,), name='Day')
    time = Input(shape=(timesteps, time_dim,), name='Time')

    if (concat_time):
        if (concat_day):
            inputs = Concatenate()([X, day, time])
        else:
            inputs = Concatenate()([X, time])
    else:
        if (concat_day):
            inputs = Concatenate()([X, day])
        else:
            inputs = X
    # LSTM encoding
    h = LSTM(intermediate_dim, activation=activ, name='LSTM_Encoder')(inputs)

    # VAE Z layer
    z_mean = Dense(latent_dim, activation='linear', name='Z_Mean')(h)
    z_log_sigma = Dense(latent_dim, activation='linear', name='Z_log_sigma')(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon_std

    z = Lambda(sampling, output_shape=(latent_dim,), name='Sampling')([z_mean, z_log_sigma])

    def get_last(x):
        return x[:, -1]

    if (concat_time):
        if (concat_day):
            zc = Concatenate()([z, Lambda(get_last)(day), Lambda(get_last)(time)])
        else:
            zc = Concatenate()([z, Lambda(get_last)(time)])
    else:
        if (concat_day):
            zc = Concatenate()([z, Lambda(get_last)(day)])
        else:
            zc = z
            
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, activation=activ, return_sequences=True)
    decoder_mean = LSTM(input_dim, activation='sigmoid',return_sequences=True)

    h_decoded = RepeatVector(timesteps)(zc)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim+time_dim+day_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    def recon_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        return xent_loss

    def KL_loss(x, x_decoded_mean):
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        return kl_loss
    
    generator = Model(decoder_input, _x_decoded_mean)
    
    if (concat_time):
        if (concat_day):
            vae = Model([X, day, time], x_decoded_mean)
            encoder = Model([X, day, time], z_mean)
        else:
            vae = Model([X, time], x_decoded_mean)
            encoder = Model([X, time], z_mean)
    else:
        if (concat_day):
            vae = Model([X, day], x_decoded_mean)
            encoder = Model([X, day], z_mean)
        else:
            vae = Model(X, x_decoded_mean)
            encoder = Model(X, z_mean)

    if (comp):
        vae.compile(optimizer=optim,
                    loss=vae_loss,
                    metrics = [KL_loss, recon_loss])
    
    return vae, encoder, generator


def buid_cvae(original_dim,
              batch_size = 20,
              latent_dim = 2,
              intermediate_dim = 10,
              intermediate_dim2 = 0,
              activ='relu',
              optim=Adam(lr=0.0005),
              epsilon_std=1.,
              time_dim=0,
              day_dim=0,
             comp=True):
    
    def sample_z(args):
        mu, l_sigma = args
        eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
        return mu + K.exp(l_sigma / 2) * eps
    
    m = batch_size # batch size
    n_z = latent_dim # latent space size
    encoder_dim1 = intermediate_dim # dim of encoder hidden layer
    encoder_dim2 = intermediate_dim2 # dim of encoder hidden layer
    decoder_dim = intermediate_dim # dim of decoder hidden layer
    decoder_dim2 = intermediate_dim2 # dim of decoder hidden layer
    decoder_out_dim = original_dim # dim of decoder output layer

    n_x = original_dim
    # n_y = y_train.shape[1]


    n_epoch = 100
    n_d = day_dim
    n_t = time_dim
    concat_day = not (n_d == 0)
    concat_time = not (n_t == 0)
    n_epoch = 100
    
    X = Input(shape=(n_x,), name='Input')
    day = Input(shape=(n_d,), name='Day')
    time = Input(shape=(n_t,), name='Time')
    
    if (concat_time):
        if (concat_day):
            inputs = Concatenate()([X, day, time])
        else:
            inputs = Concatenate()([X, time])
    else:
        if (concat_day):
            inputs = Concatenate()([X, day])
        else:
            inputs = X

    encoder_h = Dense(encoder_dim1, activation=activ, name='Encoder_H')(inputs)
    if encoder_dim2 > 0:
        encoder_h = Dense(encoder_dim2, activation=activ, name='Encoder_H2')(encoder_h)
    mu = Dense(n_z, activation='linear', name='mu')(encoder_h)
    l_sigma = Dense(n_z, activation='linear', name='l_sigma')(encoder_h)


    # Sampling latent space
    z = Lambda(sample_z, output_shape = (n_z, ), name='Sampling')([mu, l_sigma])
    
    if (concat_time):
        if (concat_day):
            zc = Concatenate()([z, day, time])
        else:
            zc = Concatenate()([z, time])
    else:
        if (concat_day):
            zc = Concatenate()([z, day])
        else:
            zc = z

    decoder_hidden = Dense(decoder_dim, activation=activ, name='Decoder_H')
    decoder_hidden2 = Dense(decoder_dim2, activation=activ, name='Decoder_H2')
        
    decoder_out = Dense(decoder_out_dim, activation='sigmoid', name='Output')
    if encoder_dim2 > 0:
        zc = decoder_hidden2(zc)
    h_p = decoder_hidden(zc)
    outputs = decoder_out(h_p)

    def vae_loss(y_true, y_pred):
        recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
        kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1)
        return recon + kl

    def KL_loss(y_true, y_pred):
        return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

    def recon_loss(y_true, y_pred):
        return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))

    # build a model to project inputs on the latent space
    

    d_in = Input(shape=(n_z+n_t+n_d,))
    if encoder_dim2 > 0:
        d_h = decoder_hidden2(d_in)
        d_h = decoder_hidden(d_h)
    else:
        d_h = decoder_hidden(d_in)
    d_out = decoder_out(d_h)
    
    if (concat_time):
        if (concat_day):
            vae = Model([X, day, time], outputs)
            encoder = Model([X, day, time], mu)
            generator = Model(d_in, d_out)
        else:
            vae = Model([X, time], outputs)
            encoder = Model([X, time], mu)
            generator = Model(d_in, d_out)
    else:
        if (concat_day):
            vae = Model([X, day], outputs)
            encoder = Model([X, day], mu)
            generator = Model(d_in, d_out)
        else:
            vae = Model(X, outputs)
            encoder = Model(X, mu)
            generator = Model(d_in, d_out)

    if (comp):
        vae.compile(optimizer=optim,
                loss=vae_loss,
                metrics = [KL_loss, recon_loss])
    
    return vae, encoder, generator

def build_vae(original_dim,
              batch_size = 20,
              latent_dim = 2,
              intermediate_dim = 10,
              activ='relu',
              optim=Adam(lr=0.0005),
              epsilon_std=1.):
    
    def sample_z(args):
        mu, l_sigma = args
        eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
        return mu + K.exp(l_sigma / 2) * eps

    m = batch_size # batch size
    n_z = latent_dim # latent space size
    encoder_dim1 = intermediate_dim # dim of encoder hidden layer
    decoder_dim = intermediate_dim # dim of decoder hidden layer
    decoder_out_dim = original_dim # dim of decoder output layer

    n_x = original_dim
    # n_y = y_train.shape[1]


    n_epoch = 100

    X = Input(shape=(n_x,), name='Input')
    # activity_regularizer = 'l2',
    encoder_h = Dense(encoder_dim1, activation=activ,  name='Encoder_H')(X)
    mu = Dense(n_z, activation='linear', name='mu')(encoder_h)
    l_sigma = Dense(n_z, activation='linear', name='l_sigma')(encoder_h)


    # Sampling latent space
    z = Lambda(sample_z, output_shape = (n_z, ), name='Sampling')([mu, l_sigma])

    decoder_hidden = Dense(decoder_dim, activation=activ, name='Decoder_H')
    decoder_out = Dense(decoder_out_dim, activation='sigmoid', name='Output')
    h_p = decoder_hidden(z)
    outputs = decoder_out(h_p)

    def vae_loss(y_true, y_pred):
        recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
        kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1)
        return recon + kl

    def KL_loss(y_true, y_pred):
        return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

    def recon_loss(y_true, y_pred):
        return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))

    # build a model to project inputs on the latent space
    vae = Model(X, outputs)

    # Build Encoder and Generator models

    # Generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(n_z,))
    _h_decoded = decoder_hidden(decoder_input)
    _x_decoded_mean = decoder_out(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    encoder = Model(X, mu)

    vae.compile(optimizer=optim,
                loss=vae_loss,
                metrics = [KL_loss, recon_loss])

    return vae, encoder, generator

    