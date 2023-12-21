#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/03/05 15:26:47
@Author  :   Bo 
'''
import numpy as np 
from scipy.special import softmax 
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
import random 
import os 
import configs 
import get_data as gsc 
import tensorflow as tf
import evidential_deep_learning as edl
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


def EvidentialRegressionLoss(true, pred):
    return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)
def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta = tf.split(y_pred, 4, axis = 1) # Hyperparameters of evidential distributions
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))#variance of the evidential distribution ; Students-t distribution
    var = np.minimum(var, 1e3)[:, 0] # clip variance for plotting
    # # Set uncertainty to 0 for x_test values above 13.5
    # mask = x_test > 13.5
    # var[mask] = 0
    
    plt.figure(figsize=(16, 6), dpi=100)
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
    plt.plot(x_test, mu, color='#007cab', zorder=3, label="Pred") # prediction line
    # Uncertainty epistemic   
    for k in np.linspace(0, n_stds+1, 4+1):
        plt.fill_between(
            x_test, (mu - k * var), (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
        
    plt.gca().set_ylim(-8, 22)
    plt.gca().set_xlim(-2, 50)
    plt.legend(loc="upper left")
    plt.show()
def get_uncertainty_estimation(predictions, x_test):
    x_test = x_test[:, 0]  
    # extract parameters from NIG (normal inverse-gamma)
    mu, v, alpha, beta = tf.split(predictions, 4, axis=1)
    mu = mu[:, 0]
    aleatoric_uncertainty = (beta / (alpha - 1))
    epistemic_uncertainty = (beta / (v * (alpha - 1)))
    total_evidence = 2*v + alpha

    # 
    df_uncertainty = pd.DataFrame({
        "beta": beta [:, 0], # scale parameters, influences the scale/spread
        "alpha": alpha [:, 0], # determines how the probability density of the IG distr. is shaped -> part of the prior distribution over variance of normal dist.
        "v": v[:, 0], # degrees of freedom -> amount of data or information that have informed this prediction
        "Aleatoric": aleatoric_uncertainty[:, 0],  # data uncertainty
        "Epistemic": epistemic_uncertainty[:, 0],  # model uncertainty due to lack of knowledge
        "total evidence" : total_evidence [:, 0], # overal strength / relability of prediction 
        "mu/prediction " : mu # mean of the normal distribution / PREDICTION
    }, index=x_test)

    df_uncertainty.index.name = 'x Test data point'

    return df_uncertainty


def create_model(conf):
    # Define your model architecture here
    # For example, a simple sequential model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(conf['units'], activation='relu'),
        tf.keras.layers.Dense(conf['units'], activation='relu'),
        edl.layers.DenseNormalGamma(1)
    ])
    return model
def initial_model_tensorflow():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        edl.layers.DenseNormalGamma(1)
    ])
    return model


def create_wind_dir(conf):
    model_mom = "../experiments_wind/"
    model_dir = model_mom + "experiment_1" 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    conf.folder_name = "experiment_1" 
    conf.dir_name = "2023"
    return conf
    
if __name__ == "__main__":
    conf = configs.give_args()
    if conf.dataset == "cifar10":
        pass
    elif conf.dataset == "wind":
        print("WIND")
        conf = create_wind_dir(conf)
        

    
        
        






        
    

