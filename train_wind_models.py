#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train_winds.py
@Time    :   2023/11/01 14:12:43
@Author  :   Matija
'''
import fl_utils as fl_utils 
import numpy as np 
import torch 
import os 
import time 
import configs 
import pickle 

# MATIJA
import get_data as get_data
import shutil
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')



# device=torch.device("cuda")
device=torch.device("cpu")

# MATIJA
def aggregate_the_model(conf, model_group):
    num_params = len(model_group)

    # Create a new model instance
    averaged_model = fl_utils.initial_model_tensorflow()

    if not averaged_model.built:
        input_shape = (2,)
        if input_shape is None:
            raise ValueError("Input shape must be specified in 'conf' to build the model.")
        averaged_model.build((None,) + input_shape)

    # Set the averaged weights
    averaged_model_weights = [model_group[f'param_{i}'] for i in range(num_params)]
    averaged_model.set_weights(averaged_model_weights)

    return averaged_model
          
def federated_averaging(conf, client_models):
    n_clients = conf.n_clients

    num_params = len(client_models[0].values())
    if not all(len(model.values()) == num_params for model in client_models):
        raise ValueError("All models must have the same number of parameters")

    # List to hold the summed parameters
    summed_params = [0] * num_params

    # Iterate over each client's model parameters and sum them
    for model in client_models:
        model_params_list = list(model.values())
        for i in range(num_params):
            summed_params[i] += tf.convert_to_tensor(model_params_list[i])

    # Initialize model_group as an empty dictionary
    model_group = {}

    # Compute the average for each parameter
    for i in range(num_params):
        averaged_param = summed_params[i] / n_clients
        model_group[f'param_{i}'] = averaged_param.numpy()

    model = aggregate_the_model(conf, model_group)

    return model

def run_train_tensorflow(conf, train_data, test_data, exist_model):
    exist_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(5e-4), loss=fl_utils.EvidentialRegressionLoss)
    # TRAIN data
    client_train_data_x = train_data[['Wind Speed (m/s)', 'Blade Length (m)']]
    client_train_data_y = train_data[['Power Output (MW)']]
    # TEST data
    test_data_x = test_data[['Wind Speed (m/s)', 'Blade Length (m)']]
    test_data_y = test_data[['Power Output (MW)']]
    # FIT THE MODEL
    exist_model.fit(client_train_data_x.values, client_train_data_y.values,  batch_size=conf.batch_size, epochs=conf.num_local_epochs, verbose=0)
    print("Done Local ID %02d" % conf.use_local_id )
    # SAVE THE MODEL
    exist_model.save(conf.model_dir + "/client_id_%02d.pt" % conf.use_local_id)
    # EVALUATE THE MODEL
    client_uncertainty = model_evaluation_tensorflow(test_data_x, exist_model)

    return exist_model

def model_evaluation_tensorflow(test_data_x, exist_model):
    prediction = exist_model.predict(test_data_x.values)
   # x_test = test_data_x.values[:, 0]
    mu, v, alpha, beta = tf.split(prediction, 4, axis = 1) # Hyperparameters of evidential distributions

    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))#
    var = np.minimum(var, 1e3)[:, 0]

    df_uncertainty = fl_utils.get_uncertainty_estimation(prediction, test_data_x.values)
   # print(df_uncertainty)
    return df_uncertainty

def average_weights(model_weights):
    """ Averages weights from provided list of model weight sets. """
    average_weights = list()

    # Number of layers
    num_layers = len(model_weights[0])
    
    for layer in range(num_layers):
        # Collect this layer's weights from each model
        layer_weights = np.array([model[layer] for model in model_weights])
        
        # Average layer weights across all models
        layer_average = np.mean(layer_weights, axis=0)
        
        average_weights.append(layer_average)

    return average_weights

def run_server(conf):
    time_init = time.time()
    print("starting to calculate the aggregated model")
    
    files = os.listdir(conf.model_dir)

    models_array = []
    for file in files:
        load_model = fl_utils.initial_model_tensorflow()
        
        if not load_model.built:
            input_shape = (2,)
            if input_shape is None:
                raise ValueError("Input shape must be specified in 'conf' to build the model.")
            load_model.build((None,) + input_shape)
        
        load_model = tf.keras.models.load_model(conf.model_dir+file, custom_objects={'EvidentialRegressionLoss': fl_utils.EvidentialRegressionLoss})
        models_array.append(load_model)
  #  print("READ IN THE MODELS")
        
    model_weights = [model.get_weights() for model in models_array]

  #  print("GOT THE PARAMETERS")

    #model_fed_avg = federated_averaging(conf, model_parameters)
    average_weights1 = average_weights(model_weights)

    model_fed_avg = fl_utils.initial_model_tensorflow()
    if not model_fed_avg.built:
        input_shape = (2,)
        if input_shape is None:
            raise ValueError("Input shape must be specified in 'conf' to build the model.")
        model_fed_avg.build((None,) + input_shape) 

    model_fed_avg.set_weights(average_weights1)

    # print(model_fed_avg)
    # print(model_fed_avg.summary())
   # print("FINAL MODEL")
    
    model_fed_avg.save(conf.model_dir + "/aggregated_model.pt")
    mse, uncertainty, edl_loss = test_aggregated_model(model_fed_avg)

    print("time on the server", time.time() - time_init)
    
    return mse, uncertainty, edl_loss

def test_aggregated_model(aggregated_model):
    # data
    wind_test_dataset = get_data.get_wind_test_dataset()
    test_data_x = wind_test_dataset[['Wind Speed (m/s)', 'Blade Length (m)']]
    test_data_y = wind_test_dataset[['Power Output (MW)']].values

    # model
    final_prediction = aggregated_model.predict(test_data_x.values)

    #x_test = test_data_x.values[:, 0]
    mu, v, alpha, beta = tf.split(final_prediction, 4, axis = 1) # Hyperparameters of evidential distribution
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))#
    var = np.minimum(var, 1e3)[:, 0]

    df_uncertainty = fl_utils.get_uncertainty_estimation(final_prediction, test_data_x.values)
    edl_loss = fl_utils.EvidentialRegressionLoss(test_data_y, final_prediction)

    # Calculate MSE
    mse = fl_utils.mean_squared_error(test_data_y, mu)
    return mse, df_uncertainty, edl_loss

   
def train_with_conf(conf):    
    model_mom = "experiments_wind/"

    conf.folder_name = "experiment_1" 
    conf.dir_name = "2023_11_27"
    
    model_dir = model_mom + "%s/%s/" % (conf.folder_name, conf.dir_name)     
    
    stat_use = model_dir + "/stat.obj" 
    if os.path.exists(stat_use): # Check if the statistics folder exists 
        if conf.use_local_id == 0:
            content = pickle.load(open(stat_use, "rb")) 
    else:
        content = {}
        content["server_mse"] = []
        content["server_uncertainty"] = []
        content["server_edl_loss"] = []


    wind_train_subset = get_data.get_wind_train_dataset(conf) 
    wind_test_dataset = get_data.get_wind_test_dataset()

    # print("GPU availability", torch.cuda.is_available())
    # print("The used learning rate", conf.lr)


    model_path = model_dir + "/communication_round_%03d/" % conf.round 
    if conf.use_local_id == 0: # 0 -> server in federated learning setup
        
        # If model_path is a directory, delete it
        if os.path.isdir(model_path): # 
            shutil.rmtree(model_path)

        # If the directory for model_path doesn't exist, create it
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

    conf.model_dir = model_path
    
    if conf.round == 0: # if first round, initialize model
        exist_model = fl_utils.initial_model_tensorflow() 
    else:
        try:
            model_path = model_dir + "communication_round_%03d/" % (conf.round-1) + "/aggregated_model.pt"
            exist_model = tf.keras.models.load_model(model_path, custom_objects={'EvidentialRegressionLoss': fl_utils.EvidentialRegressionLoss})
            print(exist_model.get_weights())
        except FileNotFoundError:
            print("Failed at round ", conf.round - 1)
            return []    
    time_init = time.time()
    _model = run_train_tensorflow(conf, wind_train_subset, wind_test_dataset, exist_model)
    
    
    print("finish training model time", time.time() - time_init)
    
    while True:
        # Check if the model files for all clients are present in the directory. 
        # create a list of booleans for each client whether the model file is present or not
        if np.sum([os.path.isdir(conf.model_dir + "client_id_%02d.pt" % j) for j in range(conf.n_clients)]) == conf.n_clients:
            
           if conf.use_local_id == 0: # if server
                time.sleep(10)
                mse, uncertainty, edl_loss = run_server(conf) # aggregated model training
                content['server_mse'].append(mse)
                content['server_edl_loss'].append(edl_loss)
                content['server_uncertainty'].append(uncertainty)
                with open(stat_use, "wb") as f:
                    pickle.dump(content, f)
                print("Finish getting the server model at round", conf.round)
                break  
           else:
               break 
    del exist_model
    del _model 
    if conf.round >= 4 and conf.use_local_id == 0:
        path3remove(model_dir + "communication_round_%03d/" % (conf.round-4))
          
def path3remove(model_dir):
    # if "communication_round_" in model_dir and os.path.exists(model_dir):
    #     # Remove all contents within the directory
    #     shutil.rmtree(model_dir)
    #     print(f"Removed directory: {model_dir}")
    # else:
    #     print(f"Directory not found or does not match criteria: {model_dir}")    
    pass

if __name__ == "__main__":
    conf = configs.give_args()
    conf.lr = float(conf.lr) # learning rate
    if conf.round == 0:
        for arg in vars(conf):
            print(arg, getattr(conf, arg)) # print all the configurations
    train_with_conf(conf)