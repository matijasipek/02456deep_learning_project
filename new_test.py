import configs 
import get_data
import fl_utils
import tensorflow as tf
import numpy as np
import pandas as pd
import os

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

def run_train_tensorflow(conf, train_data, test_data, exist_model):

    model = exist_model 
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(5e-4), loss=fl_utils.EvidentialRegressionLoss)
    # TRAIN data
    client_train_data_x = train_data[['Wind Speed (m/s)', 'Blade Length (m)']]
    client_train_data_y = train_data[['Power Output (MW)']]
    # TEST data
    test_data_x = test_data[['Wind Speed (m/s)', 'Blade Length (m)']]
    test_data_y = test_data[['Power Output (MW)']]
    # FIT THE MODEL
    model.fit(client_train_data_x.values, client_train_data_y.values,  batch_size=conf.batch_size, epochs=conf.num_local_epochs, verbose=0)
    print("Done Local ID %02d" % conf.use_local_id )
    # SAVE THE MODEL
    model.save(conf.model_dir + "/client_id_%02d.pt" % conf.use_local_id)
    # EVALUATE THE MODEL
    client_uncertainty = model_evaluation_tensorflow(test_data_x, model)

    return model, client_uncertainty

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


if __name__ == "__main__":
    ### CONFIGURATION
    conf = configs.give_args()
    model_mom = "experiments_wind/"

    conf.folder_name = "experiment_1" 
    conf.dir_name = "new_test2"
    
    model_dir = model_mom + "%s/%s/" % (conf.folder_name, conf.dir_name) 
    model_path = model_dir + "communication_round_%03d/" % conf.round 
    conf.model_dir = model_path

    print(conf.model_dir)

    ### GLOBAL MODEL
    global_model = fl_utils.initial_model_tensorflow()
    if not global_model.built:
        input_shape = (2,)
        if input_shape is None:
            raise ValueError("Input shape must be specified in 'conf' to build the model.")
        global_model.build((None,) + input_shape) 

    ### TESTDATA
    wind_test_dataset = get_data.get_wind_test_dataset() # train
    
    ### statistics
    client_0 = {}
    client_1 = {}
    client_2 = {}

    client_0['model'] = []
    client_1['model'] = []
    client_2['model'] = []

    client_0['uncertainty'] = []
    client_1['uncertainty'] = []
    client_2['uncertainty'] = []

    # ### ROUND 1
    # client 1
    conf.use_local_id = 0
    wind_train_client_0 = get_data.get_wind_train_dataset(conf) 
    #print(wind_train_client_0)
    round_1_client_0, client_0_uncertainty = run_train_tensorflow(conf, wind_train_client_0, wind_test_dataset, global_model)
    print("trained model 0")
    client_0['model'].append(round_1_client_0)
    client_0['uncertainty'].append(client_0_uncertainty)
    # client 2
    conf.use_local_id = 1
    wind_train_client_1 = get_data.get_wind_train_dataset(conf) 
   # print(wind_train_client_1)
    round_1_client_1, client_1_uncertainty = run_train_tensorflow(conf, wind_train_client_1, wind_test_dataset, global_model)
    print("trained model 1")
    client_1['model'].append(round_1_client_1)
    client_1['uncertainty'].append(client_1_uncertainty)
    # client 3
    conf.use_local_id = 2
    wind_train_client_2 = get_data.get_wind_train_dataset(conf) 
   # print(wind_train_client_2)
    round_1_client_2, client_2_uncertainty = run_train_tensorflow(conf, wind_train_client_2, wind_test_dataset, global_model)
    print("trained model 1")
    client_2['model'].append(round_1_client_2)
    client_2['uncertainty'].append(client_2_uncertainty)

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
    print("Round 0: loaded in the files")
    #print(models_array)
    #print()
    # for model in models_array:
    #     print(model)
    #     print(model.get_weights()[0])  # Print the first layer weights of each model
    #     print()
    #     print()
    #     print()
    # Collect model weights from each model
    model_weights = [model.get_weights() for model in models_array]
    #print("Round 0: Collect model weights from each model")

    # Average model weights
    average_weights1 = average_weights(model_weights)
    global_model.set_weights(average_weights1)
    for model in models_array:
        print(model)
        print(model.get_weights()[0])  # Print the first layer weights of each model
        print()
        print()
        print()
    print("GLOBAL MODEL")
    print(global_model.get_weights()[0])

  #  print(global_model.get_weights()[0])  # Print the first layer weights of the global model
    #####################################################################################################################
    # #### ROUND 2
    print("ROUND 2")
    conf.round = 1
    model_path = model_dir + "communication_round_%03d/" % conf.round 
    conf.model_dir = model_path

    # client 1
    conf.use_local_id = 0
    wind_train_client_0 = get_data.get_wind_train_dataset(conf) 
    round_1_client_0, client_0_uncertainty = run_train_tensorflow(conf, wind_train_client_0, wind_test_dataset, global_model)
    print("trained model " + str(conf.use_local_id))
    client_0['model'].append(round_1_client_0)
    client_0['uncertainty'].append(client_0_uncertainty)
    # client 2
    conf.use_local_id = 1
    wind_train_client_1 = get_data.get_wind_train_dataset(conf) 
    round_1_client_1, client_1_uncertainty = run_train_tensorflow(conf, wind_train_client_1, wind_test_dataset, global_model)
    print("trained model " + str(conf.use_local_id))
    client_1['model'].append(round_1_client_1)
    client_1['uncertainty'].append(client_1_uncertainty)
    # client 3
    conf.use_local_id = 2
    wind_train_client_2 = get_data.get_wind_train_dataset(conf) 
    round_1_client_2, client_2_uncertainty = run_train_tensorflow(conf, wind_train_client_2, wind_test_dataset, global_model)
    print("trained model " + str(conf.use_local_id))
    client_2['model'].append(round_1_client_2)
    client_2['uncertainty'].append(client_2_uncertainty)

    print("Directory : ", conf.model_dir)
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
    
    model_weights = [model.get_weights() for model in models_array]
    print("Round 1: Collect model weights from each model")
    # Average model weights
    average_weights2 = average_weights(model_weights)
    global_model.set_weights(average_weights2)
    print("Round 1: GLobal model set weights")

    for model in models_array:
        print(model)
        print(model.get_weights()[0])  # Print the first layer weights of each model
        print()
        print()
        print()
    print("GLOBAL MODEL")
    print(global_model.get_weights()[0])


    #####################################################################################################################
    # #### ROUND 3
    print("ROUND 3")
    conf.round = 2
    model_path = model_dir + "communication_round_%03d/" % conf.round 
    conf.model_dir = model_path

    # client 1
    conf.use_local_id = 0
    wind_train_client_0 = get_data.get_wind_train_dataset(conf) 
    round_1_client_0, client_0_uncertainty = run_train_tensorflow(conf, wind_train_client_0, wind_test_dataset, global_model)
    print("trained model " + str(conf.use_local_id))
    client_0['model'].append(round_1_client_0)
    client_0['uncertainty'].append(client_0_uncertainty)
    # client 2
    conf.use_local_id = 1
    wind_train_client_1 = get_data.get_wind_train_dataset(conf) 
    round_1_client_1, client_1_uncertainty = run_train_tensorflow(conf, wind_train_client_1, wind_test_dataset, global_model)
    print("trained model " + str(conf.use_local_id))
    client_1['model'].append(round_1_client_1)
    client_1['uncertainty'].append(client_1_uncertainty)
    # client 3
    conf.use_local_id = 2
    wind_train_client_2 = get_data.get_wind_train_dataset(conf) 
    round_1_client_2, client_2_uncertainty = run_train_tensorflow(conf, wind_train_client_2, wind_test_dataset, global_model)
    print("trained model " + str(conf.use_local_id))
    client_2['model'].append(round_1_client_2)
    client_2['uncertainty'].append(client_2_uncertainty)

    print("Directory : ", conf.model_dir)
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
    
    model_weights = [model.get_weights() for model in models_array]
    print("Round 1: Collect model weights from each model")
    # Average model weights
    average_weights3 = average_weights(model_weights)
    global_model.set_weights(average_weights3)
    print("Round 1: GLobal model set weights")

    for model in models_array:
        print(model)
        print(model.get_weights()[0])  # Print the first layer weights of each model
        print()
        print()
        print()
    print("GLOBAL MODEL")
    print(global_model.get_weights()[0])