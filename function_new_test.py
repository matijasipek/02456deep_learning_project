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
    average_weights = []

    # Number of layers
    num_layers = len(model_weights[0])

    for layer in range(num_layers):
        # Collect this layer's weights from each model
        layer_weights = np.array([weights[layer] for weights in model_weights])

        # Average layer weights across all models
        layer_average = np.mean(layer_weights, axis=0)

        average_weights.append(layer_average)

    return average_weights

def run_round(conf, round_number, client_0, client_1, client_2, global_model, wind_test_dataset):
    model = global_model
    print(f"ROUND {round_number}")
    conf.round = round_number
    model_path = model_dir + f"communication_round_%03d/" % conf.round
    conf.model_dir = model_path

    clients = [client_0, client_1, client_2]

    for i in range(len(clients)):
        conf.use_local_id = i
        wind_train_client = get_data.get_wind_train_dataset(conf) 
        trained_model, client_uncertainty = run_train_tensorflow(conf, wind_train_client, wind_test_dataset, model)
        print("trained model " + str(conf.use_local_id))
        clients[i]['model'].append(trained_model)
        clients[i]['uncertainty'].append(client_uncertainty)

    print("Directory : ", conf.model_dir)
    files = os.listdir(conf.model_dir)
    models_array = []
    for file in files:
        load_model = fl_utils.initial_model_tensorflow()
        
        if not load_model.built:
            input_shape = (2,) # adjust this as per your requirements
            if input_shape is None:
                raise ValueError("Input shape must be specified in 'conf' to build the model.")
            load_model.build((None,) + input_shape)
        
        load_model = tf.keras.models.load_model(conf.model_dir + file, custom_objects={'EvidentialRegressionLoss': fl_utils.EvidentialRegressionLoss})
        models_array.append(load_model)
    
    model_weights = [model.get_weights() for model in models_array]
    print(f"Round {round_number}: Collect model weights from each model")
    # Average model weights
    average_weights1 = average_weights(model_weights)  # assuming fl_utils.average_weights is your function to average weights
    model.set_weights(average_weights1)
    print(f"Round {round_number}: Global model set weights")

    for i in models_array:
        print(i)
        print(i.get_weights()[0])  # Print the first layer weights of each model
        print()
        print()
        print()
    print("GLOBAL MODEL")
    print(model.get_weights()[0])

    return model



if __name__ == "__main__":
    ### CONFIGURATION
    conf = configs.give_args()
    model_mom = "experiments_wind/"

    conf.folder_name = "experiment_1" 
    conf.dir_name = "new_test_function"
    
    model_dir = model_mom + "%s/%s/" % (conf.folder_name, conf.dir_name) 
    model_path = model_dir + "communication_round_%03d/" % conf.round 
    conf.model_dir = model_path

    print(conf.model_dir)

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


    initial_model = fl_utils.initial_model_tensorflow()
    if not initial_model.built:
        input_shape = (2,)
        if input_shape is None:
            raise ValueError("Input shape must be specified in 'conf' to build the model.")
        initial_model.build((None,) + input_shape) 
    wind_test_dataset = get_data.get_wind_test_dataset() # train

    # Usage example
    #def run_round(conf, round_number, client_0, client_1, client_2, global_model, wind_test_dataset):

    global_model_round_0 = run_round(conf,0, client_0, client_1, client_2, initial_model, wind_test_dataset)
    print(global_model_round_0.get_weights())

    global_model_round_1 = run_round(conf,1, client_0, client_1, client_2, global_model_round_0, wind_test_dataset)
    print(global_model_round_1.get_weights())
    
    global_model_round_2 = run_round(conf,2, client_0, client_1, client_2, global_model_round_1, wind_test_dataset)
    print(global_model_round_2.get_weights())
    
    global_model_round_3 = run_round(conf,3, client_0, client_1, client_2, global_model_round_2, wind_test_dataset)
    print(global_model_round_3.get_weights())


