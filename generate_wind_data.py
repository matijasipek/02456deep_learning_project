import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import os
import torch 
import fl_utils as fl_utils 
import torch.nn as nn 
import get_data as gd
import tensorflow as tf
import evidential_deep_learning as edl
#import evidential_learning_pytorch as edl_pytorch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

device=torch.device("cpu")

### EVIDENTIAL REGRESSION STUFF
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
    x_test = x_test.reshape(-1,)    
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

def turbine_power_formula(df, blade_lengths):
    # Constants
    Cp = 0.25  # coefficient of performance 
    rho = 1.225  # Air density in kg/m^3 at sea level
    cut_in_speed = 3.5  # m/s
    rated_speed = 13.5  # m/s
    cut_out_speed = 25.0  # m/s
    noise_level = 0.05 # noise level as a percentage of power output
    # Empty list to store the rows
    rows = []
    
    for _, row in df.iterrows():
        for blade_length in blade_lengths:
            wind_speed = row['x_wind_speed']
            
            if cut_in_speed <= wind_speed < rated_speed:
                power = 0.5 * Cp * rho * np.pi * (blade_length**2) * (wind_speed**3)
                noise = np.random.normal(0, noise_level * power)
                power += noise
            elif wind_speed >= rated_speed and wind_speed < cut_out_speed:
                power = 0.5 * Cp * rho * np.pi * (blade_length**2) * (rated_speed**3)
                noise = np.random.normal(0, (noise_level / 200) * power)
                power += noise
            else:
                power = 0

            power_kW = power / 1000
            power_MW = power_kW / 1000
            
            # Append the data to the list
            rows.append(row.tolist()+[wind_speed, blade_length, power, power_kW, power_MW])
    
    # Convert list of rows to DataFrame
    columns = df.columns.tolist()+['Wind Speed (m/s)', 'Blade Length (m)', 'Power Output (W)', 'Power Output (kW)', 'Power Output (MW)'] 
    output_df = pd.DataFrame(rows, columns=columns)
    output_df = output_df.sort_values(by=['Wind Speed (m/s)', 'Blade Length (m)']).reset_index(drop=True)
    return output_df

def data_initialization(x_min, x_max, n, blade_lengths):
    # Generate wind speed data
    x = np.linspace(x_min, x_max, n)
    df = pd.DataFrame(x, columns=['x_values']) 

    # CONSTANTS    
    Systematic_Bias_R = 1.5  #  systematic bias
    MULT_wind = 1.20  # the noise in wind speed measurements

    sigma_X = 0.5  #  noise in wind speed measurement
    sigma_Y = 1.5 # noise for power output
    sigma_Model = 3  # Slightly increase model noise

    beta0 = 0.5  # Intercept 
    beta1 = 1.67 #  the slope 
    
    # Generate random wind speeds with noise
    random_wind_speeds = np.random.normal(df['x_values'], sigma_X) * Systematic_Bias_R * MULT_wind

    # Ensure generated values are always positive
    df['x_wind_speed'] = random_wind_speeds + np.abs(np.min(random_wind_speeds)) + 0.25  # Adding 1 to avoid zeros
    #true power output
    df['y_ground_truth']=np.random.normal(beta0+df['x_values']*beta1, 0)
    # observed power output with noise and bias
    df['observed_data']=np.abs(np.random.normal((beta0)+(df['x_wind_speed'])*(beta1), sigma_Y)*Systematic_Bias_R) #MEASUREMENTS WITH ALEATORIC and SYSTEMATIC BIAS
    # prediction
    df['model']=np.abs(np.random.normal((beta0)+df['x_wind_speed']*beta1, sigma_Model))
    
    # Calculate power by formula
    df = turbine_power_formula(df, blade_lengths)
    # Sort values
   # df = df.sort_values(by='x_wind_speed')
    return df

def initialize_visualize():
    # Create a directory for the client data files
    os.makedirs('clients', exist_ok=True)

    for i in range(10):
        blade_lengths = [random.randint(40, 220) for _ in range(6)]
        blade_lengths.sort()
        
        x_max = random.randint(15, 25)
        x_min = 0
        n = 4096

        client = data_initialization(x_min, x_max, n, blade_lengths)
        # Save each client's data to a separate JSON file
        with open(f'clients/client_{i}.json', 'w') as f:
            f.write(client.to_json(orient='records', lines=True))
    #Load the data
    df_clients = {}
    for i in range(10):
        with open(f'clients/client_{i}.json', 'r') as f:
            df_clients[f'client_{i}'] = pd.read_json(f, orient='records', lines=True)


    ### HELPERS
    # print(len(train_dataset_client_tensor))
    # print(data_loader)
    # for i, data in enumerate(data_loader):
    #     print(len(data))
    # specific_client_data = df_clients["client_" + str(local_id)]
    # print(f"{specific_client_data}:\n{specific_client_data.head(30).to_string()}\n")
    

    # Plot the data for each client
    for client_name, df in df_clients.items():
        plt.figure(figsize=(10, 6))
        
        blade_lengths = df['Blade Length (m)'].unique()
        for blade_length in blade_lengths:
            subset = df[df['Blade Length (m)'] == blade_length]
            plt.plot(subset['Wind Speed (m/s)'], subset['Power Output (MW)'], '-o', label=f'Blade Length: {blade_length}m')

        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power Output (MW)')
        plt.title(f'Power Output vs. Wind Speed for Various Blade Lengths for {client_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

def create_test_dataset():
    # Create a directory for the client data files
    os.makedirs('test_dataset', exist_ok=True)


    blade_lengths = [random.randint(40, 220) for _ in range(16)]
    blade_lengths.sort()
    
    x_max = random.randint(15, 25)
    x_min = 0
    n = 40960

    # Generate data for each client
    client = data_initialization(x_min, x_max, n, blade_lengths)
    
    client = client.to_json(orient='records', lines=False)
    print(type(client))
    print(client)

    with open('test_dataset/data.json', 'w') as file:
        file.write(client)
        print("Global test set saved in 'test_dataset/test.json'")

def load_test_dataset():
    # Read the JSON file
    with open('test_dataset/data.json', 'r') as file:
        json_str = file.read()
    # Convert JSON string to DataFrame
    df = pd.read_json(json_str, orient='records')

    # 3) Convert DataFrame to Tensor
    test_dataset_tensor = torch.tensor(df.values)

    # 4) Create DataLoader for test dataset
    test_data_loader = torch.utils.data.DataLoader(test_dataset_tensor, 
                                                batch_size=1024, 
                                                shuffle=False, 
                                                num_workers=4, 
                                                pin_memory=False, # NO CUDA
                                                drop_last=True)
    return test_data_loader

def run_train(conf, tr_loader, tt_loader, exist_model): 
    """
    model_dir: ../exp_data/../communication_round_%02d/    
    """    
    print("===========================================================")
    print("                    Local ID %02d " % conf.use_local_id)
    print("===========================================================")

    print("The used batch size %d for client %d at round %d" % (conf.batch_size, conf.use_local_id, conf.round))

    # TRAIN CUSTOM MODEL HERE
    train_obj = Train(conf, [tr_loader, tt_loader], conf.num_local_epochs, conf.sigma, exist_model)
    client_model = train_obj.run()
    print("Done Local ID %02d" % conf.use_local_id )
    torch.save(client_model, 
            conf.model_dir + "/client_id_%02d.pt" % conf.use_local_id)
    return client_model

class Evidential(nn.Module):
    def __init__(self, num_neurons):
        super(Evidential, self).__init__()
        self.num_neurons = num_neurons
        self.layer_1 = nn.Linear(num_neurons, num_neurons)
        self.layer_2 = nn.Linear(num_neurons, num_neurons)
        self.output_layer = edl.DenseNormalGamma(1)
    
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.output_layer(x)
    
def get_wind_train_dataset(conf):
    local_id = conf["use_local_id"]
    train_dataset_client = []

    # Read in subset based on local_id
    with open(f'clients_single_blade/client_{local_id}.json', 'r') as f:
            train_dataset_client = pd.read_json(f, orient='records', lines=True)

    # Convert JSON string to DataFrame
    return train_dataset_client

def get_wind_test_dataset():
    train_dataset_client = []

    # Read in subset based on local_id
    with open(f'clients_single_blade/test_data.json', 'r') as f:
            train_dataset_client = pd.read_json(f, orient='records', lines=True)

    return train_dataset_client
    
def initial_model_tensorflow(conf, device=torch.device("cuda")):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(conf['units'], activation='relu'),
        tf.keras.layers.Dense(conf['units'], activation='relu'),
        edl.layers.DenseNormalGamma(1)
    ])
    return model

def run_train_tensorflow(conf, train_data, test_data, exist_model):
    exist_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(5e-4), loss=EvidentialRegressionLoss)
    # TRAIN
    client_train_data_x = train_data[['Wind Speed (m/s)']]
    client_train_data_y = train_data[['Power Output (MW)']]
    # TEST
    test_data_x = test_data[['Wind Speed (m/s)']]
    test_data_y = test_data[['Power Output (MW)']]
    
    exist_model.fit(client_train_data_x.values, client_train_data_y.values,  batch_size=conf['batch_size'], epochs=conf['num_local_epochs'], verbose=1)
    print("Done Local ID %02d" % conf['use_local_id'])
    model_evaluation_tensorflow(test_data_x, exist_model)
    exist_model.save(conf['model_dir'] + "/client_id_%02d.pt" % conf['use_local_id'])
    return exist_model

def model_evaluation_tensorflow(test_data_x, exist_model):
    prediction = exist_model.predict(test_data_x.values)
    x_test = test_data_x.values[:, 0]
    mu, v, alpha, beta = tf.split(prediction, 4, axis = 1) # Hyperparameters of evidential distributions

    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))#
    var = np.minimum(var, 1e3)[:, 0]

    df_uncertainty = fl_utils.get_uncertainty_estimation(prediction, x_test)
    print(df_uncertainty)

def federated_averaging(conf, client_models):
    n_clients = conf['n_clients']
    #client_models = [model_parameters, model2_parameters, model3_parameters]  

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

def aggregate_the_model(conf, model_group):
    num_params = len(model_group)

    # Create a new model instance
    averaged_model = fl_utils.create_model(conf)

    if not averaged_model.built:
        input_shape = conf.get('input_shape')
        if input_shape is None:
            raise ValueError("Input shape must be specified in 'conf' to build the model.")
        averaged_model.build((None,) + input_shape)

    # Set the averaged weights
    averaged_model_weights = [model_group[f'param_{i}'] for i in range(num_params)]
    averaged_model.set_weights(averaged_model_weights)

    return averaged_model

def test_aggregated_model(aggregated_model):
    # data
    wind_test_dataset = get_wind_test_dataset()
    test_data_x = wind_test_dataset[['Wind Speed (m/s)']]
    test_data_y = wind_test_dataset[['Power Output (MW)']].values

    # model
    final_prediction = aggregated_model.predict(test_data_x.values)

    x_test = test_data_x.values[:, 0]
    mu, v, alpha, beta = tf.split(final_prediction, 4, axis = 1) # Hyperparameters of evidential distribution
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))#
    var = np.minimum(var, 1e3)[:, 0]

    df_uncertainty = get_uncertainty_estimation(final_prediction, x_test)
    edl_loss = edl.losses.EvidentialRegression(test_data_y, mu, coeff=1e-2)
    # Convert test_data_y to numpy array

    # Calculate MSE
    mse = mean_squared_error(test_data_y, mu)
    return mse, df_uncertainty, edl_loss

if __name__ == "__main__":
    
    model_mom = "experiments_wind/"

    folder_name = "experiment_1" 
    dir_name = "2023_11_23"
    
    model_dir = model_mom + "%s/%s/" % (folder_name, dir_name)     

    conf = {
        "use_local_id": 2,
        "round": 0,
        "batch_size": 24,
        "sigma": 0,
        "num_local_epochs": 10,
        "lr": 0.1,
        'units': 64,
        'n_clients': 4,
        "model_dir": model_dir,
        'input_shape': (1,)  # Example input shape, change as needed
    }
    # # Load data
    wind_train_subset = get_wind_train_dataset(conf)
    # wind_test_dataset = get_wind_test_dataset()
    
    # # Split the trian data
    # client_train_data1_x = wind_train_subset[['Wind Speed (m/s)']]
    # client_train_data1_y = wind_train_subset[['Power Output (MW)']]
    
    # # Test data
    # test_data_x = wind_test_dataset[['Wind Speed (m/s)']]
    # test_data_y = wind_test_dataset[['Power Output (MW)']]
    
    # exist_model = fl_utils.initial_model_tensorflow() 
    # _model = run_train_tensorflow(conf, wind_train_subset, wind_test_dataset, exist_model)
    
    
############### figure out how to save the model and check the server setup    
    # print(model_dir)
    # print(conf['model_dir'] + "client_id_%02d.pt" % (1))
    # print(os.path.isdir(conf['model_dir'] + "/client_id_%02d.pt" % (1)))
    
    # first = np.sum([os.path.isdir(conf['model_dir'] + "client_id_%02d.pt" % (j)) for j in range(conf['n_clients'])])
    # second = conf['n_clients']
    
    # files = os.listdir(model_dir)
    # print(files)
    # print(len(files))
    # x = (first == second)
    # print(first)
    # print(second)
    # print(x)
    

    # n_clients = conf['n_clients']
    


##### READ IN THE MODELS
#     files = os.listdir(model_dir)
#     models_array = []
#     for file in files:
#         load_model = fl_utils.create_model(conf)
        
#         if not load_model.built:
#             input_shape = conf.get('input_shape')
#             if input_shape is None:
#                 raise ValueError("Input shape must be specified in 'conf' to build the model.")
#             load_model.build((None,) + input_shape)
        
#         load_model = tf.keras.models.load_model(model_dir+file, custom_objects={'EvidentialRegressionLoss': EvidentialRegressionLoss})
#         models_array.append(load_model)
#     print("READ IN THE MODELS")
# # ##### GET THE PARAMETERS
#     model_parameters = []
#     for model in models_array: 
#         model_params = {weight.name: weight.numpy() for weight in model.weights} # extract parameters from model
#         model_parameters.append(model_params)
#     print("GOT THE PARAMETERS")
#     # print(len(model_parameters))
#     # print(model_parameters)
#     final_model = federated_averaging(conf, model_parameters)
#     final_model.save(conf['model_dir'] + "/aggregated_model.pt")

#     print(final_model)
#     print(final_model.summary())
#     print("FINAL MODEL")

#     wind_test_dataset = get_wind_test_dataset()
#     test_data_x = wind_test_dataset[['Wind Speed (m/s)']]
#     test_data_y = wind_test_dataset[['Power Output (MW)']]
#     mse, uncertainty, edl_loss = test_aggregated_model(final_model)
#     print("Server MSE: ", mse)
#     print()
#     print("Server EDL Loss: ", edl_loss)
#     print()
#     print("Server Uncertainty: ", uncertainty)

######## DONE WITH ROUND 0
    # final_prediction = final_model.predict(test_data_x.values)

    # x_test = test_data_x.values[:, 0]
    # mu, v, alpha, beta = tf.split(final_prediction, 4, axis = 1) # Hyperparameters of evidential distributions

    # mu = mu[:, 0]
    # var = np.sqrt(beta / (v * (alpha - 1)))#
    # var = np.minimum(var, 1e3)[:, 0]
    
    ###### load all data
    # train_dataset_client = []

    # # Read in subset based on local_id
    # for i in range(4):
    #     local_id = i
    #     with open(f'clients_single_blade/client_{local_id}.json', 'r') as f:
    #             train_dataset_client.append(pd.read_json(f, orient='records', lines=True))

    # # Combining X data (features)
    # for i in range(4):
    #     if i == 0:
    #         client_train_data1_x = train_dataset_client[i][['Wind Speed (m/s)']]
    #     elif i == 1:
    #         client_train_data2_x = train_dataset_client[i][['Wind Speed (m/s)']]
    #     elif i == 2:
    #         client_train_data3_x = train_dataset_client[i][['Wind Speed (m/s)']]
    #     elif i == 3:
    #         client_train_data4_x = train_dataset_client[i][['Wind Speed (m/s)']]
    # combined_train_data_x = pd.concat([client_train_data1_x, client_train_data2_x, client_train_data3_x], ignore_index=True)

    # for i in range(4):
    #     if i == 0:
    #         client_train_data1_y = train_dataset_client[i][['Power Output (MW)']]
    #     elif i == 1:
    #         client_train_data2_y = train_dataset_client[i][['Power Output (MW)']]
    #     elif i == 2:
    #         client_train_data3_y = train_dataset_client[i][['Power Output (MW)']]
    #     elif i == 3:
    #         client_train_data4_y = train_dataset_client[i][['Power Output (MW)']]
    # # Combining Y data (labels)
    # combined_train_data_y = pd.concat([client_train_data1_y, client_train_data2_y, client_train_data3_y], ignore_index=True)

   # df_uncertainty = get_uncertainty_estimation(final_prediction, x_test)
   # plot_predictions(combined_train_data_x.values, combined_train_data_y.values, test_data_x.values, test_data_y.values, final_prediction)
    #print(df_uncertainty)

    model_path = conf['model_dir'] + "/aggregated_model.pt"
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'EvidentialRegressionLoss': EvidentialRegressionLoss})
    print(loaded_model.summary())

    wind_test_dataset = get_wind_test_dataset()
    test_data_x = wind_test_dataset[['Wind Speed (m/s)']]
    test_data_y = wind_test_dataset[['Power Output (MW)']]
    mse, uncertainty, edl_loss = test_aggregated_model(loaded_model)
    print("Server MSE: ", mse)
    print()
    print("Server EDL Loss: ", edl_loss)
    print()
    print("Server Uncertainty: ", uncertainty)
