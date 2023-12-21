# Federated Learning 02456
Deep Learning 02456 final project code repository.

This repository contains the code and results for a project focused on federated learning model performance and uncertainty estimation.

## Contents
- `final_results.ipynb`: Jupyter Notebook with the results of the federated learning model performance compared to a centralized model, including an illustration of uncertainty estimation.
- `run_cifar.sh`: Starting script that initiates the entire framework.
- `train_wind_models.py`: Core script containing the logic for server-client federated learning model training.
- `fl_utils.py`: Script containing the EvidentialLearning framework modules and utility functions.
- `configs.py`: Configuration settings for the training.
- `clients_single_blade`: Dataset directory.
- `experiments_wind/experiment_1`: Directory containing all federated learning test results. The `2023_11_27` subdirectory includes the most recent complete model.

## Getting Started
To run the project, execute the `run_cifar.sh` script. This will set up and start the training process as per the configurations defined in `configs.py`.

## Results
The `final_results.ipynb` Notebook provides an in-depth analysis and comparison of the federated learning models versus centralized models. It also includes visualizations for understanding the uncertainty estimations in the models' predictions.

## Dataset
The dataset used for this project is located in the `clients_single_blade` directory. It is tailored for federated learning scenarios with a focus on wind model predictions.

## Experiment Results
Detailed results from the experiments can be found in the `experiments_wind/experiment_1` directory. The latest and most comprehensive results are in the `2023_11_27` subdirectory.

---

This project is part of the Deep Learning 02456 course, focusing on the application and implications of federated learning in deep learning models.
