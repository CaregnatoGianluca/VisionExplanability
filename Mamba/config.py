
"""
Configuration file for Mamba experiments with different datasets.
This file defines various configurations for running experiments with the Mamba Vision Classifier.
Each configuration includes parameters such as file paths, image types, loader engines, and model dimensions.
"""


shared_config = {
    "target_size": (100, 100),
    "batch_size": 32, 
    "epochs": 15, # Number of epochs for training
    "learning_rate": 2e-4, # Learning rate for the optimizer
    "train_split": 0.7,
    "validation_split": 0.15,
    "test_split": 0.15,
    "d_state": 64, # Dimension of the state in the Mamba layer
    "d_conv": 4, # Dimension of the convolutional layer in the Mamba layer
    "expand": 8, # Expansion factor for the Mamba layer
}

config_2 = {**shared_config, **{
    "experiment_name": "mamba_on_Datas_2",
    "file_path": 'Data/Datas_2.mat',
    "loader_engine": "scipy",
    "image_type": "grayscale", 
}}

config_42 = {**shared_config, **{
    "experiment_name": "mamba_on_Datas_42",
    "file_path": 'Data/Datas_42.mat',
    "loader_engine": "h5py",
    "image_type": "grayscale", 
}}

config_43 = {**shared_config, **{
    "experiment_name": "mamba_on_Datas_43",
    "file_path": 'Data/Datas_43.mat',
    "loader_engine": "h5py",    
    "image_type": "grayscale", 
}}

config_44 = {**shared_config, **{
    "experiment_name": "mamba_on_Datas_44",
    "file_path": 'Data/Datas_44.mat',
    "loader_engine": "h5py",    
    "image_type": "color",
}}

config_37 = {**shared_config, **{
    "experiment_name": "mamba_on_DatasColor_37",
    "file_path": 'Data/DatasColor_37.mat',
    "loader_engine": "scipy",    
    "image_type": "color", 
}}

config_38 = {**shared_config, **{
    "experiment_name": "mamba_on_DatasColor_38",
    "file_path": 'Data/DatasColor_38.mat',
    "loader_engine": "scipy",    
    "image_type": "color", 
}}

config_50 = {**shared_config, **{
    "experiment_name": "mamba_on_DatasColor_50",
    "file_path": 'Data/DatasColor_50.mat',
    "loader_engine": "scipy",    
    "image_type": "color", 
}}

config_72 = {**shared_config, **{
    "experiment_name": "mamba_on_DatasColor_72",
    "file_path": 'Data/DatasColor_72.mat',
    "loader_engine": "h5py",    
    "image_type": "color", 
}}
config_77 = {**shared_config, **{
    "experiment_name": "mamba_on_DatasColor_77",
    "file_path": 'Data/DatasColor_77.mat',
    "loader_engine": "h5py",    
    "image_type": "color", 
}}

"""A list containing all configurations to run experiments
This allows for easy iteration over all configurations in the main script."""
ALL_CONFIGS = [
    config_42,
    config_2,
    config_43,
    config_44,
    config_37,
    config_38,
    config_50,
    config_72,
    config_77,
]