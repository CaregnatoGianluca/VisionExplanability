import torch
from torch.utils.data import Dataset
import h5py
import scipy.io
import numpy as np
import torchvision.transforms as T

class MatFileDataset(Dataset):
    """
    A PyTorch Dataset for loading images and labels from a .mat file.
    The dataset supports both grayscale and color images, and can handle different data types.
    The images are resized to a target size specified in the configuration.
    The class also supports different loading engines (h5py or scipy) based on the the specific type of the dataset.
    """
    def __init__(self, file_path, config):

        self.config = config
        # images and labels of the dataset
        self.images = []
        self.labels = []
        
        print(f"Loading '{file_path}' using '{config['loader_engine']}' engine")

        if config['loader_engine'] == 'h5py':

            with h5py.File(file_path, 'r') as mat_file:
                images_cell = mat_file[mat_file['DATA'][0, 0]]
                labels_array = mat_file[mat_file['DATA'][1, 0]][:]
                # get all the images from the dataset
                for i in range(images_cell.shape[0]):
                    image = np.array(mat_file[images_cell[i, 0]])
                    if len(image.shape) == 3:  
                        image = np.transpose(image, (1, 2, 0))  # we save the images from HDF5 (C, H, W) to (H, W, C) numpy standard.
                    self.images.append(image)

        elif config['loader_engine'] == 'scipy':

            mat_data = scipy.io.loadmat(file_path)
            images_cell = mat_data['DATA'][0, 0]
            labels_array = mat_data['DATA'][0, 1]
            for i in range(images_cell.shape[1]):
                self.images.append(images_cell[0, i])
        
        # get all the lables from the dataset (starting from 0) and count how many different classes there are.
        self.labels = (labels_array.flatten() - 1).astype(int)
        self.num_classes = len(np.unique(self.labels))

        # Trasformation to resize the images to a fixed size defined in config['target_size'].
        # It is necessary because Mamba uses inputs with a fixed size (d_model)
        self.resize_transform = T.Resize(config['target_size'], antialias=True)
        
        is_color = config['image_type'] == 'color'
        # Check if the images of the dataset are effectively a colored images (check if the config information is correct)
        first_image_shape = self.images[0].shape
        actual_is_color = len(first_image_shape) == 3 and first_image_shape[2] == 3  # one colored image has shape (H, W, C = 3)
        if is_color != actual_is_color:
            raise ValueError(
                f"Mismatch for '{config['experiment_name']}': config expects '{config['image_type']}' images, "
                f"but the data looks like {'color' if actual_is_color else 'grayscale'}."
            )
        
        # d_model is the size of the vector that represents each row of the image.
        if is_color:
             self.d_model = config['target_size'][1] * 3
        else:
             self.d_model = config['target_size'][1]
        
        print(f"Dataset loaded: {len(self.images)} samples, {self.num_classes} classes. d_model set to {self.d_model}.")



    """
    Returns the number of samples in the dataset.
    """
    def __len__(self):
        return len(self.images)
    


    """
    Returns the sample at the specified index ready for the mamba model.

    The sample is a tuple of the image tensor and the label tensor.
    The image tensor is resized to the target size and converted to a float tensor.
    The label tensor is converted to a long tensor.
    """
    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        # Process images for Pytorch: 
        # we read read images of shape (H, W, 3) and (H, W, 1) and we convert it in a Pytotch tensor of shape (3, H, W) o (1, H, W)
        
        if len(image.shape) == 2: # if it is grayscale image, add a "channel" dimension turning (H, W) into (H, W, 1)
            image = np.expand_dims(image, axis=2) 
        image_tensor = torch.from_numpy(image.copy()).float()
        image_tensor = image_tensor.permute(2, 0, 1) # We use .permute() to temporarily swap the dimensions from (H, W, C) to (C, H, W) 
        
        
        # Process images for the Mamba model: the image is treated as a sequence of rows of lenght d_model
        resized_tensor = self.resize_transform(image_tensor)
        if resized_tensor.shape[0] == 3:
            mamba_input = resized_tensor.permute(1, 2, 0).flatten(start_dim=1) # shape (H, W * 3)
        else:
            mamba_input = resized_tensor.squeeze(0) # shape (H, W)

        # Image normalization
        mamba_input = mamba_input / 255.0 
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        return mamba_input, label_tensor