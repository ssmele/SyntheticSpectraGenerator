from torch.utils.data import Dataset
from torch import torch

import h5py as h5
import numpy as np
import os

class SyntheticFluxDataset(torch.utils.data.Dataset):

    def __init__(self, filename, load_to_memory=False):
        """
        Constructor used for synthetic flux dataset. This class assumes the
        dataset is stored in a h5 file given by the filename specified. This
        class also allows you to specify if loading the whole arrays to memory
        is needed.

        :param filename: filename of the h5 file to read datasetfrom.
        :param load_to_memory: boolean flag to determine if we should load
        arrays to the memory
        """
        super(SyntheticFluxDataset, self).__init__()

        # Read in the pertinant information from the dataset
        self.filename = filename
        self.load_to_memory = load_to_memory
        file = h5.File(self.filename, 'r')
        if self.load_to_memory:
            self.fluxs = file['flux'][:]
            self.zs = file['zs'][:]
            file.close()
        else:
            self.fluxs = file.get('flux')
            self.zs = file.get('zs')

    def __len__(self):
        """
        This method returns the length of the dataset.
        """
        return len(self.fluxs)

    def __getitem__(self, idx):
        """
        Method to retrieve an item from the dataset based on the index given.

        :param idx: index of the flux, and label we want to read.
        """
        X = torch.from_numpy(np.expand_dims(self.fluxs[idx], 0))
        y = torch.Tensor([self.zs[idx]])
        return X, y

class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, filename, keys, load_to_memory=False):
        """
        Constructor for a generic dataset that is encapsulated within a
        h5py file. Each key given is assumed to be a dataset within the h5 file
        that will be attached to the class and used as the data for the
        dataset.

        :param filename: location of h5 file.
        :param keys: list of dataset keywords to extract.
        :param load_to_memory: boolean if data should be loaded to memory.
        """
        super(H5Dataset, self).__init__()

        self.filename = filename
        self.keys = keys
        self.load_to_memory = load_to_memory

        # Ensure atleast one key is present.
        if len(self.keys) == 0:
            raise ValueError("Keys must be atleast length one.")

        # Read in the file.
        if not os.path.isfile(self.filename):
            raise ValueError("Can't find file.")
        file = h5.File(self.filename, 'r')

        # Ensure all datasets are of the same length.
        if len(set(len(file[k]) for k in self.keys)) != 1:
            raise ValueError("All datasets must be of the same length.")

        # Go through each key and attribute its data to the class.
        for k in keys:
            if self.load_to_memory:
                # Set a copy of the array onto the class.
                setattr(self, k, file[k][:])
            else:
                # Just a reference to the dataset in the file.
                setattr(self, k, file.get(k))

        # If we loaded into memory we can close the file.
        if self.load_to_memory:
            file.close()

    def __len__(self):
        """
        Gets the length of dataset.
        """
        return len(getattr(self, self.keys[0]))

    def __getitem__(self, idx):
        """
        Constructs an item of the dataset as a tuple in the same order as the
        keys given on dataset construction. Assumes a numpy array is stored
        at each index of the different h5 datasets. Does this so from_numpy can
        be used to generate tensors for the data.

        :param idx: index of the datasets to retrieve.
        """
        return tuple(torch.from_numpy(getattr(self, k)[idx]) for k in self.keys)

class SynH5Dataset(H5Dataset):
    """
    Dataset for synthetic dataset encapsulated in a H5 file. By default assumes
    datasets are labeled 'flux' and 'id'.

    :param filename: File location of the h5 file.
    :param keys: dataset names in h5 file.
    :param load_to_memory: If dataset should be loaded to memory.
    """
    def __init__(self, filename, keys=['flux', 'zs'], load_to_memory=False):
        super(SynH5Dataset, self).__init__(filename, keys, load_to_memory)

class ConvModSyn(torch.nn.Module):

    def __init__(self, conv_config, full_config, pooling_ixs, dropout_ixs,
                 final_act):
        """
        Constructor for the Synthetic Flux Model. Takes in a configuration
        list for the convolutional layers. The configuration is a list of
        tuples that contain the c_in, c_out, and kernel size of the
        corresponding layers.

        A fully connected layer config is needed that is a list of tuples
        with an in and out for each layer.

        Batch norm and relu are applied to each of the convolutions
        and relu is applied to each of the linear layers.

        :param conv_config:
        :param full_config:
        :param pooling_ixs:
        :param dropout_ixs:
        """
        super(ConvModSyn, self).__init__()

        # Set up the convolutional layers.
        self.conv_layers = torch.nn.Sequential()
        for ix, (c_in, c_out, ks) in enumerate(conv_config):
            self.conv_layers.add_module("conv_{}".format(ix),
                    torch.nn.Sequential(
                        torch.nn.Conv1d(c_in, c_out, ks),
                        torch.nn.BatchNorm1d(c_out),
                        torch.nn.ReLU()))
            # If we are at a pooling ix location add it.
            if ix in pooling_ixs.keys():
                self.conv_layers.add_module("pool_{}".format(ix),
                        torch.nn.MaxPool1d(pooling_ixs[ix]))

        # Set up the fully connected layers.
        self.fc_layers = torch.nn.Sequential()
        for ix, (f_in, f_out) in enumerate(full_config[:-1]):
            self.fc_layers.add_module("fc_{}".format(ix),
                    torch.nn.Sequential(
                        torch.nn.Linear(f_in, f_out),
                        torch.nn.ReLU()))
            # If we are at a dropout ix location add it.
            if ix in dropout_ixs.keys():
                self.fc_layers.add_module("dropout_{}".format(ix),
                        torch.nn.Dropout(dropout_ixs[ix]))

        # Add the last layer without relu to apply final act function.
        f_in, f_out = full_config[-1]
        self.fc_layers.add_module("fc_{}".format(len(self.fc_layers)),
                torch.nn.Linear(f_in, f_out))

        self.final_act = final_act

    def forward(self, x):
        # Pass batch of spectra through conv layers.
        for cl in self.conv_layers:
            x = cl(x)

        # Pass flattened images through fully connected layers.
        x = x.view(x.size(0), -1)
        for fl in self.fc_layers:
            x = fl(x)
        x = self.final_act(x)
        return x

