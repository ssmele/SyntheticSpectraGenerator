from torch.utils.data import Dataset
from torch import torch

import h5py as h5

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
        return torch.from_numpy(self.fluxs[idx]), self.zs[idx]

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
            if ix in pooling_ixs:
                self.conv_layers.add_module("pool_{}".format(ix),
                        torch.nn.MaxPool1d(2))

        # Set up the fully connected layers.
        self.fc_layers = torch.nn.Sequential()
        for ix, (f_in, f_out) in enumerate(full_config[:-1]):
            self.fc_layers.add_module("fc_{}".format(ix),
                    torch.nn.Sequential(
                        torch.nn.Linear(f_in, f_out),
                        torch.nn.ReLU()))
            # If we are at a dropout ix location add it.
            if ix in dropout_ixs:
                self.fc_layers.add_module("dropout_{}".format(ix),
                        torch.nn.Dropout(.5))

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

