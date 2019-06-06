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
            self.flux = file['flux'][:]
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

    def __init__(self, conv_config, full_config):
        super(ConvModSyn, self).__init__()
        self.conv_layers = []
        for c_in, c_out, ks in conv_config:
            self.conv_layers.append(torch.Conv1d(c_in, c_out, ks))

        self.full_connnected_layers = []
        for f_in, f_out in full_config:
            self.fully_connected_layers.append(torch.nn.Linear(f_in, f_out))

        self.act = torch.nn.Sequential()

    def forward(self, x):
        for cl in self.conv_layers:
            x = cl(x)
            x = self.act(x)

        for fl in self.full_connected_layers:
            x = fl(x)

        return x

