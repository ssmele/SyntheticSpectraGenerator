import pytest
import h5py as h5
import numpy as np
import datetime
import os
import torch

from synthetic_torch_helpers import H5Dataset

@pytest.fixture(scope="module")
def h5_files(request):
    """
    Module level fixture to set up some testing h5 files to use throughout
    the testing process. Provides a dictionary of file names that will
    be present throughout the testing script. All the files are cleanedup after
    testing is completed by the finailzer method.
    """
    now_dt = datetime.datetime.now()

    # Creating file that should be successful in creating H5Dataset from it.
    fn_good = 'test_good_{}.h5'.format(now_dt)
    with h5.File(fn_good, 'w') as f:
        f.create_dataset('test0', shape=(10, 1, 10))
        f['test0'][:] = np.zeros(shape=(10,1,10))
        f.create_dataset('test1',shape=(10, 1, 20))
        f['test1'][:] = np.ones(shape=(10, 1, 20))
        f.create_dataset('test2', shape=(10, 20))
        f['test2'][:] = np.full(shape=(10, 20), fill_value=2)

    # Creating file that should not be sucessful in creating H5Dataset from.
    fn_mismatch = 'test_mismatch_{}.h5'.format(now_dt)
    with h5.File(fn_mismatch, 'w') as f:
        f.create_dataset('test0', shape=(5, 1, 10))
        f['test0'][:] = np.zeros(shape=(5,1,10))
        f.create_dataset('test1',shape=(11, 1, 20))
        f['test1'][:] = np.ones(shape=(11, 1, 20))
        f.create_dataset('test2', shape=(10, 20))
        f['test2'][:] = np.full(shape=(10, 20), fill_value=2)

    def cleanup():
        os.remove(fn_good)
        os.remove(fn_mismatch)
    request.addfinalizer(cleanup)

    return { "good": fn_good, "mismatch": fn_mismatch }

def test_no_keys(h5_files):
    """
    Test to ensure when no keys are provided class is no succesfully created.
    """
    with pytest.raises(ValueError, match="Keys must be atleast length one."):
        d = H5Dataset(h5_files['good'], keys=[])

def test_mismatch(h5_files):
    """
    Test to ensure when all datasets in h5 are not of the same length then
    the class is not successfully created.
    """
    with pytest.raises(ValueError,
                       match="All datasets must be of the same length"):
        d = H5Dataset(h5_files['mismatch'], keys=['test0', 'test1', 'test2'])

def test_loaded(h5_files):
    """
    Test to ensure dataset are loaded to memory if specified.
    """
    d = H5Dataset(h5_files['good'], keys=['test0', 'test1'],
                  load_to_memory=True)
    assert isinstance(d.test0, np.ndarray)
    assert isinstance(d.test1, np.ndarray)

def test_infile(h5_files):
    """
    Test to ensure dataset is not loaded to memory if specified.
    """
    d = H5Dataset(h5_files['good'], keys=['test0', 'test1'],
                 load_to_memory=False)
    # TOOD: Gotta be a better way to test that this is a h5 object.
    assert isinstance(d.test0, h5.Dataset)
    assert isinstance(d.test1, h5.Dataset)

def test_no_file():
    """
    Test to ensure class is not successfully created if file doesnt exist.
    """
    with pytest.raises(ValueError, match="Can't find file."):
        d = H5Dataset("not_there.h5", keys=['1', '2'])

def test_good(h5_files):
    """
    Test to ensure a valid h5 file operates well as a pytorch dataset.
    """
    d = H5Dataset(h5_files['good'], keys=['test0', 'test1', 'test2'])

    # Ensuring we can extract correct properties of the dataset.
    assert len(d) == 10
    assert d.test0.shape == (10, 1, 10)
    assert d.test1.shape == (10, 1, 20)
    assert d.test2.shape == (10, 20)
    assert len(d[0]) == 3
    assert d[0][0].shape == (1, 10)
    assert d[0][1].shape == (1, 20)
    assert d[0][2].shape == (20,)
    assert list(map(type, d[0])) == [torch.Tensor]*3

