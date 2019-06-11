import pytest
import numpy as np

from flux_gen_redux import FluxGenerator, FluxDatasetGenerator
from astropy.io import fits


template_location = 'spMLpcaGal-55331-7temp.fits'
manga_wave_location = 'manga_wave.npy'

# Testing for reddining.
def test_bad_shape():
    fg = FluxGenerator(None, None, None, None, None)
    with pytest.raises(ValueError):
        fg.apply_reddening(np.array([1,2,3]), np.array([3,4]))

# Testing for combining spectra
def test_combination():
    fg = FluxGenerator(None, None, None, None, None)

    ex_spectra = np.ones(shape=(5, 10))
    combined = fg.combine_spectra(ex_spectra, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(combined, np.full(10, 15.0))

# Testing for shifting wave.
def test_shift_zero_z():
    fg = FluxGenerator(None, None, None, None, None)
    assert np.array_equal(np.ones(5), fg.shift_wave(0, np.array([1,1,1,1,1])))


# Testing for removal of continuum.
# TODO: ADD SOME WHEN FINAL METHOD IS CHOOSEN.

# Testing base wave.
# TODO: ADD SOME WHEN FINAL METHOD IS CHOOSEN.

# Testing generate.
def test_generate_size():
    templates = fits.open(template_location)[0].data
    manga_wave = np.load(manga_wave_location)
    fg = FluxGenerator(templates, manga_wave, 0.12, 10000, 0.01, False)
    assert (4563,) ==  fg.generate().shape

# Testing parameter selection.
def test_select_param_list():
    fdg = FluxDatasetGenerator(None, None, None, None, None, None, None, None)
    assert fdg.select_parameter([1, 2, 3]) in [1, 2, 3]

def test_select_param_numpy():
    fdg = FluxDatasetGenerator(None, None, None, None, None, None, None, None)
    assert fdg.select_parameter(np.array([1, 2, 3])) in [1, 2, 3]

@pytest.mark.parametrize("l, h", [(10, 20), (0, 100), (0, 10)])
def test_select_param_tuple(l, h):
    fdg = FluxDatasetGenerator(None, None, None, None, None, None, None, None)
    assert l <= fdg.select_parameter((l, h)) <= h

@pytest.mark.parametrize("val", [True, 10, 1.0])
def test_select_param_bad(val):
    fdg = FluxDatasetGenerator(None, None, None, None, None, None, None, None)
    with pytest.raises(ValueError):
        fdg.select_parameter(10)
