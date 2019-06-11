import numpy as np
import h5py as h5

class FluxDatasetGenerator:
    """
    This class represents a generator for sythetic flux datasets.
    """

    def __init__(self, filename, templates, manga_wave,
                 num_spectra, zs, rs, noises, continuums,
                 track_params = False, description = None):
        """
        Constructor used to set up parameters used when generating dataset.

        :param filename: location to store dataset file.
        :param tempaltes: eigen spectra templates
        :param manga_wave: manga_wave wavelength locations.
        :param num_spectra: number of spectra to generate.
        :param zs: array of possible z-values.
        :param as: array of possible a parameter values.
        :param noises: array of possible noises values.
        :param continuums: array of possible contiuum values.
        :param track_params: boolean that determins if all dataset parameters
        are stored in the dataset.
        :param description: optinal description to attribute to the dataset.
        """
        self.filename = filename
        self.templates = templates
        self.manga_wave = manga_wave
        self.num_spectra = num_spectra
        self.zs = zs
        self.rs = rs
        self.noises = noises
        self.continuums = continuums
        self.track_params = track_params
        self.description = description

    def select_parameter(self, param_cond):
        """
        This method selects a parameter based on the type given. Any type of
        python list or numpy array will have an item selected at random from it.
        If a tuple is given the two values will be used as a min and max
        value with a random integer choosen within [min, max).

        :param param_cond: parameter condition to use.
        """
        if isinstance(param_cond, list) or isinstance(param_cond, np.ndarray):
            return np.random.choice(param_cond)
        elif isinstance(param_cond, tuple) and len(param_cond) == 2 \
                and param_cond[0] <= param_cond[1]:
            return np.random.randint(low=param_cond[0], high=param_cond[1])
        else:
            raise ValueError("Invalid parameter specification")

    def generate_dataset(self):
        """
        This method creates a dataset based on the parameters specified on class
        creation.
        """
        with h5.File(self.filename, "w") as file:
            # Set up datasets for the flux values and its corresponding z value.
            file.create_dataset(
                    'flux',
                    shape=(self.num_spectra, 1, len(self.manga_wave))
                    )
            file.create_dataset('zs', shape=(self.num_spectra, 1))

            # Save both the manga wave and descriptions as attributes
            file.attrs['manga_wave'] = self.manga_wave
            if self.description:
                file.attrs['description'] = self.description

            # If we specified set up datasets for various parameters.
            if self.track_params:
                file.create_dataset('as', shape=(self.num_spectra,))
                file.create_dataset('noises', shape=(self.num_spectra,))
                file.create_dataset('continuums', shape=(self.num_spectra,))

            fg = FluxGenerator(self.templates, self.manga_wave, 0)
            for ix in range(self.num_spectra):
                # Generate a flux value with new randomly choosen dataset params
                fg.z = self.select_parameter(self.zs)
                fg.a = self.select_parameter(self.rs)
                fg.noise = self.select_parameter(self.noises)
                fg.continuum = self.select_parameter(self.continuums)

                # Save it within h5 file.
                file['flux'][ix] = fg.generate()
                file['zs'][ix] = fg.z

                # Track the other data parameters.
                if self.track_params:
                    file['as'][ix] = fg.a
                    file['noises'][ix] = fg.noise
                    file['continuums'][ix] = fg.continuum

class FluxGenerator:
    """
    This module represents a generator for sythetic flux data. The desired
    attributes of the flux should be specified within the constructor.
    To generate a new flux value from the given attributes simply call the
    class object as a function.
    """

    def __init__(self, templates, manga_wave, z,
                 a = None, noise = None, continuum = False):
        """
        Constructor used to set up parameters of the specific flux spectra
        we are trying to correct.

        :param templates: eigen spectra components to use.
        :param manga_wave: manga_wave wavelength values.
        :param z: redshift z-value to shift galaxy too.
        :param a: reddening coefficient to fix galaxy with.
        :param noise: represents std dev of random gaussian noise to add
        independtly to each wavelength value of the spectra.
        :param continuum: boolean to determine if contiuum should be removed.
        If true the contiuum will be removed.
        """
        self.templates = templates
        self.manga_wave = manga_wave
        self.z = z
        self.a = a
        self.noise = noise
        self.continuum = continuum

    def apply_reddening(self, spectra, wave):
        """
        This method applies a fix to the generated spectra assuming it has been
        affected by space dust. Fix is based on wavelength location, and alpha
        parameter.

        :param spectra: spectra to apply reddening to.
        :param wave: wavelength values.
        """
        if spectra.shape != wave.shape:
            raise ValueError("spectra and wave inputs must be same shape.")
        return spectra - (self.a*np.power(wave, -1.5))

    def combine_spectra(self, spectras, weights):
        """
        This method performs a linear combination between the eigen spectras
        given and the specified weights.

        Ex:
        spec1*w1 + spec2*w2 + spec3*w3 + spec4*w4

        :param spectras: spectras to combine.
        :param weights: weights to use as the coefficients for linear combo.
        """
        return np.dot(spectras.T, weights)

    def shift_wave(self, z, wave):
        """
        This method takes given wavelength value and shifts it to the gaven
        z-value.
        """
        return (1 + z)*wave


    def remove_continuum(self, spectra):
        """
        This method removes the continuum from the given flux. Does this with a
        moving median window.

        :param spectra: spectra to remove continuum from.
        """
        return spectra - medfilt(spectra, 51)

    # TODO: Figure out the eign values.
    # TODO: Figure out if the weights should dip into the negatives.
    # TODO: Figure out he correct distribution to sample his from JP says gaus
    def generate_spectra_weights(self, eig_vals=[1,1,1,1,1,1,1]):
        """
        This method generates representative weightings for the given eigen
        spectra templates. These weights are derived from a guassian
        distribution and is further weighted by the eigen values of the
        respective template.
        """
        ws = np.random.rand(len(self.templates))
        #ws = np.random.normal(size=len(self.templates))
        return (ws/ws.sum())*eig_vals

    def base_wave(self):
        """
        This method generates the base flux wavelength values the spectra
        corresponds to asssuming a redshift z value of 0.
        """
        COEFF0 = 3.3385
        COEFF1 = 0.0001
        min_emsn_loc = 10**(COEFF0 + COEFF1*0)
        max_emsn_loc = 10**(COEFF0 + COEFF1*6625)
        return np.linspace(min_emsn_loc, max_emsn_loc, 6625)

    def generate(self):
        """
        This method generates a flux, and wave array based on the current
        attributes specified on the class.
        """
        # Generate a linear combination of the templates.
        spec_weights = self.generate_spectra_weights()
        combined_spectra = self.combine_spectra(self.templates, spec_weights)

        # Shift wave over to specified z-value.
        shifted_wave = self.shift_wave(self.z, self.base_wave())

        # Extract needed wave values from generated flux.
        syn_flux = np.interp(self.manga_wave, shifted_wave, combined_spectra)

        # Remove the continuum of the syn flux if specified.
        if self.continuum:
            syn_flux = self.remove_continuum(syn_flux)

        # Add independent noise to all wavelenght locations.
        if self.noise:
            syn_flux += np.random.normal(scale=self.noise, size=syn_flux.shape)

        # Apply reddening fix if specified.
        if self.a:
            syn_flux = self.apply_reddening(syn_flux, self.manga_wave)

        return syn_flux

