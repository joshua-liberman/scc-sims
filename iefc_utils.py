import numpy as np
from astropy.io import fits
from hcipy import *

def make_pairwise_probing_sensor(probe_1=400, probe_2=401, probe_amp=None, dm=None, optical_system=None):
    """Generate a sequence of pairwise probe, difference images

    Args:
        probe_1 (int, optional): Probe 1. Defaults to 400.
        probe_2 (int, optional): Probe 2. Defaults to 401.
        probe_amp (_type_, optional): Probe amplitude.
        dm (_type_, optional): Specify DM.
        optical_system (_type_, optional): Specify the optical system.

    Returns:
        _type_: A PWP sensor
    """
    
    # Generate a 'perform_pwp' function that can be evaluated for a given WF
    def func(wavefront):
        dm.flatten()

        difference_images = []
        for probe_pattern in [probe_1, probe_2]:
            # Apply +/- probe 1
            probed_psfs = []
            for amp in [-probe_amp, probe_amp]:

                dm.actuators[probe_pattern] += amp # Apply +/- probe 1 
                # Propagate WF through system
                dm_wf = dm.forward(wavefront) # WF after DM
                psf = optical_system(dm_wf).power
                probed_psfs.append(psf)
                dm.actuators[probe_pattern] -= amp

            # Compute difference image 1
            diff_image = probed_psfs[1] - probed_psfs[0]
            difference_images.append(diff_image)

        return difference_images

    return func

def make_scc_sensor(optical_system_scc=None, optical_system_vort=None, optical_system_pinhole=None):
    """Generate an SCC difference image
     Args:
        pinhole_pos (array): Pinhole position.
        dm (_type_, optional): Specify DM.
        optical_system_scc (_type_,): Specify the SCC optical system.
        optical_system_vort (_type_,): Specify the vortex optical system.

    Returns:
        _type_: An SCC sensor

    """
    def func(wavefront):
        # dm.flatten()
        
        
        # difference_image = []
        psf_scc = optical_system_scc(wavefront).power
        psf_lyot = optical_system_vort(wavefront).power
        psf_pinhole = optical_system_pinhole(wavefront).power



        # sideband_psf_phase = np.imag(sideband_psf)

        return psf_scc, psf_lyot, psf_pinhole
    return func

def extract_measurement_from_scc_image(psf_scc, psf_lyot, psf_pinhole, dark_hole_mask=None):

    psf_diff = psf_scc - psf_lyot - psf_pinhole
    
    wf_measurement = psf_diff[dark_hole_mask]
   
    return wf_measurement

def extract_measurement_from_difference_images(difference_images, dark_hole_mask=None, number_of_probes=2):
    """Create function which will obtain measurements from PWP images

    Args:
        difference_images (array): + and - probe images
        dark_hole_mask (array, optional): The dark hole mask being used.
        number_of_probes (int, optional): Defaults to 2.

    Returns:
        array: Array of PWP measurements
    """

    difference_dark_hole_pixels = [difference_images[i][dark_hole_mask] for i in range(number_of_probes)]   # Make list of diff DH images
    pwp_measurement = np.concatenate(difference_dark_hole_pixels)   # Combine real + imagi parts into a list
    return pwp_measurement