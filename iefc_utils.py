import numpy as np
from astropy.io import fits
from hcipy import *

def make_magaox_bump_mask(normalized=False, with_spiders=True):
    '''Make the Magellan bump mask.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 6.5 meters.
    with_spiders: boolean
        If this is False, the spiders will be left out.

    Returns
    -------
    Field generator
        The Magellan aperture.
    '''

    # TODO: Magnify bump mask vals. and preserve default vals

    magnification_factor = 6.5/9e-3 # Mag factor to scale 9 mm bump mask up to 6.5 m pupil diameter
    mask_inner = 2.79e-3 * magnification_factor # meter
    mask_outer = 8.604e-3 * magnification_factor # meter

    bump_mask_diameter = 0.5742e-3 * magnification_factor

    bump_mask_pos = [2.853e-3 * magnification_factor, -0.6705e-3 * magnification_factor] 

    radius = np.hypot(bump_mask_pos[0], bump_mask_pos[1])
    theta = np.arctan2(bump_mask_pos[1], bump_mask_pos[0]) - np.rad2deg(38.7747) + np.pi/2 # Adjusted bump angle to better center it on spider
    bump_mask_pos = [radius * np.cos(theta), radius * np.sin(theta)]



    pupil_diameter = 6.5 # meter
    spider_width1 = 0.1917e-3 * magnification_factor # meter 
    spider_width2 = 0.1917e-3  * magnification_factor # meter
    central_obscuration_ratio = mask_inner / mask_outer 
    spider_offset = [0, 0.34]  # meter

    if normalized:
        spider_width1 /= pupil_diameter
        spider_width2 /= pupil_diameter
        bump_mask_pos = [x / pupil_diameter for x in bump_mask_pos]
        bump_mask_diameter = (0.5742e-3 * magnification_factor) / pupil_diameter


        spider_offset = [x / pupil_diameter for x in spider_offset]
        pupil_diameter = 1.0

    spider_offset = np.array(spider_offset)

    mirror_edge1 = (pupil_diameter / (2 * np.sqrt(2)), pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge2 = (-pupil_diameter / (2 * np.sqrt(2)), pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge3 = (pupil_diameter / (2 * np.sqrt(2)), -pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge4 = (-pupil_diameter / (2 * np.sqrt(2)), -pupil_diameter / (2 * np.sqrt(2)))

    obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)
    bump_mask = make_circular_aperture(bump_mask_diameter, center=bump_mask_pos) # Generate bump cover for Magellan pupil
    
    if not with_spiders:
        return obstructed_aperture

    spider1 = make_spider(spider_offset, mirror_edge1, spider_width1)
    spider2 = make_spider(spider_offset, mirror_edge2, spider_width1)
    spider3 = make_spider(-spider_offset, mirror_edge3, spider_width2)
    spider4 = make_spider(-spider_offset, mirror_edge4, spider_width2)

    def func(grid):
        return obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * (1 - bump_mask(grid))
    return func


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

def make_scc_sensor():
    """Generate an SCC difference image
    """
    def func(wavefront):
        dm.flatten()

        difference_images = []

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
    pwp_measurement = np.concatenate(difference_dark_hole_pixels)   # Combine into a list
    return pwp_measurement