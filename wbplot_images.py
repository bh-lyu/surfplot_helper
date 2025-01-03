"""Auxiliary functions pertaining to the manipulation of neuroimaging files. """

import numpy as np
import nibabel as nib
from os import system, remove
from matplotlib import colors as clrs
from matplotlib import cm
import xml.etree.cElementTree as eT
from nibabel.cifti2.parse_cifti2 import Cifti2Parser
from utils.plot_util import check_vrange, check_cmap_plt
from utils.cifti_util import TEMPLATE_dlabel

def map_unilateral_to_bilateral(pscalars, hemisphere):
    """
    Map 180 unilateral pscalars to 360 bilateral pscalars, padding contralateral
    hemisphere with zeros.

    Parameters
    ----------
    pscalars : numpy.ndarray
        unilateral parcellated scalars
    hemisphere : 'left' or 'right' or None

    Returns
    -------
    numpy.ndarray

    """
    hemisphere = check_parcel_hemi(pscalars=pscalars, hemisphere=hemisphere)
    if hemisphere is None:
        return pscalars
    pscalars_lr = np.zeros(360)
    if hemisphere == 'right':
        pscalars_lr[:180] = pscalars
    elif hemisphere == 'left':
        pscalars_lr[180:] = pscalars
    return pscalars_lr


def check_pscalars_unilateral(pscalars):
    """
    Check that unilateral pscalars have the expected size and shape.

    Parameters
    ----------
    pscalars : numpy.ndarray
        parcellated scalars

    Returns
    -------
    None

    Raises
    ------
    TypeError : pscalars is not array_like
    ValueError : pscalars is not one-dimensional and length 180

    """
    if not isinstance(pscalars, np.ndarray):
        raise TypeError(
            "pscalars: expected array_like, got {}".format(type(pscalars)))
    if pscalars.ndim != 1 or pscalars.size != 180:
        e = "pscalars must be one-dimensional and length 180"
        e += "\npscalars.shape: {}".format(pscalars.shape)
        raise ValueError(e)


def check_pscalars_bilateral(pscalars):
    """
    Check that bilateral pscalars have the expected size and shape.

    Parameters
    ----------
    pscalars : numpy.ndarray
        parcellated scalars

    Raises
    ------
    TypeError : pscalars is not array_like
    ValueError : pscalars is not one-dimensional and length 360

    """
    if not isinstance(pscalars, np.ndarray):
        raise TypeError(
            "pscalars: expected array_like, got {}".format(type(pscalars)))
    if pscalars.ndim != 1 or pscalars.size != 360:
        e = "pscalars must be one-dimensional and length 180"
        e += "\npscalars.shape: {}".format(pscalars.shape)
        raise ValueError(e)


def check_dscalars(dscalars):
    """
    Check that dscalars have the expected size and shape.


    Parameters
    ----------
    dscalars : numpy.ndarray
        dense scalars

    Returns
    -------
    None

    Raises
    ------
    TypeError : pscalars is not array_like
    ValueError : pscalars is not one-dimensional and length 59412

    """
    if not isinstance(dscalars, np.ndarray):
        raise TypeError(
            "dscalars: expected array_like, got {}".format(type(dscalars)))
    if dscalars.ndim != 1 or dscalars.size != 59412:
        e = "dscalars must be one-dimensional and length 59412"
        e += "\ndscalars.shape: {}".format(dscalars.shape)
        raise ValueError(e)


def check_parcel_hemi(pscalars, hemisphere):
    """
    Check hemisphere argument for package compatibility.

    Parameters
    ----------
    pscalars : numpy.ndarray
        parcels' scalar quantities
    hemisphere : 'left' or 'right' or None
        if bilateral, use None

    Returns
    -------
    'left' or 'right' or None

    Raises
    ------
    RuntimeError : pscalars is not length-360 but hemisphere not indicated
    ValueError : invalid hemisphere argument

    """
    if pscalars.size != 360 and hemisphere is None:
        raise RuntimeError(
            "you must indicate which hemisphere these pscalars correspond to")
    options = ['left', 'l', 'L', 'right', 'r', 'R', None, 'lr', 'LR']
    if hemisphere not in options:
        raise ValueError("{} is not a valid hemisphere".format(hemisphere))
    if hemisphere in ['left', 'l', 'L']:
        return 'left'
    if hemisphere in ['right', 'r', 'R']:
        return 'right'
    if hemisphere in ['None', 'lr', 'LR']:
        return None


def check_dense_hemi(hemisphere):
    """
    Check hemisphere argument for compatibility.

    Parameters
    ----------
    hemisphere : 'left' or 'right' or None
        if bilateral, use None

    Returns
    -------
    'left' or 'right' or None

    Raises
    ------
    ValueError : invalid hemisphere argument

    """
    options = ['left', 'l', 'L', 'right', 'r', 'R', None, 'lr', 'LR']
    if hemisphere not in options:
        raise ValueError("{} is not a valid hemisphere".format(hemisphere))
    if hemisphere in ['left', 'l', 'L']:
        return 'left'
    if hemisphere in ['right', 'r', 'R']:
        return 'right'
    if hemisphere in ['None', 'lr', 'LR']:
        return None


def extract_nifti_data(of):
    """Extract array of scalar quantities from a NIFTI2 image.

    Parameters
    ----------
    of : :class:~`nibabel.Nifti2Image` instance
        the NIFTI2 image from which to extract scalar data

    Returns
    -------
    data : numpy.ndarray

    """
    return np.asanyarray(of.dataobj).squeeze()


def extract_gifti_data(of):
    """Extract array of scalar quantities from a GIFTI image.

    Parameters
    ----------
    of : :class:~`nibabel.gifti.GiftiImage` instance
        the GIFTI image from which to extract scalar data

    Returns
    -------
    data : numpy.ndarray

    """
    return np.asanyarray(of.darrays[0].data).squeeze()


def write_parcellated_image(
        data, fout, hemisphere=None, cmap='magma', vrange=None):
    """
    Change the colors for parcels in a dlabel file to illustrate pscalar data.

    Parameters
    ----------
    data : numpy.ndarray
        scalar map values
    fout : str
        absolute path to output neuroimaging file with *.dlabel.nii* extension
        (if an extension is provided)
    hemisphere : 'left' or 'right' or None, default None
        which hemisphere `pscalars` correspond to. for bilateral data use None
    cmap : str
        a valid MATPLOTLIB colormap used to plot the data
    vrange : tuple
        data (min, max) for plotting; if None, use (min(data), max(data))

    Returns
    -------
    None

    Notes
    -----
    The file defined by wbplot.config.PARCELLATION_FILE is used as a template to
    achieve this. Thus the data provided to this function must be in the same
    parcellation as that file. By default, this is the HCP MMP1.0 parcellation;
    thus, `data` must be ordered as (R_1, R_2, ..., R_180, L_1, L_2, ..., L_180)
    if bilateral. If unilateral, they must be ordered from area V1 (parcel 1) to
    area p24 (parcel 180).
    """

    # Check provided inputs and pad contralateral hemisphere with 0 if necessary
    #check_parcel_hemi(pscalars=data, hemisphere=hemisphere)
    cmap = check_cmap_plt(cmap)

    # Change the colors assigned to each parcel and save to `fout`
    c = Cifti()
    data[np.where(np.isnan(data))]=0
    c.set_cmap(data=data, cmap=cmap, vrange=vrange)
    c.save(fout)


class Cifti(object):
    """
    A class for changing the colors inside the metadata of a DLABEL neuroimaging
    file. Some of this code was contributed by Dr. Murat Demirtas while he was
    a post-doctoral researcher at Yale.
    """

    def __init__(self):
        of = nib.load(TEMPLATE_dlabel)  # must be a DLABEL file!!
        self.data = np.asanyarray(of.dataobj)
        self.header = of.header
        self.nifti_header = of.nifti_header
        # self.extensions = eT.fromstring(  BROKEN AS OF NIBABEL 3.2
        #     self.nifti_header.extensions[0].get_content().to_xml())
        self.tree = eT.fromstring(self.header.to_xml())
        self.vrange = None
        self.ischanged = False

    def set_cmap(self, data, cmap='magma', vrange=None,  mappable=None):
        """
        Map scalar data to RGBA values and update file header metadata.

        Parameters
        ----------
        data : numpy.ndarray
            scalar data
        cmap : str or None, default 'magma'
            colormap to use for plotting
        vrange : tuple or None, default None
            data (min, max) for illustration; if None, use (min(data),max(data))
        mappable : Callable[float] or None, default None
            can be used to override arguments `cmap` and `vrange`, e.g. by
            specifying your own map from scalar input to RGBA output

        Returns
        -------
        None

        """
        #if data.size != 360:
        #    raise RuntimeError(
         #       "pscalars must be length 360 for :class:~wbplot.images.Cifti")

        # Check input arguments
        cmap = check_cmap_plt(cmap)
        self.vrange = (
            np.min(data), np.max(data)) if vrange is None else vrange
        self.vrange = check_vrange(self.vrange)

        # Map scalar data to colors (R, G, B, Alpha)
        if mappable is None:
            cnorm = clrs.Normalize(vmin=self.vrange[0], vmax=self.vrange[1])
            clr_map = cm.ScalarMappable(cmap=cmap, norm=cnorm)
            colors = clr_map.to_rgba(data)
        else:
            colors = np.array([mappable(d) for d in data])

        # Update file header metadata
        for ii in range(1, len(self.tree[0][1][0][0])):
            self.tree[0][1][0][0][ii].set(
                'Red', str(colors[ii - 1, 0]))
            self.tree[0][1][0][0][ii].set(
                'Green', str(colors[ii - 1, 1]))
            self.tree[0][1][0][0][ii].set(
                'Blue', str(colors[ii - 1, 2]))
            self.tree[0][1][0][0][ii].set(
                'Alpha', str(colors[ii - 1, 3]))
        self.ischanged = True

    def save(self, fout):
        """
        Write self.data to image `fout`.

        Parameters
        ----------
        fout : str
            absolute path to output neuroimaging file. must be a DLABEL file!!

        Returns
        -------
        None

        """
        suffix = ".dlabel.nii"
        #suffix=".ptseries.nii"
        if self.ischanged:
            cp = Cifti2Parser()
            cp.parse(string=eT.tostring(self.tree))
            header = cp.header
        else:
            header = self.header
        #if fout[-11:] != suffix:  # TODO: improve input handling
        #    fout += suffix
         #   print('The suffix is not %s, changing it...')%(suffix)
        new_img = nib.Cifti2Image(
            self.data, header=header, nifti_header=self.nifti_header)
        nib.save(new_img, fout)


# Pythonic version of this workbench command (primarily so I don't forget)
def cifti_parcellate(cifti_in, dlabel_in, cifti_out, direction='COLUMN'):
    cmd = "wb_command -cifti-parcellate {} {} {} {}".format(
        cifti_in, dlabel_in, direction, cifti_out)
    system(cmd)

