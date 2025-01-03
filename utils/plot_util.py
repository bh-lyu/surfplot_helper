
from PIL import Image
from matplotlib import cm

def check_cmap_plt(cmap):
    """
    Check that a colormap exists in matplotlib.

    Parameters
    ----------
    cmap : str or None
        a valid matplotlib colormap; if None, return default colormap
        defined in wbplot.config

    Returns
    -------
    cmap : str

    Raises
    ------
    ValueError : colormap is not available matplotlib

    """
    try:
        _ = cm.get_cmap(cmap)
    except ValueError as e:
        raise ValueError(e)
    return cmap


def check_vrange(vrange):
    """
    Check vrange argument (used by other functions).


    Parameters
    ----------
    vrange : tuple or iterable
        data (min, max) for plotting; if iterable, must have length 2

    Returns
    -------
    vrange : tuple

    Raises
    ------
    ValueError : vrange is not length-2 iterable obj with vrange[0] > vrange[1]

    """
    if type(vrange) is not tuple:
        if not hasattr(vrange, "__iter__"):
            raise ValueError(
                'if vrange is not a tuple, it must be an iterable object')
        if len(vrange) != 2:
            raise ValueError("vrange must contain only two elements")
    if vrange[0] >= vrange[1]:
        raise ValueError("vrange[0] must be strictly less than vrange[1]")
    return tuple(list(vrange))