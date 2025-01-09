import os
import os.path as op
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def add_title_png(img_file,title, fontname = 'Arial', fontsize=80, title_position=300):
    """
    Add the title of the image: the contrast

    Parameters
    ----------
    img_file : str absolute path to a PNG image file_hctsa
    title: the title being shown on the figure
    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """
    img = Image.open(img_file)
    default_font = font_manager.findfont(fontname)
    titlefont = ImageFont.truetype(default_font, size=fontsize)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((title_position, 15), title, (0,0,0), font=titlefont)
    img.save(img_file, "PNG", dpi=(300.0, 300.0))

# add color bar11
def add_colorbar_png(data, img_file, cmap, vrange = None , orientation='horizontal'):
    """
    add a color bar for the image_file
    :param img_file:
    :param vmin: the minimum value for the color bar
    :param vmax: the maximum value for the color bar
    :return: None
    This function overwrites the existing file_hctsa.
    """
    import uuid
    
    # open the original figure
    img = Image.open(img_file) 
    img_x,img_y = img.size
    
    # draw a colorbar and open it
    colorbar_id = uuid.uuid1().hex
    fig_dir = os.path.dirname(img_file)
    file_colorbar = op.join(fig_dir, 'colorbar_%s.png'%(colorbar_id))
    draw_colorbar(data, figsize=(6, 0.7), file_colorbar=file_colorbar, cmap=cmap, vrange=vrange, orientation=orientation)
    
    img_colorbar = Image.open(file_colorbar)
    img_colorbar_x, img_colorbar_y = img_colorbar.size

    # add the color bar to the original figure to the centers
    img.paste(img_colorbar,(np.int16(img_x/2-img_colorbar_x/2), np.int16(img_y/2-img_colorbar_y/2)) )

    # save it
    img.save(img_file, dpi=(300.0, 300.0))

    os.system('rm -v %s'%(file_colorbar))

def draw_colorbar(data, figsize, file_colorbar, cmap, fontsize=20, vrange=None, orientation='horizontal'):
    

    if vrange is None:
        vrange=(np.min(data), np.max(data)) 
    else:
        check_vrange(vrange)
    
    vmin=np.round(vrange[0],1)
    vmax=np.round(vrange[1],1)
    if vmin>=0:
        
        vmid=(vmin+vmax)/2
        if vmax <=6:
            vmid=np.round(vmid,2)
        else:
            vmid=np.round(vmid,1)
    else:
        vmid=0


    ticks = [vmin, vmid, vmax]
    #draw a custom colorbar and save a temporary picture
    fig = plt.figure(constrained_layout=True,figsize=figsize)
    plt.rcParams.update({'font.sans-serif':'Arial'})
    ax = fig.subplots(1,1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    fcb = mpl.colorbar.ColorbarBase(norm=norm, cmap=mpl.cm.get_cmap(cmap), orientation=orientation,ax=ax ,ticks = ticks )
    fcb.ax.set_xticklabels(ticks, fontsize=fontsize)

    
    #save
    fig.savefig(file_colorbar, dpi=300,transparent=True)
    plt.close()

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
        _ = mpl.cm.get_cmap(cmap)
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