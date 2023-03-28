
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import nibabel as nib
from nilearn.plotting.surf_plotting import plot_surf_stat_map
from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges
from nilearn.plotting.surf_plotting import _get_ticks_matplotlib, _get_cmap_matplotlib
import os
import numpy as np
from constants import *
from constants import _fetch_fsLR_32k_surf
from PIL import Image
from PyPDF2 import PdfWriter, PdfReader


def _crop_image_pil(image_path, top_margin_r=0.25, bottom_margin_r=0.25):
    """
    crop the white space margin at the top and bottom of the picture
    """

    # Open the image
    im = Image.open(image_path)

    # Get the image size
    width, height = im.size

    # Crop the image
    top_margin = int(top_margin_r * height)
    bottom_margin = int(bottom_margin_r * height)
    im_cropped = im.crop((0, top_margin, width, height - bottom_margin))

    # Save the cropped image
    im_cropped.save(image_path)

    # Open the PDF file

def _crop_image_pdf(image_path, top_margin_r=0.25, bottom_margin_r=0.25):
    
    reader = PdfReader(image_path)
    writer = PdfWriter()

    page = reader.pages[0]
    height = page.mediabox.height
    page.mediabox.bottom = page.mediabox.bottom + bottom_margin_r * height
    page.mediabox.top = page.mediabox.top * (1-top_margin_r)
    
    writer.add_page(page)
    with open("cropped.pdf", "wb") as fp:
        writer.write(fp)


    """
    with open(image_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        # Get the first page of the PDF
        page = pdf.getPage(0)
        # Get the first image on the page (assuming there is only one)
        xObject = page['/Resources']['/XObject'].getObject()
        image = None
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                image = xObject[obj]
                break
        if image:
            # Get the image dimensions
            width = image['/Width']
            height = image['/Height']
            top_margin = int(top_margin_r * height)
            bottom_margin = int(bottom_margin_r * height)
            # Set the cropping rectangle (left, bottom, right, top)
            crop = [0, bottom_margin, width, height - top_margin]
            # Apply the crop to the image
            image.update({
                '/CropBox': crop,
                '/PZ': height
            })
        # Write the modified PDF to a new file
        with open('cropped.pdf', 'wb') as outfile:
            writer = PyPDF2.PdfWriter()
            writer.addPage(page)
            writer.write(outfile)
    
    # Rename the new file to the original filename (be careful not to overwrite the original file)
    if os.path.isfile('cropped.pdf'):
        os.rename('cropped.pdf', image_path)
    """

def crop_image(image_path, top_margin_r=0.25, bottom_margin_r=0.25):
    if (image_path.endswith('.png') or image_path.endswith('.jpg') or image_path.endswith('.jpeg')):
        _crop_image_pil(image_path, top_margin_r, bottom_margin_r)
    elif image_path.endswith('.pdf'):
        _crop_image_pdf(image_path, top_margin_r, bottom_margin_r)
    else:
        raise ValueError('only support png, jpg, or pdf format')


def extra_surface_data_fsLR(cifti, length = 'full'):
    """
    get surface data from dlabel.nii or dscalar.nii
    the data within the cifti must have length of 91282
    vertices_num_L = 29696
    vertices_num_R = 29716
    vertices_num_32k = 32492
    :param vertices_32: int; shape: 32492
    :param hemi: hemisphere: L or R


    :return: 
        if length == 'part', return (data_lh, data_rh) with shape of (29696,) and (29716,)
        if length == 'full' (default), return (data_lh, data_rh) with both shape (32492,) by filling zeros 
    """

    template = nib.load(Greyordinates_91282)
    if isinstance(cifti, np.ndarray):
        data = cifti
    elif isinstance(cifti, str):
        if  (cifti.endswith('.dlabel.nii') or cifti.endswith('dscalar.nii')):
            cii = nib.load(cifti)
            data = cii.get_fdata()
    elif isinstance(cifti, nib.Cifti2Image):
        cii = cifti
        data = cii.get_fdata()

    data = data.ravel()
    if data.shape[0] != 91282:
        raise ValueError('the fsLR cifti data must be in 91k .dlabel.nii or ndarray with (91282,) shape')    
    
    for brain_model in template.header.get_index_map(1).brain_models:

        if brain_model.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            cortex_lh = brain_model

        if brain_model.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            cortex_rh = brain_model
         
    # get the data for each hemisphere, without zeros
   
    index_lh = range(cortex_lh.index_offset, cortex_lh.index_offset+cortex_lh.index_count)
    index_rh = range(cortex_rh.index_offset, cortex_rh.index_offset+cortex_rh.index_count)
    data_lh_pure, data_rh_pure = data[index_lh], data[index_rh] 

    # get the data onto the 32k space
    if length == 'full':
        data_lh = np.zeros(32492)
        data_rh = np.zeros(32492)
        vertex_indices_lh = list(cortex_lh.vertex_indices)
        vertex_indices_rh = list(cortex_rh.vertex_indices)
        data_lh[vertex_indices_lh] = data_lh_pure
        data_rh[vertex_indices_rh] = data_rh_pure
    elif length== 'part':
        data_lh, data_rh = data_lh_pure, data_rh_pure
    
    return (data_lh, data_rh)


def plot_surface_data_fsLR(data, file_output=None, surf_type = 'inflated', title=None, colorbar=True, vrange=None, 
                        cmap = 'jet',symmetric_cbar="auto", cbar_tick_format='%.2g', threshold=None):
    
    """
    nilearn only supports plot one view at a time without consistent colobar
    This is a helpful function to plot [lh-lateral, lh-medial, rh-lateral, rh-medial] onto a single plot
    The colorbar is consitent across subplots

    This only support data in the following format
    :data: (1) .dlabel.nii cifti data with 91k density
           (2) tuple containing (data_lh, data_rh) with 
            data_lh.shape = (29696, ) data_rh.shape=(29716,)
           (3) np.ndarrya with shape (91282)
    
    
    """

    if isinstance(data, tuple):
        data_lh, data_rh = data
    elif isinstance(data, str) or isinstance(data, np.ndarray):
        data_lh, data_rh = extra_surface_data_fsLR(data)

    fsLR_32k_surf_L, fsLR_32k_surf_R = _fetch_fsLR_32k_surf(surf_type)

    # set up the default figure
    mpl.rcParams['font.family']='Arial' #optional
    
    _default_figsize = (12,5)
    fig, axes = plt.subplots(1, 4, figsize=_default_figsize, subplot_kw={'projection': '3d'}) 
    plt.tight_layout()

    # Plot the left hemisphere
    plot_surf_stat_map(fsLR_32k_surf_L, data_lh, hemi='left', view='lateral', bg_map=fsLR_32k_sulc_L ,colorbar=False, threshold=threshold, cmap=cmap, axes=axes[0], figure=fig)
    plot_surf_stat_map(fsLR_32k_surf_L, data_lh, hemi='left', view='medial', bg_map=fsLR_32k_sulc_L ,colorbar=False, threshold=threshold, cmap=cmap, axes=axes[1],figure=fig)

    # Plot the right hemisphere
    plot_surf_stat_map(fsLR_32k_surf_R, data_rh, hemi='right', view='lateral', bg_map=fsLR_32k_sulc_R, colorbar=False, threshold=threshold,   cmap=cmap, axes=axes[2],figure=fig)
    plot_surf_stat_map(fsLR_32k_surf_R, data_rh, hemi='right', view='medial', bg_map=fsLR_32k_sulc_R, colorbar=False, threshold=threshold,   cmap=cmap, axes=axes[3],figure=fig)

    # reduce viewing distance to remove space around mesh
    for i in range(4): axes[i].set_box_aspect(None, zoom=1.2)

    # Create a shared colorbar
    if colorbar:
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        if vrange is None: # get the adpated range if not provided with vmax
            cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(np.concatenate((data_lh,data_rh)), vmax=None, symmetric_cbar=symmetric_cbar, kwargs={})
            cbar_vmin = cbar_vmin if cbar_vmin is not None else vmin
            cbar_vmax = cbar_vmax if cbar_vmax is not None else vmax
        else:
            cbar_vmin, cbar_vmax = vrange
        
        ticks = _get_ticks_matplotlib(cbar_vmin, cbar_vmax, cbar_tick_format)
        our_cmap, norm = _get_cmap_matplotlib(cmap, cbar_vmin, cbar_vmax, threshold)
        bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)

        # we need to create a proxy mappable
        proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
        proxy_mappable.set_array(np.concatenate((data_lh,data_rh)))
        cax = fig.add_axes([0.94, 0.3, 0.02, 0.4]) # colorbar on the right
        #cax, _ = make_axes(axes[3], location='right', fraction=.15, shrink=.5, pad=.0, aspect=10.)
        fig.colorbar(proxy_mappable, cax=cax, ticks=ticks,boundaries=bounds, spacing='proportional',format=cbar_tick_format, orientation='vertical', shrink= .3)

    # adjust the space 
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.9, top=0.99, wspace=0.03, hspace=0.03)
    
    if title is not None:
        fig.suptitle(title, ha='left', fontsize=12, x=0.01, y=0.75)

    if file_output is None:
        return fig, axes
    else:
        plt.savefig(file_output)
        print(f'figure saved at {file_output}')


