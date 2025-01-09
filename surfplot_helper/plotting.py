import os
from os import system
import os.path as op
import numpy as np
import itertools
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from surfplot_helper.wbplot_images import write_parcellated_image # from wb plot images
from zipfile import ZipFile
import nibabel as nib
from nilearn.plotting.surf_plotting import plot_surf_stat_map
from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges
from nilearn.plotting.surf_plotting import _get_ticks_matplotlib, _get_cmap_matplotlib
#custom functions
from .utils.cifti_util import *
from .utils.cifti_util import _fetch_fsLR_32k_surf
from .utils.plot_util import add_colorbar_png, add_title_png



def plot_surface_data_fsLR(data, file_output=None, surf_type='inflated', title=None, colorbar=True, vrange=None, 
                           threshold=None, cmap = 'jet', symmetric_cbar="auto", cbar_tick_format='%.2g', threshold_cbar=True,
                           fontname='Arial', orientation='square'):
    
    """
    nilearn only supports plot one view at a time without consistent colobar
    This is a helpful function to plot [lh-lateral, lh-medial, rh-lateral, rh-medial] onto a single plot
    The colorbar is consitent across subplots

    This only support data in the following format
    :data: (1) .dlabel.nii cifti data with 91k density
           (2) tuple containing (data_lh, data_rh) with 
            data_lh.shape = (29696, ) data_rh.shape=(29716,)
           (3) numpyarray, in shape (91282,) for fsLR_32k_surf
    
    """
    # set up the default figure
    mpl.rcParams['font.family']=fontname
    
    if isinstance(data, tuple):
        data_lh, data_rh = data
    elif isinstance(data, str) or isinstance(data, np.ndarray):
        data=np.nan_to_num(data)
        data_lh, data_rh = extra_surface_data_fsLR(data, length='full')
    fsLR_32k_surf_L, fsLR_32k_surf_R = _fetch_fsLR_32k_surf(surf_type=surf_type)
    
    if threshold is None:
        threshold = 0.01
    data_lh[np.abs(data_lh)<threshold]=0
    data_rh[np.abs(data_rh)<threshold]=0
    
    if orientation=='landscape':
        _default_figsize = (12, 2.5)
        fig, axes = plt.subplots(1, 4, figsize=_default_figsize, subplot_kw={'projection': '3d'}) 
        cbar_coord = [0.92, 0.25, 0.02, 0.55]
        cbar_orient = 'vertical'
        title_y=0.98
        
        # Plot the left and righ hemisphere
        plot_surf_stat_map(fsLR_32k_surf_L, data_lh, hemi='left', view='lateral', bg_map=fsLR_32k_sulc_L ,colorbar=False, threshold=threshold, symmetric_cbar=symmetric_cbar, cmap=cmap, axes=axes[0], figure=fig)
        plot_surf_stat_map(fsLR_32k_surf_L, data_lh, hemi='left', view='medial', bg_map=fsLR_32k_sulc_L ,colorbar=False, threshold=threshold, symmetric_cbar=symmetric_cbar, cmap=cmap, axes=axes[1],figure=fig)
        plot_surf_stat_map(fsLR_32k_surf_R, data_rh, hemi='right', view='lateral', bg_map=fsLR_32k_sulc_R, colorbar=False, threshold=threshold, symmetric_cbar=symmetric_cbar,  cmap=cmap, axes=axes[2],figure=fig)
        plot_surf_stat_map(fsLR_32k_surf_R, data_rh, hemi='right', view='medial', bg_map=fsLR_32k_sulc_R, colorbar=False, threshold=threshold, symmetric_cbar=symmetric_cbar,  cmap=cmap, axes=axes[3],figure=fig)
        
        # reduce viewing distance to remove space around mesh
        for i in range(4): axes[i].set_box_aspect(None, zoom=1.5)
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.9, top=0.99, wspace=0.05, hspace=0)
        
    elif orientation=='square':
        _default_figsize = (5,5)
        fig, axes = plt.subplots(2, 2, figsize=_default_figsize, subplot_kw={'projection': '3d'}) 
        cbar_coord = [0.3, 0.04, 0.4, 0.02]
        cbar_orient = 'horizontal'
        title_y = 0.98
        # Plot the left and righ hemisphere
        plot_surf_stat_map(fsLR_32k_surf_L, data_lh, hemi='left', view='lateral', bg_map=fsLR_32k_sulc_L,colorbar=False, threshold=threshold, symmetric_cbar=symmetric_cbar, cmap=cmap, axes=axes[0,0], figure=fig)
        plot_surf_stat_map(fsLR_32k_surf_L, data_lh, hemi='left', view='medial', bg_map=fsLR_32k_sulc_L, colorbar=False, threshold=threshold, symmetric_cbar=symmetric_cbar, cmap=cmap, axes=axes[0,1],figure=fig)
        plot_surf_stat_map(fsLR_32k_surf_R, data_rh, hemi='right', view='lateral', bg_map=fsLR_32k_sulc_R, colorbar=False, threshold=threshold, symmetric_cbar=symmetric_cbar,  cmap=cmap, axes=axes[1,0],figure=fig)
        plot_surf_stat_map(fsLR_32k_surf_R, data_rh, hemi='right', view='medial', bg_map=fsLR_32k_sulc_R, colorbar=False, threshold=threshold, symmetric_cbar=symmetric_cbar,  cmap=cmap, axes=axes[1,1],figure=fig)
        
        for i,j in itertools.product(range(2), repeat=2): 
            axes[i,j].set_box_aspect(None, zoom=1.5)
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0)
    else:
        raise ValueError('orientation must be landscape(1x4), or square(2x2)')


    # Create a shared colorbar
    if colorbar and np.any(data!=0):
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        if vrange is None: # get the adpated range if not provided with vmax
            cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(np.concatenate((data_lh,data_rh)), vmax=None, symmetric_cbar=symmetric_cbar)
            cbar_vmin = cbar_vmin if cbar_vmin is not None else vmin
            cbar_vmax = cbar_vmax if cbar_vmax is not None else vmax
        else:
            cbar_vmin, cbar_vmax = vrange
        
        if threshold_cbar:
            our_cmap, norm = _get_cmap_matplotlib(cmap, cbar_vmin, cbar_vmax, threshold)
        else:
            our_cmap, norm = _get_cmap_matplotlib(cmap, cbar_vmin, cbar_vmax)
        ticks = _get_ticks_matplotlib(cbar_vmin, cbar_vmax, cbar_tick_format)
        bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)

        # we need to create a proxy mappable
        proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
        proxy_mappable.set_array(np.concatenate((data_lh,data_rh)))
        
        cax = fig.add_axes(cbar_coord) # colorbar on the right
        #cax, _ = make_axes(axes[3], location='right', fraction=.15, shrink=.5, pad=.0, aspect=10.)
        cbar =fig.colorbar(proxy_mappable, cax=cax, ticks=ticks,boundaries=bounds, spacing='proportional',format=cbar_tick_format, orientation=cbar_orient)
        cbar.ax.tick_params(length=0)

    if title is not None:
        fig.suptitle(title, ha='center',fontsize=12, x=0.5, y=title_y)

    if file_output is None:
        return fig, axes
    else:
        fig.savefig(file_output)

        print(f'figure saved at {file_output}')
        

def plot_Kong_parcellation(data, figfile, surf_type='very_inflated', 
                           figsize=(2400,2400), cmap= 'nipy_spectral', colorbar=True, 
                           save_cifti=False, vrange = None, title=None, title_position=180):
    import uuid
    """
    :param data: is 400 values of the network, shape is (400,1)
    :param figfile: the output figure filename ending with '.png'
    :surf_type, str: [very_inflated, inflated], specify the surface type
    :return: save the png file to the specified figfile 
    """
    

    if type(data) is np.ndarray:
        pscalar = data
    elif isinstance(data, str):
        d = nib.load(data).get_fdata()
        pscalar = d[0]
    else:
        raise ValueError('data must be a data array or a cifti file ')
    if surf_type == 'very_inflated':
        scene = 1
    elif surf_type == 'inflated':
        scene = 2
    else:
        raise ValueError('scene 1 = very_inflated, 2 = inflated surface. no other scenes are supported yet ')
    
    # The original scene file you created 
    scene_zip_file=op.join(path_scene,'scene_Kong_dlabel.zip')
    filename_scene = 'Schaefer_417_dlabel.scene'
    temp_name = 'ImageParcellated.dlabel.nii'

    try:
        #generate temporary working folder in output figure directory
        fig_id = uuid.uuid1().hex
        fig_dir = os.path.dirname(figfile)
        temp_dir = op.join(fig_dir,'temp_scene_'+fig_id)
        os.mkdir(temp_dir)
        
        # copy the scene file & SchaeferParcellations directory to the
        # temp directory as well
        with ZipFile(scene_zip_file, "r") as z:  # unzip to temp dir
            z.extractall(temp_dir)
        scene_file = op.join(temp_dir, 'scene_Kong_dlabel', filename_scene)
        if not op.isfile(scene_file):
            raise RuntimeError(
                "scene file was not successfully copied to {}".format(scene_file))

        # Write `pscalars` to the neuroimaging file which is pre-loaded into the
        # scene file, and update the colors for each parcel using the file metadata
        temp_cifti1 = op.join(temp_dir, temp_name)
        temp_cifti2 = op.join(temp_dir,'scene_Kong_dlabel',temp_name)
        write_parcellated_image(data=pscalar, fout=temp_cifti1,  cmap=cmap, vrange = vrange)

        #overwrite the original files
        cmd='cp -f %s %s'%(temp_cifti1, temp_cifti2)
        system(cmd)
        
        if save_cifti:
            cifti_file = figfile[:-4]+'.dlabel.nii'
            cmd='cp -f %s %s'%(temp_cifti1, cifti_file)
            system(cmd)
        
        # copy scene file

            
        width, height = figsize   # height = 835
        cmd = path_wb + ' -show-scene "{}" {} "{}" {} {}'.format(scene_file, scene, figfile, width, height)
        system(cmd)
        
        # remove the copied scene folder
        cmd = 'rm -rf %s'%(temp_dir)
        system(cmd)

        # add title and colorbar on ImageFile
        if colorbar:
            add_colorbar_png(pscalar, figfile, cmap, vrange, orientation='horizontal')
        if not title is None:
            add_title_png(figfile, title, title_position=title_position)
    except Exception as e:
        # remove the copied scene folder
        cmd = 'rm -rf %s'%(temp_dir)
        system(cmd)
        print(f'An error occured : {e}')



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
    
    # check the data shape
    if data.shape[0] == 64984:
        return(data[:32492], data[32492:])
    if data.shape[0] == 59412:
        data = np.pad(data, (0, 91282-59412), constant_values=0)
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

