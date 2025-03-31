import numpy as np
from surfplot_helper.plotting import plot_surface_data_fsLR
from time import time
############## Example one - plot fsLR_32k surface ############## 

# generate a random den-91k scalar data 
# you can read if from the .dscalar.nii file
data = np.random.uniform(low=-3, high=3, size=91282)

file_output = 'fsLR_32_surf_example.png'
cmap = 'jet'  # matplotlib compatible colormap
threshold = 0.3 # not display the value with absvalue < 0.3
figure_title = 'A random generated data'

t1 = time()
plot_surface_data_fsLR(data, file_output, surf_type='inflated',threshold = threshold, cmap=cmap, title=figure_title, orientation='landscape')
t2 = time()
duration = t2-t1
print(f'original data caused {duration:.2f} seconds ')

t1 = time()
plot_surface_data_fsLR(data, file_output, surf_type='inflated',threshold = threshold, cmap=cmap, title=figure_title, orientation='square')
t2 = time()
duration = t2-t1
print(f'threaded data caused {duration:.2f} seconds ')

############## Example two - plot kong400 parcellation ############## 

# from surfplot_helper.plotting import plot_Kong_parcellation
#
# data = np.random.uniform(low=-3, high=3, size=400)
# cmap = 'nipy_spectral'
# file_output = 'Kong400_random.png'
# plot_Kong_parcellation(data, file_output, surf_type='inflated', cmap= cmap, title='Kong400_random', title_position=200)