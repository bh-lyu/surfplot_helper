# surfplot_helper

A simple function to plot the surface data based on fsLR_32k template for both hemispheres, medial and lateral view.
This is basically the implementation from `nilearn.plotting.surf_plotting.plot_surf_stat_map` but used fsLR_32k template from [HCP-pipeline](https://github.com/Washington-University/HCPpipelines). `nilearn` only supports plotting one view at once.


The plot function is based on `nilearn` and `matplotlib` in python packages.



Below is an example. Also check `plot_example.py` for how to use it. 

``` python
import numpy as np
from plotting import plot_surface_data_fsLR


# generate a random den-91k scalar data 
# you can read if from the .dscalar.nii file
data = np.random.uniform(low=-3, high=3, size=91282)

file_output = 'fsLR_32_surf_example.png' # could also be other format
cmap = 'jet'  # matplotlib compatible colormap
threshold = 0.3 # not display the value with absvalue < 0.3
figure_title = 'A random generated data'
fig, ax = plot_surface_data_fsLR(data, file_output, threshold = threshold, cmap=cmap, title=figure_title)

# if in png format, you could crop it 
```
from plotting import crop image
crop_image(file_output)
```

![random generaged ](fsLR_32_surf_example.png)


you can also fix the range by specifing `vrange`

```python
vrange = (-3, 3)
fig, ax = plot_surface_data_fsLR(data, file_output, threshold = threshold, vrange=vrange, cmap=cmap, title=figure_title)
```