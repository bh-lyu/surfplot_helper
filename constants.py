import os.path as op


root = op.dirname(op.abspath(__file__))

TEMPLATE_DIR = 'templates'
Greyordinates_91282 = op.join(root, TEMPLATE_DIR, '91282_Greyordinates.dscalar.nii')

fsLR_32k_sulc_L =  op.join(root, TEMPLATE_DIR, 'fs_LR.32k.L.sulc.shape.gii' )
fsLR_32k_sulc_R = op.join(root, TEMPLATE_DIR, 'fs_LR.32k.R.sulc.shape.gii' )

surf_types = ['midthickness', 'inflated', 'very_inflated']
fsLR_32k_surf = {'midthickness': ( 'fs_LR.32k.L.midthickness.surf.gii', 'fs_LR.32k.R.midthickness.surf.gii'),
                 'inflated':('fs_LR.32k.L.inflated.surf.gii','fs_LR.32k.R.inflated.surf.gii'),
                'very_inflated': ('fs_LR.32k.L.very_inflated.surf.gii',  'fs_LR.32k.R.very_inflated.surf.gii') }



def _fetch_fsLR_32k_surf(surf_type = 'inflated'):
    assert surf_type in surf_types

    surf_L_file, surf_R_file = fsLR_32k_surf[surf_type]

    surf_L = op.join(root, TEMPLATE_DIR, surf_L_file)
    surf_R = op.join(root, TEMPLATE_DIR, surf_R_file)

    return (surf_L, surf_R)


if __name__ == '__main__':
    fsLR_32k_surf_L, fsLR_32k_surf_R = _fetch_fsLR_32k_surf(surf_type='very_inflated')

    print (fsLR_32k_surf_L + '\n' + fsLR_32k_surf_R)