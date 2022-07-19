# iBen-atlas
> A cholinergic and catecholaminergic 3D Atlas of the developing mouse brain

<img src="https://github.com/iBen-foundation/iBen-atlas/blob/main/src/iBen-atlas.png" width="50%" height="50%">

[Link to the atlas files](http://www.babiomedical.com/ressources/)

## Reproducibility:

iBen atlas has been produced by following a multiresolution and iterative registration process of autofluorescence images. Interested users can reproduce it on their own images by following the below instruction. For more details, see the paper published in NeuroImage:


1 - Clone the iBen-atlas repository in a directory, e.g. `~/registration_test`, on your local machine.

2 - Set current working directory of your python to `~/registration_test/iBen-atlas/src/python`.

3a – Create a directory in the path `~/registration_test/iBen-atlas/data`, call it `AF` and put your autofluorescence stack tif images in it.

3b - Create a directory in the path `~/registration_test/iBen-atlas/data`, call it what you want, e.g. TH, and put your stained stack tif images in it.
Note: Stained image file and its corresponding AF image file must have the same name.

4 – Choose one of your images as fixed image and rename it to `fixed_image.tif` in the AF and stained image directories.

5 – Run `global_registration.py`.

6 – Do post-processing steps given at the end of `global_registration.py` file.

7 – Run `bspline_registration.py` for planned resolutions.
This is a multiresolution and iterative process. So, give appropriate values to the following variables before running the code:

   - `res` : resolution of the images on which the bspline registration is run.

   - `iter_index` : iteration index of the process.

   - `previous_registration_step` : if the previous registration step was global registration, put ‘global’; else put ‘bspline’.

   - `previos_pixel_resolution` : if the previous registration step was global registration, put ‘Native’; else put the previous image resolution for which the bspline registration was done.

8 - Do post-processing steps given at the end of `bspline_registration.py` file.

9 - Run `stained_images_registration.py`.

10 - Run `collect_created_average_templates.py` to collect created autofluorescence and stained templates in 1 directory `~/registration_test/iBen-atlas/output/created_average_templates`.
