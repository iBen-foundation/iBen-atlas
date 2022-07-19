
"""
Global registeration of all subject images to a fixed image.

Global registeration = similarity + affine transformations
"""


import os
import tifffile

from registration_utils import get_io_directories
from registration_utils import affine_registration
from registration_utils import compute_average_image
from registration_utils import similarity_registration
from registration_utils import create_output_directories_global_registration


###############################################################################


""" 
Path setup : Find path organization in path_structure.txt
"""

## Collect input/output directories
data_dir, output_dir, _, __ = get_io_directories(None)

## Collect input fixed and moving image file paths
moving_images_dir = os.path.join(data_dir, "AF")
images_fname_list = os.listdir(moving_images_dir)
fixed_image_path = os.path.join(data_dir, "AF", "fixed_image.tif")

## Collect Elastix parameters file paths
elastix_similarity_parameters_file_path = os.path.join(data_dir, 
                                                       "elastix_parameter_files",
                                                       "similarity.txt")
elastix_affine_parameters_file_path = os.path.join(data_dir, 
                                                   "elastix_parameter_files",
                                                   "affine.txt")
## Create output directories
subject_output_directories, template_output_dir = \
              create_output_directories_global_registration(output_dir,
                                                            images_fname_list)


###############################################################################


""" Do the registration. Save the registered image for visual inspection. """

for image_fname in images_fname_list:
    print("\n--- START registration of " + image_fname + " ---\n")
    
    moving_image_path = os.path.join(moving_images_dir, image_fname)
    
    image_name, image_ext = os.path.splitext(image_fname)
    subject_output_directory = [subject_output_dir for subject_output_dir in
                                subject_output_directories if image_name in
                                subject_output_dir][0]
    
    similarity_registration(fixed_image_path, 
                            moving_image_path, 
                            elastix_similarity_parameters_file_path,
                            subject_output_directory)
    
    affine_registration(fixed_image_path, 
                        elastix_affine_parameters_file_path,
                        subject_output_directory)
    
    print("\n--- END of registration of " + image_fname + " ---")


###############################################################################


""" Compute the average template T0 and save it """

subject_affine_output_directories = []
for subject_output_directory in subject_output_directories:
    subject_affine_output_directory = os.path.join(subject_output_directory, 
                                                   'affine')
    subject_affine_output_directories.append(subject_affine_output_directory)

T0 = compute_average_image(subject_affine_output_directories)

## save the average image
template_output_path = os.path.join(template_output_dir, 'T0.tif')
with tifffile.TiffWriter(template_output_path) as tif:
    for frame in T0:
        tif.save(frame, contiguous=True)
    

###############################################################################


"""
Postprocessing steps in Fiji:

A) Image downsampling in Fiji (Image -> Scale...):
    
   Given pixel resolution values in your multiresolution bspline 
   resigtration plan, e.g. v1, v2 and v3 um where v1>v2>v3=native resolution:
            
     1) Downsample each registered image (affine/result.0.tif) to v1 and v2 um
        pixel resolution and save it as result.0-res{v1}um.tif,
        result.0-res{v2}um.tif in the same directory.

     2) Rename result.0.tif to result.0-res{v3}um.tif.
             
     3) Downsample the created template (template/T0.tif) to v1 um pixel 
        resolution and save it as T0-res{v1}um.tif in the same directory.


B) Change image pixel unit from inch to pixel in Fiji (Image > Properties...):
    
    1) Change the pixel unit of each registered image 
       (affine/result.0-res{v1}um.tif and affine/result.0-res{v2}um.tif)
       and the created template (template/T0-res{v1}um.tif) from inch to pixel.
       put 1 for the Pixel width, Pixel height and Voxel depth.	

"""


