
"""
Apply the transformations, computed through the registration of AF images,
to corresponding stained images.

Important note: Stained image file and corresponding AF image file must have 
                similar names.
"""


import os
import tifffile

from registration_utils import threshold_image
from registration_utils import run_transformix
from registration_utils import signal_attenuation
from registration_utils import get_io_directories
from registration_utils import compute_average_stained_image
from registration_utils import create_output_directories_stained_image_registration


###############################################################################


""" Parameter initialization by user """

## Final image resolution of bspline registration of AF channel
res = 6

## Final iteration number of bspline registration of AF channel
final_iter_index = 1

## Name of directory where stained images are found e.g. 'ChAT', or 'TH'
staining = 'TH'


###############################################################################


""" Path setup : Find path organization in path_structure.txt """

## Collect input/output directories
data_dir, output_dir, global_registration_dir, res_output_dir = get_io_directories(res)

## Collect bspline transformation file directory
final_bspline_registration_dir = os.path.join(res_output_dir,
                                              f"iter_{final_iter_index}")

## Collect input image file paths
stained_images_dir = os.path.join(data_dir, staining)
stained_images_fname_list = os.listdir(final_bspline_registration_dir)
stained_images_fname_list = [image_fname for image_fname in stained_images_fname_list
                             if image_fname!="collective_outputs"]

## Create output directories
subject_output_directories, average_output_dir = \
   create_output_directories_stained_image_registration(output_dir, 
                                                        stained_images_fname_list,
                                                        staining)


###############################################################################


""" Apply the transformations """

for image_fname in stained_images_fname_list:

    print("\n--- START transformation of " + image_fname + " ---\n")
    
    input_image_path = os.path.join(stained_images_dir, image_fname+'.tif')
    
    subject_output_directory = [subject_output_dir for subject_output_dir in
                                subject_output_directories if image_fname 
                                in subject_output_dir][0]
    
    ## file paths of Transformation parameters
    similarity_transformParameters_file_path = os.path.join(
                                                   global_registration_dir,
                                                   image_fname, "similarity",
                                                   "TransformParameters.0.txt")
    
    affine_transformParameters_file_path = os.path.join(
                                                   global_registration_dir,
                                                   image_fname, "affine",
                                                   "TransformParameters.0.txt")
    
    bspline_transformParameters_file_path = os.path.join(
                                                final_bspline_registration_dir,
                                                image_fname,
                                                "TransformParameters.0.txt")
    ## Apply transformations
    run_transformix(input_image_path, 
                    subject_output_directory,
                    similarity_transformParameters_file_path)
    
    input_image_path = os.path.join(subject_output_directory, "result.mhd")
    
    run_transformix(input_image_path, 
                    subject_output_directory,
                    affine_transformParameters_file_path)
    
    input_image_path = os.path.join(subject_output_directory, "result.tif")
    
    ## Remove artifacts by intensity attenuation
    signal_attenuation(input_image_path, 
                       attenuation_threshold=60000,
                       attenuation_factor=10000)
    
    run_transformix(input_image_path, 
                    subject_output_directory,
                    bspline_transformParameters_file_path)
    
    ## Remove artifacts by thresholding
    threshold_image(input_image_path, threshold=60000)
       
    print("\n--- END of transformation of " + image_fname + " ---")
    print("-"*50)
    

###############################################################################


""" Compute the average image and save it """

average_image = compute_average_stained_image(subject_output_directories)

average_image_output_path = os.path.join(average_output_dir, 'result.tif')
with tifffile.TiffWriter(average_image_output_path) as tif:
    for frame in average_image:
        tif.save(frame, contiguous=True)
        
