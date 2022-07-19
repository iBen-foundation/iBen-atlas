
"""
Multiresolution iterative deformable registeration process using 
Bspline transformation
"""


import os
import pickle
import tifffile

from registration_utils import threshold_image
from registration_utils import run_transformix
from registration_utils import create_MMADF_file
from registration_utils import get_io_directories
from registration_utils import collect_input_files
from registration_utils import bspline_registration
from registration_utils import compute_average_image
from registration_utils import retrieve_last_template
from registration_utils import check_registration_scores
from registration_utils import compute_the_average_deformation_field
from registration_utils import compute_the_inverse_average_transformation
from registration_utils import create_output_directories_bspline_registration


###############################################################################


""" Parameter initialization by user """

## Current pixel resolution (values used in the paper: 45, 30, 15, 9, 6, 3)
res = 6

## Iteration counter
iter_index = 1

## Previous registration step
previous_registration_step = 'bspline'  # possible values: 'global', 'bspline'

## Previos pixel resolution
if previous_registration_step == 'bspline':
    previous_pixel_resolution = 15
else:
    previous_pixel_resolution = 'Native'


###############################################################################


""" Path setup : Find path organization in path_structure.txt """

## Collect input/output directories
data_dir, output_dir, global_registration_dir, res_output_dir = get_io_directories(res)


## Collect input fixed and moving image file paths
if iter_index==0:
    last_template_file_path = retrieve_last_template(previous_registration_step,
                                                     previous_pixel_resolution,
                                                     res,
                                                     output_dir)
    ## collect registered downsampled images from the global registration step
    subject_names_list, subject_file_pathes_list = collect_input_files(
                                              global_registration_dir, res)
else:
    file_path_to_load = os.path.join(res_output_dir, 'files_path.pkl')
    with open(file_path_to_load, 'rb') as f:
        subject_names_list, subject_file_pathes_list, last_template_file_path = \
                                                                 pickle.load(f)

## Collect Elastix parameters file paths
elastix_parameters_bspline_file_path = os.path.join(data_dir, 
                                                    "elastix_parameter_files",
                                                    f"bspline_res{res}um.txt")

elastix_parameters_inverse_bspline_file_path = os.path.join(data_dir, 
                                                    "elastix_parameter_files",
                                                    f"bspline_res{res}um_inverse.txt")

## Create output directories
subject_output_directories, average_output_dir, template_output_dir,\
    inverse_average_transformation_output_dir =\
        create_output_directories_bspline_registration(res_output_dir, 
                                                       iter_index,
                                                       subject_names_list)

## Create/get MMADF file
if iter_index==0:
    MMADF_file_path = create_MMADF_file(res_output_dir, "MMADF.txt")
else:
    MMADF_file_path = os.path.join(res_output_dir, "MMADF.txt")


###############################################################################


""" Do the registration. Save the registered image for visual inspection. """

for subject_name in subject_names_list:
    print("\n--- START registration of " + subject_name + " ---\n")
    
    moving_image_path = [img_path for img_path in subject_file_pathes_list if
                         subject_name in img_path][0]
    
    subject_output_directory = [subject_output_dir for subject_output_dir in
                                subject_output_directories if subject_name 
                                in subject_output_dir][0]
    
    bspline_registration(last_template_file_path, 
                         moving_image_path, 
                         elastix_parameters_bspline_file_path,
                         subject_output_directory)
    
    print("\n--- END of registration of " + subject_name + " ---")


## Optional: Check the quality of registration through looking at registration scores.
check_registration_scores(subject_output_directories)


## Remove artifacts (produced by bspline registration) by thresholding
for image_dir in subject_output_directories:
    image_path = os.path.join(image_dir, 'result.0.tif')
    threshold_image(image_path, threshold=60000)
    

###############################################################################


""" Compute the average image and save it """

average_image = compute_average_image(subject_output_directories)

## save the average image
average_image_output_path = os.path.join(average_output_dir, 
                                         f'average_image_{res}um_'+
                                         f'iter{iter_index}.tif')
with tifffile.TiffWriter(average_image_output_path) as tif:
    for frame in average_image:
        tif.save(frame, contiguous=True)

print("\n\nChange the voxel unit of the average image to pixel in Fiji.")
print("\nThe average image is found in: \n\t" + average_image_output_path)
a = input("\nThen, press any key to continue ... ")


###############################################################################


""" Compute the average of deformation fields and save it """

MMADF, average_transformParameters_filePath = compute_the_average_deformation_field(
                                                subject_output_directories, 
                                                average_output_dir, iter_index)

## Write the MMADF on the specified text file created above
with open(MMADF_file_path, "a") as f:
    f.write(str(iter_index)+" "*10+str(MMADF)+"\n")


###############################################################################


""" Compute the inverse average transformation """

inverse_tranformParameters_file_path = compute_the_inverse_average_transformation(
                                           last_template_file_path,
                                           average_transformParameters_filePath,
                                           elastix_parameters_inverse_bspline_file_path,
                                           inverse_average_transformation_output_dir
                                           )


###############################################################################


""" Apply the inverse average transformation to the average image """

run_transformix(average_image_output_path, 
                template_output_dir,
                inverse_tranformParameters_file_path)

## Remove artifacts by thresholding
new_template_file_path = os.path.join(template_output_dir, 'result.tif') 
threshold_image(new_template_file_path, threshold=60000)

print("\n\nChange the voxel unit of the created template image to pixel in Fiji.")
print("\nThe created template image is found in: \n\t" + new_template_file_path)
a = input("\nThen, press any key to continue ... ")


###############################################################################


""" save some variables to load in the next iteration """

file_path = os.path.join(res_output_dir, 'files_path.pkl')
with open(file_path, 'wb') as f:
    pickle.dump([subject_names_list, subject_file_pathes_list,
                 new_template_file_path], f)
    

###############################################################################

"""
To proceed with more iterations, increase the value of iter_index variable by 1,
and rerun the code.
"""

###############################################################################

"""
Postprocessing steps in Fiji:
    After all iterations for the current resolution v1, before going to a 
    higher resolution v2:
        
    A) Template upsampling in Fiji (Image -> Scale...):
       Upsample the last created template (created_template/result.tif)
       to the resolution v2 and save it as result-res{v2}um.tif in 
       the same directory.
    
    B) Change the values of the following variables on top of the code and 
       rerun the code:
           res = v2
           iter_index = 0
           previous_registration_step = 'bspline'
           previos_pixel_resolution = v1
"""
