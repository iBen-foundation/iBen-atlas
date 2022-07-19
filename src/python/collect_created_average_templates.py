
"""
This code collects the average autofluorescence (AF) and stained average image 
template files created through bspline_registration.py and 
stained_images_registration.py respectively.
"""

import os
import shutil

from registration_utils import get_io_directories


###############################################################################


""" Parameter initialization by user """

## Final image resolution of bspline registration of AF channel
res = 6

## Final iteration number of bspline registration of AF channel
final_iter_index = 1

## List of directory names where stained images are found e.g. 'ChAT', 'TH'
staining_list = ['TH']


###############################################################################


""" Collect average template files """

## Collect input/output directories
_, output_dir, __, res_output_dir = get_io_directories(res)

## Create a directory to paste all average template files
created_average_templates_path = os.path.join(output_dir, 
                                              "created_average_templates")
os.makedirs(created_average_templates_path)

## copy/paste AF average template file
final_AF_template_file = os.path.join(res_output_dir,
                                     f"iter_{final_iter_index}",
                                     "collective_outputs",
                                     "created_template",
                                     "result.tif")
dest_path = os.path.join(created_average_templates_path, "AF_template.tif")
shutil.copy2(final_AF_template_file, dest_path)


## copy/paste stained average template file
for staining_dir in staining_list:
    final_staining_template_file = os.path.join(output_dir,
                                                f"{staining_dir}_registration",
                                                "template",
                                                "result.tif")
    dest_path = os.path.join(created_average_templates_path, 
                             f"{staining_dir}_template.tif")
    shutil.copy2(final_staining_template_file, dest_path)
    
                                              