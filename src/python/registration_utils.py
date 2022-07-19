
"""
Utils used in global and bspline registration steps
"""

import os
import tifffile
import numpy as np
import SimpleITK as sitk
from shutil import rmtree
import pickle
from tqdm import tqdm
from glob import glob


###############################################################################


def get_io_directories(current_pixel_resolution):
    """
    Get input/output directories for the current resolution.
    Find the organization of directories in path_structure.txt file.

    Parameters
    ----------
    current_pixel_resolution : int
        Current pixel resolution.

    Returns
    -------
    data_dir : str
        Input data directory.
    general_output_dir : str
        Main output diectory.
    global_registration_dir : str
        Global registration result directory.
    res_output_dir : str
        Bspline registration result directory for the current resolution.

    """
    working_directory = os.getcwd()
    project_dir = os.path.dirname(os.path.dirname(working_directory))
    data_dir = os.path.join(project_dir, "data")
    general_output_dir = os.path.join(project_dir, "output")
    global_registration_dir = os.path.join(general_output_dir, "global_registration")
    res_output_dir = os.path.join(general_output_dir, "bspline_registration",
                                  f"resolution_{current_pixel_resolution}um")
    
    return data_dir, general_output_dir, global_registration_dir, res_output_dir


def retrieve_last_template(previous_registration_step, 
                           previos_pixel_resolution,
                           current_pixel_resolution,
                           general_output_dir):
            
    if previous_registration_step=='global':
        global_registration_dir = os.path.join(general_output_dir, "global_registration")
        last_template_file_path = os.path.join(global_registration_dir, 
                                               "template", 
                                               f"T0-res{current_pixel_resolution}um.tif")
    elif previous_registration_step=='bspline':
        previous_registration_output_dir = os.path.join(general_output_dir, 
                                                        "bspline_registration",
                                                        f"resolution_{previos_pixel_resolution}um"
                                                        )
        iter_dirs = glob(os.path.join(previous_registration_output_dir, 'iter_*'))
        iter_dirs.sort()
        last_iteration_dir = iter_dirs[-1]
        last_template_file_path = os.path.join(last_iteration_dir, 
                                               "collective_outputs", 
                                               "created_template", 
                                               "result.tif")
        
        ## modify the template file name to return the upsampled template
        file_path, file_ext = os.path.splitext(last_template_file_path)
        last_template_file_path = file_path + f'-res{current_pixel_resolution}um' + file_ext
        
    return last_template_file_path


def collect_input_files(global_registration_dir, res):
    """
    Collect the file path of registered downsampled images in the Global Registration step

    Parameters
    ----------
    global_registration_dir : str
        Directory of the global registration results.
    res : int
        Current resolution.

    Returns
    -------
    subject_names_list : list of str
        List of moving image names
    subject_file_pathes_list : list of str
        List of file paths of registered downsampled images in the Global Registration step.

    """
    ## find the list of globally registered downsampled images
    globally_registered_image_dir_list = os.listdir(global_registration_dir)
    
    subject_names_list = []
    subject_file_pathes_list = []
    
    for image_dir in globally_registered_image_dir_list:
        if image_dir!='template':
            img_path = os.path.join(global_registration_dir, image_dir, 
                                    'affine', f'result.0-res{res}um.tif')
            subject_names_list.append(image_dir)
            subject_file_pathes_list.append(img_path)
            
    return subject_names_list, subject_file_pathes_list


def create_path(dir_to_be_created):
    """
    Create a directory. Warn the user if the directory already exists.

    Parameters
    ----------
    dir_to_be_created : str
        Path of the directory

    Returns
    -------
    None.

    """
    if not os.path.exists(dir_to_be_created):
        os.makedirs(dir_to_be_created)
    else:
        print("\n\n!!! " + dir_to_be_created + " already exists !!!")
        while True:
            replace_input = input("\nDo you want to rewrite it?  y/n ...   ")
            if replace_input.lower()=="y":
                rmtree(dir_to_be_created)
                os.makedirs(dir_to_be_created, exist_ok=True)
                break
            elif replace_input.lower()=="n":
                print("\n\n!!! Attention! All the folder contents will be replaced with new contents !!!")
                print("Consider to copy important contents of the folder elsewhere, then")
                input("press any key to continue ...")
                break


def create_output_directories_global_registration(output_dir, images_fname_list):
    """
    Create output directory for:
        1) each subject to save the similarity and affine transformations and
           registered image
        2) the average image

    Parameters
    ----------
    output_dir : str
        Output directory.
        
    images_fname_list : list of str
        List of moving image file names 
        e.g. [img1.tif, img2.tif]

    Returns
    -------
    subject_output_directories : list of str
    
    template_output_dir : str
        Average image output directory

    """
    subject_output_directories = []
    for fname in images_fname_list:
        subject_name = fname.split('.')[0] # remove .tif
        subject_output_dir = os.path.join(output_dir, 'global_registration',
                                          subject_name)
        similarity_dir = os.path.join(subject_output_dir, 'similarity')
        affine_dir = os.path.join(subject_output_dir, 'affine')
        create_path(similarity_dir)
        create_path(affine_dir)
        subject_output_directories.append(subject_output_dir)
        
    ### create output directory for the average image
    template_output_dir = os.path.join(output_dir, 'global_registration',
                                                   'template')
    create_path(template_output_dir)
   
    return subject_output_directories, template_output_dir


def create_output_directories_bspline_registration(res_output_dir, iter_index,
                                                   subject_names_list):
    """
    create output directory for:
        1) each subject to save the bspline transformation and registered image
        2) collective outputs e.g. the average image, inverse average transformation info

    Parameters
    ----------
    res_output_dir : str
        Output directory for current resolution.
    iter_index : int
        Current iteration index.
    subject_names_list : List of str
        List of moving image names.

    Returns
    -------
    subject_output_directories : List of str
        List of output directories for all moving images.
    average_output_dir : str
        Output directory for the average image and the average deformation field.
    template_output_dir : str
        Output directory for the created template.
    inverse_average_transformation_output_dir : str
        Output directory for inverse average transformation.        

    """
    ## output directory for current iteration
    iter_res_output_dir = os.path.join(res_output_dir, f'iter_{iter_index}')
    
    ## create output directory for each subject
    subject_output_directories = []
    for subject_name in subject_names_list:
        subject_output_dir = os.path.join(iter_res_output_dir, subject_name)
        create_path(subject_output_dir)
        subject_output_directories.append(subject_output_dir)
    
    ## create output directory for the average image and the average deformation field
    average_output_dir = os.path.join(iter_res_output_dir, 
                                      'collective_outputs',
                                      'average_image_and_deformation_field')
    create_path(average_output_dir)
    
    ## create output directory for the created template
    template_output_dir = os.path.join(iter_res_output_dir, 
                                       'collective_outputs',
                                       'created_template')
    create_path(template_output_dir)
    
    ## create output directory for inverse average transformation
    inverse_average_transformation_output_dir = os.path.join(
                                           iter_res_output_dir,
                                           'collective_outputs',
                                           'inverse_average_deformation_field')
    create_path(inverse_average_transformation_output_dir)
    
    return subject_output_directories, average_output_dir, template_output_dir,\
           inverse_average_transformation_output_dir


def create_output_directories_stained_image_registration(output_dir, 
                                                        stained_images_fname_list,
                                                        staining):
    """
    Create output directory for:
        1) each subject to save the transformix log file and registered image
        2) the average image

    Parameters
    ----------
    output_dir : str
        Output directory.
    stained_images_fname_list : list of str
         List of stained image names.
    staining : str
        Type of staining e.g. 'ChAT' or 'TH'.

    Returns
    -------
    subject_output_directories : List of str
        List of output directories for all stained images.
    template_output_dir : str
        Output directory for the created average template.

    """
    subject_output_directories = []
    for fname in stained_images_fname_list:
        subject_output_dir = os.path.join(output_dir, staining+'_registration', fname)
        create_path(subject_output_dir)
        subject_output_directories.append(subject_output_dir)
        
    ### create output directory for the average image
    template_output_dir = os.path.join(output_dir, staining+'_registration', 'template')
    create_path(template_output_dir)
   
    return subject_output_directories, template_output_dir


def create_MMADF_file(res_output_dir, fname):
    """
    Create a text file to save MMADF (mean magnitude average deformation field).
    It will be used as a metric to stop the iteration of registration.

    Parameters
    ----------
    res_output_dir : str
        Output directory for current resolution.
    fname : str
        MMADF file name

    Returns
    -------
    MMADF_file_path : str

    """
    MMADF_file_path = os.path.join(res_output_dir, fname)
    if not os.path.isfile(MMADF_file_path):
        with open(MMADF_file_path, 'w') as f:
            f.write("MMADF (mean magnitude average deformation field) for each"+
                    " iteration of registration process\n\n")
            f.write("iteration        MMADF\n")
            f.write("----------------------\n")
        print("\n\nA text file named <<MMADF.txt>> was created in the path:  <<" +
              MMADF_file_path+ ">>  to save the MMDF score of registration.\n\n")
    
    return MMADF_file_path
           
           
def similarity_registration(fixed_image_path, 
                            moving_image_path, 
                            elastix_parameters_similarity_file_path,
                            subject_output_directory):
    """
    Similarity registration

    Parameters
    ----------
    fixed_image_path : str
    moving_image_path : str
    elastix_parameters_similarity_file_path : str
    subject_output_directory : str

    Returns
    -------
    None.

    """
    subject_similarity_output_directory = os.path.join(subject_output_directory, 'similarity')
    run_elastix(fixed_image_path, 
                moving_image_path, 
                elastix_parameters_similarity_file_path,
                None,
                subject_similarity_output_directory)


def affine_registration(fixed_image_path, 
                        elastix_parameters_affine_file_path,
                        subject_output_directory):
    """
    Affine registration

    Parameters
    ----------
    fixed_image_path : str
    elastix_parameters_affine_file_path : str
    subject_output_directory : str

    Returns
    -------
    None.

    """
    subject_affine_output_directory = os.path.join(subject_output_directory, 'affine')
    moving_image_path = os.path.join(subject_output_directory, 'similarity', 'result.0.mhd')
    run_elastix(fixed_image_path, 
                moving_image_path, 
                elastix_parameters_affine_file_path,
                None,
                subject_affine_output_directory)


def bspline_registration(fixed_image_path, 
                         moving_image_path, 
                         elastix_parameters_bspline_file_path,
                         subject_output_directory):
    """
    Bspline registration

    Parameters
    ----------
    fixed_image_path : str
    moving_image_path : str
    elastix_parameters_bspline_file_path : str
    subject_output_directory : str

    Returns
    -------
    None.

    """
    run_elastix(fixed_image_path, 
                moving_image_path, 
                elastix_parameters_bspline_file_path,
                None,
                subject_output_directory)


def run_elastix(fixed_image_path, 
                moving_image_path, 
                elastix_parameters_file_path,
                initial_transformation_file_path,
                subject_output_directory):
    """
    Run elastix via command line

    Parameters
    ----------
    fixed_image_path : str
    moving_image_path : str
    elastix_parameters_file_path : str
    initial_transformation_file_path: str
    subject_output_directory : str

    Returns
    -------
    None.

    """
    if initial_transformation_file_path is None:
        os.system("elastix" + \
                  " -f " + fixed_image_path + \
                  " -m " + moving_image_path + \
                  " -out " + subject_output_directory + \
                  " -p " + elastix_parameters_file_path
                  )
    else:
        os.system("elastix" + \
                  " -f " + fixed_image_path + \
                  " -m " + moving_image_path + \
                  " -out " + subject_output_directory + \
                  " -p " + elastix_parameters_file_path + \
                  " -t0 " + initial_transformation_file_path
                  )


def run_transformix(image_path, 
                    output_directory,
                    tranformParameters_file_path):
    """
    Run Transformix via command line

    Parameters
    ----------
    image_path : str
        input image file path.
    output_directory : str
    tranformParameters_file_path : str

    Returns
    -------
    None.

    """
    os.system("transformix" + \
              " -in " + image_path + \
              " -out " + output_directory + \
              " -tp " +  tranformParameters_file_path)

    
def check_registration_scores(subject_output_directories):
    """
    Check the quality of bspline registration through looking at registration scores.
    It is another check besides the visual inspection of registered images.

    Parameters
    ----------
    subject_output_directories : List of str
        List of registered image directories.

    Returns
    -------
    None.

    """
    print("\n\n --- Registration scores ---")
    print("\n subject_name\tregistration score")
    print("-"*80)
    
    for subject_output_directory in subject_output_directories:
        
        subject_name = os.path.split(subject_output_directory)[1]
    
        registration_score_file_path = os.path.join(subject_output_directory,
                                                    "IterationInfo.0.R3.txt")
        try:
            with open(registration_score_file_path) as f:
                content = f.readlines()
            content = content[-1] # last line of the content
            forward_reg_score = content.split('\t')[1]
        except:
            # The forward transformation couldn't be computed
            forward_reg_score = "--------"
        
        print("  " + subject_name + "\t\t\t\t" + forward_reg_score)  


def threshold_image(image_path, threshold):
    """
    Threshold a given image by given criteria.

    Parameters
    ----------
    image_path : str
    threshold : int

    Returns
    -------
    None.

    """
    ## read the image
    img_array = tifffile.imread(image_path)
    
    ## threshold the image
    img_array[img_array>threshold] = 0
   
    ## write the image
    with tifffile.TiffWriter(image_path) as tif:
        for frame in img_array:
            tif.save(frame, contiguous=True)


def signal_attenuation(input_image_path, attenuation_threshold=60000,
                       attenuation_factor=10000):
    """
    Attenuate intensity of highly bright pixels

    Parameters
    ----------
    input_image_path : str
    attenuation_threshold : int, optional
        The default is 60000.
    attenuation_factor : ont, optional
        The default is 10000.

    Returns
    -------
    None.

    """
    img = tifffile.imread(input_image_path)
    img[img>attenuation_threshold] -= attenuation_factor
    with tifffile.TiffWriter(input_image_path) as tif:
        for frame in img:
            tif.save(frame, contiguous=True)
            

def compute_average_image(image_directories):
    """
    Parameters
    ----------
    image_directories : List of str
        List of image directories.

    Returns
    -------
    average_image : Numpy array
        16-bits average image.

    """
    img_path = os.path.join(image_directories[0], 'result.0.tif')
    img = tifffile.imread(img_path)
    sum_img_array = np.zeros(img.shape)
    
    for image_dir in tqdm(image_directories):
        img_path = os.path.join(image_dir, 'result.0.tif')
        sum_img_array += tifffile.imread(img_path)
        
    average_image = sum_img_array/(len(image_directories))
    average_image = average_image.astype('uint16')

    del sum_img_array
    
    return average_image


def compute_average_stained_image(image_directories):
    """
    Parameters
    ----------
    image_directories : List of str
        List of image directories.

    Returns
    -------
    average_image : Numpy array
        16-bits average image.

    """
    img_path = os.path.join(image_directories[0], 'result.tif')
    img = tifffile.imread(img_path)
    sum_img_array = np.zeros(img.shape)
    
    for image_dir in tqdm(image_directories):
        img_path = os.path.join(image_dir, 'result.tif')
        sum_img_array += tifffile.imread(img_path)
        
    average_image = sum_img_array/(len(image_directories))
    average_image = average_image.astype('uint16')

    del sum_img_array
    
    return average_image


def compute_the_average_deformation_field(subject_output_directories, 
                                          average_output_dir,
                                          iter_index):
    """
    compute the average deformation field across all registered images

    Parameters
    ----------
    subject_output_directories : List of str
        List of registered image directories.
    average_output_dir : str
        Output directory for the average image and the average deformation field.
    iter_index : int
        Current iteration index.

    Returns
    -------
    MMADF : float
        Mean magnitude average deformation field.
    average_transformParameters_filePath : str

    """
    ## read the TransformParameters files
    tranformParameters_all_subjects = []
    for subject_dir in subject_output_directories:
        tranformParameters_file_path = os.path.join(subject_dir, 
                                                    'TransformParameters.0.txt')
        tranformParameters = sitk.ReadParameterFile(tranformParameters_file_path)
        tranformParameters_all_subjects.append(tranformParameters["TransformParameters"])
    
    ## string to float
    tranformParameters_all_subjects_float = params_string_to_float(
                                               tranformParameters_all_subjects)
    
    ## compute the average transform parameters
    average_transform_parameters = np.mean(tranformParameters_all_subjects_float,
                                           axis=0)
    
    ## cast the average transform parameters to the string type
    average_transform_parameters_string = params_float_to_string(average_transform_parameters)
    
    ### save the average transform parameters in a file, readable by Elastix and Transformix
    ## take the last tranformParameters file as a template and replace its
    ## "TransformParameters" field with the average transform parameters
    tranformParameters["TransformParameters"] = average_transform_parameters_string
    
    # write the modified TransformParameters file to the disk
    average_transformParameters_filePath = os.path.join(
                            average_output_dir, 
                            f"Average_TransformParameters_iter{iter_index}.txt")
    sitk.WriteParameterFile(tranformParameters, 
                            average_transformParameters_filePath)
    print("\n\nAverage transformation parameters were written in")
    print('<<' + average_transformParameters_filePath + '>>\n\n')
    ## compute MMADF
    MMADF = np.mean(np.abs(average_transform_parameters))
        
    return MMADF, average_transformParameters_filePath
    

def params_string_to_float(tranformParameters_all_subjects):
    tranformParameters_all_subjects_float = []
    for tuple_ in tranformParameters_all_subjects:
        list_ = []
        for element in tuple_:
            list_.append(float(element))
        tranformParameters_all_subjects_float.append(list_)
    return tranformParameters_all_subjects_float


def params_float_to_string(average_transform_parameters):
    average_transform_parameters_string = []
    for element in average_transform_parameters:
        element = round(element, 6)
        average_transform_parameters_string.append(str(element))
    return average_transform_parameters_string


def compute_the_inverse_average_transformation(fixed_image_path, 
                                               average_transformParameters_filePath,
                                               elastix_parameters_file_path,
                                               inverse_average_transformation_output_dir
                                               ):
    """
    Compute the inverse of average transformation.
    Some fields of the inverse transformation is modified according to the
    Elastix 5.0.0 user manual [Stefan Klein and Marius Staring, October 10, 2019,
                               page 38, section 6.1.6]

    Parameters
    ----------
    fixed_image_path : str
    average_transformParameters_filePath : str
    elastix_parameters_file_path : str
    inverse_average_transformation_output_dir : str

    Returns
    -------
    inverse_tranformParameters_file_path : str

    """
    run_elastix(fixed_image_path, 
                fixed_image_path, 
                elastix_parameters_file_path,
                average_transformParameters_filePath,
                inverse_average_transformation_output_dir)
        
    ## Modify some fields of the inverse transformation
    inverse_tranformParameters_file_path = os.path.join(inverse_average_transformation_output_dir, 
                                                        'TransformParameters.0.txt')
    inverse_tranformParameters = sitk.ReadParameterFile(inverse_tranformParameters_file_path)
    inverse_tranformParameters["InitialTransformParametersFileName"] = ["NoInitialTransform"]
    
    # write the modified inverse TransformParameters file to the disk
    sitk.WriteParameterFile(inverse_tranformParameters, inverse_tranformParameters_file_path)
    
    return inverse_tranformParameters_file_path
