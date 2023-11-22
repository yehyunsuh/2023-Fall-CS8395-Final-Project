import os
import shutil
from tqdm import tqdm
from glob import glob


def divide_per_patient(pre_list, post_list, data_type):
    """
    Function that divides images based on patient's ID
    :param pre_list (list): list of pre-op images
    :param post_list (list): list of post-op images
    :param data_type (str): data type "view_ap" or "view_lateral"
    :return: None
    """
    os.makedirs(f'../data/{data_type}/patient', exist_ok=True)
    check_list = []
    for i in tqdm(range(len(pre_list))):
        patient_pre, patient_post = [], []
        patient_id = pre_list[i].split("/")[-1].split("_")[0]
        os.makedirs(
            f'../data/{data_type}/patient/{patient_id}', exist_ok=True)
        os.makedirs(
            f'../data/{data_type}/patient/{patient_id}/pre', exist_ok=True)
        os.makedirs(
            f'../data/{data_type}/patient/{patient_id}/post', exist_ok=True)

        # Check if this patient data has been already copied
        if patient_id not in check_list:
            patient_pre.append(pre_list[i])
            # Save same patient's pre-op image path
            for j in range(len(pre_list)):
                patient_id_pre = pre_list[j].split("/")[-1].split("_")[0]
                if i != j and patient_id == patient_id_pre:
                    patient_pre.append(pre_list[j])

            # Save same patient's post-op image path
            for k in range(len(post_list)):
                patient_id_post = post_list[k].split("/")[-1].split("_")[0]
                if i != k and patient_id == patient_id_post:
                    patient_post.append(post_list[k])

            # After getting the image path is done, add to check list
            check_list.append(patient_id)

        # Check if patient does not have one of the data
        if patient_pre != [] or patient_post != []:
            # Copy pre-op data
            for i in range(len(patient_pre)):
                file_name = patient_pre[i].split('/')[-1].split('.')[0]
                png_name = f"{file_name}.png"
                shutil.copy(
                    patient_pre[i],
                    f'../data/{data_type}/patient/{patient_id}/pre/{png_name}'
                )

            # Copy post-op data
            for i in range(len(patient_post)):
                file_name = patient_post[i].split('/')[-1].split('.')[0]
                png_name = f"{file_name}.png"
                shutil.copy(
                    patient_post[i],
                    f'../data/{data_type}/patient/{patient_id}/post/{png_name}'
                )
        else:
            print(f'Number of data for patient {patient_id}: \
                  pre {len(patient_pre)}, post {len(patient_post)}')


lateral_pre = sorted(glob("../data/view_lateral/pre_op/*.png"))
lateral_post = sorted(glob("../data/view_lateral/post_op/*.png"))

ap_pre = sorted(glob("../data/view_ap/pre_op/*.png"))
ap_post = sorted(glob("../data/view_ap/post_op/*.png"))

divide_per_patient(ap_pre, ap_post, 'view_ap')
divide_per_patient(lateral_pre, lateral_post, 'view_lateral')
