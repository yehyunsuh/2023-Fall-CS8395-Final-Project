import cv2
import os
import shutil
from glob import glob
from tqdm import tqdm

image_path_list = sorted(glob('data/view_ap/patient/**/*.png', recursive=True))
cropped_path = 'data/view_ap/patient_cropped'

for image_path in tqdm(image_path_list):
    image_path_split = image_path.split('/')
    patient_id = image_path_split[3]  # 8 digit id
    image_type = image_path_split[4]  # pre/post
    image_name = image_path_split[5].split('.')[0]

    cropped_patient_path = f'{cropped_path}/{patient_id}'
    if image_type == "pre":
        cropped_patient_path = f'{cropped_patient_path}/pre'
    else:
        cropped_patient_path = f'{cropped_patient_path}/post'
    os.makedirs(cropped_patient_path, exist_ok=True)

    image = cv2.imread(image_path)
    # this method is just dividing the image in half horizontally
    # if the image is not divided, user will have to do it manually
    if image.shape[0] < image.shape[1]:
        # left image (left based on patient)
        left_image = image[:, int(image.shape[1]/2):, :]
        left_image_path = f'{cropped_patient_path}/{image_name}_left.png'
        cv2.imwrite(left_image_path, left_image)

        # right image (right based on patient)
        right_image = image[:, :int(image.shape[1]/2), :]
        right_image_path = f'{cropped_patient_path}/{image_name}_right.png'
        cv2.imwrite(right_image_path, right_image)

    # original image
    original_image_path = f'{cropped_patient_path}/{image_name}.png'
    shutil.copy(image_path, original_image_path)
