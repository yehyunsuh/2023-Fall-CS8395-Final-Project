import cv2
import shutil
import os
from glob import glob
from tqdm import tqdm

erased_path_list = sorted(glob('data/view_ap/patient_cropped_LR_removed_erased/**/*.png', recursive=True))
flipped_path = 'data/view_ap/patient_cropped_LR_removed_erased_flipped'
dataset_path = 'data/view_ap/image'

os.makedirs(f'{dataset_path}/post', exist_ok=True)
os.makedirs(f'{dataset_path}/pre', exist_ok=True)

for erased_image in tqdm(erased_path_list):
    patient_id = erased_image.split('/')[-3]
    image_type = erased_image.split('/')[-2]
    image_name = erased_image.split('/')[-1].split('.')[0]
    os.makedirs(f'{flipped_path}/{patient_id}/{image_type}', exist_ok=True)

    if "left" in image_name:
        image = cv2.imread(erased_image)
        flipped_image = cv2.flip(image, 1)
        save_path = f'{flipped_path}/{patient_id}/{image_type}/{image_name}_right.png'
        cv2.imwrite(save_path, flipped_image)
    else:
        save_path = f'{flipped_path}/{patient_id}/{image_type}/{image_name}.png'
        shutil.copy(erased_image, save_path)

    image_name2 = save_path.split('/')[-1].split('.')[0]
    shutil.copy(save_path, f'{dataset_path}/{image_type}/{image_name2}.png')
