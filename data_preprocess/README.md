# preprocess

All the data preprocessing code goes here. Specific details of each file will to updated later.
```shell
python3 divide_image_by_patient.py
python3 divide_two_leg.py
python3 erase_box.py
python3 flip.py
```

image
patient
patient_cropped
patient_cropped_LR
patient_cropped_LR_removed
patient_cropped_LR_removed_erased
patient_cropped_LR_removed_erased_flipped
post_op
pre_op

1. 
post_op - raw dataset
pre_op - raw dataset

2. divide_image_by_patient.py
patient

3. divide_two_leg.py
patient_cropped

4. Manually checked if the image has right or left knee
patient_cropped_LR

5. removed unecessary & health knee
patient_cropped_LR_removed

6. erase_box.py
erased LR indicator and the letter
patient_cropped_LR_removed_erased

7. python3 flip.py
flip left knee to look like it is right knee (vertical flip)
patient_cropped_LR_removed_erased_flipped
image - divided based on pre/post op
