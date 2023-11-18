"""
reference: https://github.com/yehyunsuh/Landmark-Annotator
"""

import os
import cv2
import argparse
import shutil
import numpy as np
from glob import glob

clicked_points = []
clone = None
colors = ()


def MouseLeftClick(event, x, y, _, __):
    """
    Activated at every left click and draws the added annotation on the image
    :param event: cv2 event
    :param x: x coordinate of the left click
    :param y: y coordinate of the left click
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((y, x))
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[1], point[0]), 5, colors[2], thickness=-1)
        cv2.imshow("image", image)


def annotator(args):
    """
    This function saves txt files based on the annotations done in each image
    :param args: arguments from argparser
    """
    global clone, clicked_points, colors

    removed_image_path_list = sorted(glob(f'{args.removed_image_path}/**/*.png', recursive=True))
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0))  # BGR
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", MouseLeftClick)
    count = 0    

    while True:
        annotated = False
        image_path = removed_image_path_list[count]
        patient_id = image_path.split('/')[-3]
        image_type = image_path.split('/')[-2]
        image_name = image_path.split('/')[-1].split('.')[0]

        original_image = cv2.imread(image_path)
        cropped_image = cv2.imread(image_path)
        clone = original_image.copy()

        file_read = open("./preprocess/checkpoint.txt", "r")
        lines = file_read.readlines()
        for line in lines:
            if image_name == line.strip():
                annotated = True
                count += 1
                break
            file_read.close()
        # if annotated:
        #     print(f'{image_name} has been already annotated')

        if not annotated:
            while True:
                image = cv2.imread(image_path)
                for point in clicked_points:
                    cv2.circle(image, (point[1], point[0]), 5, colors[2], thickness=-1)
                cv2.imshow("image", image)
                key = cv2.waitKey(0)
                print(image_name, key)

                # when you press b - erase the most recent anntation
                if key == ord("b"):
                    if len(clicked_points) > 0:
                        clicked_points.pop()

                # when you press n - moves to the next image after saving the annotation
                if key == ord("n"):
                    file_write_txt = image_name + "\n"
                    file_write = open("./preprocess/checkpoint.txt", "a+")
                    file_write.write(file_write_txt)
                    file_write.close()
                    count += 1
                    # if there has been clicks
                    if clicked_points != []:
                        # crop image
                        for i in range(int(len(clicked_points)/4)):
                            x1, y1 = clicked_points[4*i]
                            x2, y2 = clicked_points[4*i+1]
                            x3, y3 = clicked_points[4*i+2]
                            x4, y4 = clicked_points[4*i+3]

                            if y1 < 50 and y2 < 50:
                                y1, y2 = 0, 0
                            if y3 > image.shape[1]-50 and y4 > image.shape[1]-50:
                                y3, y4 = image.shape[1]-1, image.shape[1]-1
                            if x1 < 70 and x4 < 70:
                                x1, x4 = 0, 0
                            if x2 > image.shape[0]-70 and x3 > image.shape[0]-70:
                                x2, x3 = image.shape[0]-1, image.shape[0]-1

                            pnts = np.array([[[y1, x1], [y2, x2], [y3, x3], [y4, x4]]], dtype=np.int32)
                            cv2.fillPoly(cropped_image, pnts, color=colors[3])                        

                        # save text erased image
                        os.makedirs(f'{args.erased_path}/{patient_id}/{image_type}', exist_ok=True)
                        cv2.imwrite(f'{args.erased_path}/{patient_id}/{image_type}/{image_name}.png', cropped_image)
                    else:
                        # copy paste image without erasing
                        os.makedirs(f'{args.erased_path}/{patient_id}/{image_type}', exist_ok=True)
                        shutil.copy(image_path, f'{args.erased_path}/{patient_id}/{image_type}/{image_name}.png')

                    clicked_points = []
                    break

                if key == ord("p"):
                    count -= 1
                    clicked_points = []
                    break
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--removed_image_path", default="./data/view_ap/patient_cropped_LR_removed")
    parser.add_argument("--erased_path", default="./data/view_ap/patient_cropped_LR_removed_erased")
    args = parser.parse_args()

    # create directory where txt file will be saved
    os.makedirs(args.erased_path, exist_ok=True)
    annotator(args)


# import shutil
# import cv2
# import os
# from glob import glob
# from tqdm import tqdm

# cropped_image_path_list = sorted(glob('data/view_ap/patient_cropped/**/*.png', recursive=True))
# text_erased_path = 'data/view_ap/patient_cropped'

# for image_path in tqdm(cropped_image_path_list):    
#     image_path_split = image_path.split('/')
#     patient_id = image_path_split[3]  # 8 digit id
#     image_type = image_path_split[4]  # pre/post
#     image_name = image_path_split[5].split('.')[0]

#     image = cv2.imread(image_path)
#     print(image_path)
#     print(image_name)
#     cv2.imshow('image', image)
#     cv2.waitKey(0) 
#     cv2.destroyAllWindows() 

#     # cropped_patient_path = f'{cropped_path}/{patient_id}'
#     # if image_type == "pre":
#     #     cropped_patient_path = f'{cropped_patient_path}/pre'
#     # else:
#     #     cropped_patient_path = f'{cropped_patient_path}/post'
#     # os.makedirs(cropped_patient_path, exist_ok=True)

#     # image = cv2.imread(image_path)
#     # # this method is just dividing the image in half horizontally
#     # # if the image is not divided, user will have to do it manually
#     # if image.shape[0] < image.shape[1]:
#     #     # left image (left based on patient)
#     #     left_image = image[:, int(image.shape[1]/2):, :]
#     #     left_image_path = f'{cropped_patient_path}/{image_name}_left.png'
#     #     cv2.imwrite(left_image_path, left_image)

#     #     # right image (right based on patient)
#     #     right_image = image[:, :int(image.shape[1]/2), :]
#     #     right_image_path = f'{cropped_patient_path}/{image_name}_right.png'
#     #     cv2.imwrite(right_image_path, right_image)

#     # # original image
#     # original_image_path = f'{cropped_patient_path}/{image_name}.png'
#     # shutil.copy(image_path, original_image_path)

#     break