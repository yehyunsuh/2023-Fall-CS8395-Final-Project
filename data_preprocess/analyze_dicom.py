import pydicom
from glob import glob

dicom_path_list = glob('./data/dicom/*')

for dicom_path in dicom_path_list:
    dcm_info = pydicom.read_file(f'{dicom_path}', force=True)
    # print(dicom_path)
    # print(dcm_info)

    file = open(f'./data/dicom_txt/{dicom_path.split("/")[-1]}.txt', 'a')
    print(dcm_info, file=file)