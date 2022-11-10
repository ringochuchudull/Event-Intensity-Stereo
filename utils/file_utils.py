import os


image_extension = ['.png', '.jpg']


def parse_image_folder_for_stereo_DVS(image_folder_path):
    file_list = os.listdir(image_folder_path)
    final_list = []
    for file in file_list:
        if file[-4:] in image_extension:
            final_list.append(file)
    final_list.sort()
    return final_list

def parse_folder_for_stereo_DVS(image_folder_path):
    file_list = os.listdir(image_folder_path)
    final_list = []
    for file in file_list:
        final_list.append(file)
    final_list.sort()
    return final_list

