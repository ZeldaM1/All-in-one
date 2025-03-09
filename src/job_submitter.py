# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import zipfile
from multiprocessing import Pool
from shutil import copyfile, disk_usage
import time
 
DMCI_version = "DMCI-3.4"
DMC_RT_version = "DMC-RT-3.0"

def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")




def get_args(argv):
    print(argv)
    working_folder = argv[1]
    dataset_folder = argv[2]
    experiment_name = argv[3]
    gpu_num = int(argv[4])
    print(f"working_folder {working_folder}")
    print(f"dataset_folder {dataset_folder}")
    print(f"experiment_name {experiment_name}")
    print(f"gpu_num {gpu_num}")
    return working_folder, dataset_folder, experiment_name, gpu_num


def install_dependency():
    os.system('pwd')
    os.system('ls')
    os.system('python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1  -f https://download.pytorch.org/whl/torch_stable.html   --user')
    os.system('python -m pip install -r requirements.txt --user')
    os.system('python -m pip install /output/v-huiminzeng/opencv_python-4.2.0.34-cp38-cp38-manylinux1_x86_64.whl ')
    os.system('python -m pip install /output/v-huiminzeng/opencv_python_headless-4.2.0.34-cp38-cp38-manylinux1_x86_64.whl ')
    os.system('python -m pip install /output/v-huiminzeng/mmcv_full-1.6.0-cp38-cp38-manylinux1_x86_64.whl ')
    # os.system('python -m pip  --timeout=5000  install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html ')
    os.system('python -m pip install mmengine pytorch_msssim einops scipy timm OmegaConf')  
    os.system('nvidia-smi')
 
def unzip_dataset(src_folder, dst_folder, is_dir=True):
    
    if is_dir:
        create_folder(dst_folder)
        print(f"unzipping from {src_folder} to {dst_folder}")
        for f in os.listdir(src_folder):
            if f.endswith('.json'):
                src_path = os.path.join(src_folder, f)
                dst_path = os.path.join(dst_folder, "description.json")
                copyfile(src_path, dst_path)

            if not f.endswith('.zip'):
                continue
            src_path = os.path.join(src_folder, f)
            with zipfile.ZipFile(src_path, 'r') as zip_ref:
                zip_ref.extractall(dst_folder) 
            print(f"{time.ctime()} extracted {f}")
    else:# is file .zip .json
        if src_folder.endswith('.json'):
            dst_folder=dst_folder+f"{os.path.basename(src_folder)}"
            print(f"copy from {src_folder} to {dst_folder}")
            copyfile(src_folder, dst_folder)
        elif src_folder.endswith('.zip'):
            print(f"unzipping from {src_folder} to {dst_folder}")
            with zipfile.ZipFile(src_folder, 'r') as zip_ref:
                zip_ref.extractall(dst_folder)
            print(f"{time.ctime()} extracted {src_folder}")
        else:
            assert NotImplementedError


def get_dataset_dst_folder():
    _, _, free = disk_usage('/dev/shm')
    if free > 100 * 1000 * 1000 * 1000:
        return '/dev/shm/'
    _, _, free = disk_usage('/dev/temp_data')
    if free > 100 * 1000 * 1000 * 1000:
        return '/dev/temp_data/'
    assert False


# def upload_dataset(dataset_folder, dataset_name, is_video, unzip_file=True):
#     sub_folder = 'video' if is_video else 'image'
#     dataset_root_folder = f'{dataset_folder}/{sub_folder}/'
#     train_dataset_src_folder = f'{dataset_root_folder}/{dataset_name}'
#     train_dataset_dst_folder = get_dataset_dst_folder() + 'dataset/train/'
#     train_dataset_path = train_dataset_dst_folder
#     if is_video:
#         train_dataset_path += f'/{dataset_name}/'

#     if not unzip_file:
#         return train_dataset_path

#     os.system('df')
#     unzip_dataset(train_dataset_src_folder, train_dataset_dst_folder)    

#     return train_dataset_path

def upload_dataset(dataset_root_folder, dataset_name, is_video=False, unzip_file=True):
    
    for dataset in dataset_name:
        train_dataset_src_folder = f'{dataset_root_folder}/{dataset}'
         
        if not unzip_file:
            return dataset_root_folder
        new_root_dst_folder=get_dataset_dst_folder()
        os.system('df')
        if dataset=='all-in-one-test.zip':
            test_dataset_dst_folder = new_root_dst_folder + f'dataset/'  
            unzip_dataset(train_dataset_src_folder, test_dataset_dst_folder, is_dir=False)    
        elif dataset=='OTS_outdoor':
            train_dataset_dst_folder = new_root_dst_folder + f'dataset/train/{dataset.split(".")[0]}/'  
            unzip_dataset(train_dataset_src_folder, train_dataset_dst_folder,is_dir=True)    
        else:#json or zip
            train_dataset_dst_folder =new_root_dst_folder+ 'dataset/train/'  
            unzip_dataset(train_dataset_src_folder, train_dataset_dst_folder,is_dir=False)    

    return f"{new_root_dst_folder}dataset/train", f"{new_root_dst_folder}dataset/all-in-one-test"


def upload_video_test_set(dataset_folder, rgb_file=False):
    dst_folder = '/dev/shm/video_test_yuv'
    if rgb_file:
        src_file = f'{dataset_folder}/benchmark/data/valid_video_bt709_96f.zip'
        test_folder = dst_folder + '/valid_video_bt709_96f'
    else:
        src_file = f'{dataset_folder}/benchmark/data/YUV_96f.zip'
        test_folder = dst_folder + '/YUV_96f'
    create_folder(dst_folder)

    with zipfile.ZipFile(src_file, 'r') as zip_ref:
        zip_ref.extractall(dst_folder)
    print(f"{time.ctime()} extracted {src_file}")
    return test_folder


def get_pretrained_weights(dataset_folder, is_rt=False, rgb=False, ssim=False):
    image_model_type = ""
    if rgb:
        image_model_type = '-rgb'
    if ssim:
        image_model_type = '-ssim'
    image_model_folder = f'{dataset_folder}/trained_image_models/{DMCI_version}/'
    image_model = f'{image_model_folder}/{DMCI_version}{image_model_type}.pth.tar'
    me_net_path = f"{dataset_folder}/trained_me/1121_ME_t03/t5/ckpt_epo99.pth.tar"

    if is_rt:
        image_model_folder = f'{dataset_folder}/trained_image_models/{DMCI_RT_version}/'
        image_model = f'{image_model_folder}/{DMCI_RT_version}{image_model_type}.pth.tar'
        me_net_path = f"{dataset_folder}/trained_me/0406_ME_t01/t5/ckpt_epo99.pth.tar"

    return image_model, me_net_path


def worker(input_command):
    print(input_command)
    os.system(input_command)


def submit_commands(commands):
    os.system("cp ./ /dev/temp_data/code/ -r")
    # os.chdir("/dev/temp_data/code/src/models/extensions/")
    # os.system("python setup.py build_ext --inplace")
    os.chdir("/dev/temp_data/code/")
    with Pool(len(commands)) as p:
        p.map(worker, commands)
