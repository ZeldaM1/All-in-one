# Copyright (c) OpenMMLab. All rights reserved.
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
import os
import json
import copy
import random

@DATASETS.register_module()
class SRFolderGTDataset(BaseSRDataset):
    """General ground-truth image folder dataset for image restoration.

    The dataset loads gt (Ground-Truth) image only,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "gt folder mode", which needs to specify the gt
    folder path, each folder containing the corresponding images.
    Image lists will be generated automatically.

    For example, we have a folder with the following structure:

    ::

        data_root
        ├── gt
        │   ├── 0001.png
        │   ├── 0002.png

    then, you need to set:

    .. code-block:: python

        gt_folder = data_root/gt

    Args:
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (List[dict | callable]): A sequence of data transformations.
        scale (int | tuple): Upsampling scale or upsampling scale range.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 gt_folder,
                 pipeline,
                 scale,
                 test_mode=False,
                 filename_tmpl='{}'):
        super().__init__(pipeline, scale, test_mode)
        self.gt_folder = str(gt_folder)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for SR dataset.

        It loads the GT image path from folder.

        Returns:
            list[dict]: A list of dicts for path of GT.
        """
        data_infos = []
        gt_paths = self.scan_folder(self.gt_folder)
        for gt_path in gt_paths:
            data_infos.append(dict(gt_path=gt_path))
        return data_infos


@DATASETS.register_module()
class GTListDataset(SRFolderGTDataset):
    def load_annotations(self):
        with open(os.path.join(self.gt_folder, 'description.json')) as json_file:
            gt_paths = json.load(json_file)

        data_infos = []
        for gt_path in gt_paths:
            gt_path_full= os.path.join(self.gt_folder, gt_path)
            data_infos.append(dict(gt_path=gt_path_full))
        return data_infos

 
@DATASETS.register_module()
class AllinOneDataset(BaseSRDataset):
    def __init__(self, gt_folder, pipeline, scale, return_param="gt_path",
                 test_mode=False, dataset_json='all_in_one.json',
                 specified_key=None,shuffle=False):
        super().__init__(pipeline, scale, test_mode)
        self.specified_key=specified_key
        self.gt_folder = str(gt_folder)
        self.return_param=return_param
        self.shuffle=shuffle
        self.dataset_json=dataset_json
        self.data_infos = self.load_annotations()
 
    def load_annotations(self):
        with open(os.path.join(self.gt_folder, self.dataset_json)) as json_file:
            lq_gt_paths = json.load(json_file)
            json_file.close()
        data_infos = []
 
        path_keys=lq_gt_paths.keys() if (self.specified_key is None) else self.specified_key
        for key in path_keys:
            for lq_gt_path in lq_gt_paths[key]:
                lq_path_full= os.path.join(self.gt_folder, lq_gt_path[0])
                gt_path_full= os.path.join(self.gt_folder, lq_gt_path[1])
                if self.return_param=="gt_path":
                    data_infos.append(dict(gt_path=gt_path_full))  
                elif self.return_param=="lq_path":
                    data_infos.append(dict(gt_path=lq_path_full))  
                elif self.return_param=="all":
                    data_infos.append(dict(lq_path=lq_path_full, gt_path=gt_path_full))
                else:
                    assert NotImplementedError
        if self.shuffle:
            random.shuffle(data_infos)
        return data_infos
 
@DATASETS.register_module()
class AllinOneDataset_contrast(AllinOneDataset):
    def load_annotations(self):
        with open(os.path.join(self.gt_folder, self.dataset_json)) as json_file:
            lq_gt_paths = json.load(json_file)
            json_file.close()
        data_infos = []

        path_keys=lq_gt_paths.keys() 
        list_all=[key for key in path_keys]
        for key in path_keys:
            for lq_gt_path in lq_gt_paths[key]:
                lq_path_full= os.path.join(self.gt_folder, lq_gt_path[0])
                gt_path_full= os.path.join(self.gt_folder, lq_gt_path[1])
                other_types=key
                while other_types==key:
                    other_types = random.choice(list_all)
                other_type_path=random.choice(lq_gt_paths[other_types])[0]
                lq2_path_full= os.path.join(self.gt_folder, other_type_path)
                data_infos.append(dict(lq_path=lq_path_full, lq2_path=lq2_path_full, gt_path=gt_path_full))
                
        if self.shuffle:
            random.shuffle(data_infos)
        return data_infos
 