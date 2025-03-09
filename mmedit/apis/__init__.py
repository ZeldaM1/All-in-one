# Copyright (c) OpenMMLab. All rights reserved.
from .generation_inference import generation_inference
from .inpainting_inference import inpainting_inference
from .matting_inference import init_model, matting_inference
from .restoration_face_inference import restoration_face_inference
from .restoration_inference import restoration_inference
from .restoration_video_inference import restoration_video_inference
from .test import multi_gpu_test, single_gpu_test
from .test_stage1 import multi_gpu_test_s1, single_gpu_test_s1
from .train_epoch import init_random_seed, set_random_seed, train_model
from .train_iter import train_model_iter
from .train_s1 import train_model_s1
from .video_interpolation_inference import video_interpolation_inference

__all__ = [
    'train_model', 'set_random_seed', 'init_model', 'matting_inference',
    'inpainting_inference', 'restoration_inference', 'generation_inference',
    'multi_gpu_test', 'single_gpu_test', 'restoration_video_inference',
    'restoration_face_inference', 'video_interpolation_inference',
    'init_random_seed','train_model_iter','multi_gpu_test_s1', 'single_gpu_test_s1',
    'train_model_s1'
]
