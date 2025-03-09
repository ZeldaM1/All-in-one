# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import torch.nn.functional as F
import os
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmedit.utils import get_padding_size,ycbcr2rgb
from mmedit.core import tensor2img
from pytorch_msssim import ms_ssim
import numpy as np
from mmedit.utils.functional import ycbcr444_to_420,ycbcr420_to_rgb
from .test import PSNR,calc_psnr,collect_results_cpu,collect_results_gpu
import time
def single_gpu_test_s1(model,
                    data_loader,
                    save_image=False,
                    save_path=None,
                    iteration=None):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        save_image (bool): Whether save image. Default: False.
        save_path (str): The path to save image. Default: None.
        iteration (int): Iteration number. It is used for the save image name.
            Default: None.

    Returns:
        list: The prediction results.
    """
    if save_image and save_path is None:
        raise ValueError(
            "When 'save_image' is True, you should also set 'save_path'.")

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(
                test_mode=True,
                save_image=save_image,
                save_path=save_path,
                iteration=iteration,
                **data)
        results.append(result)

        # get batch size
        for _, v in data.items():
            if isinstance(v, torch.Tensor):
                batch_size = v.size(0)
                break
        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test_s1(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   save_image=False,
                   save_path=None,
                   iteration=None,
                   empty_cache=False):
    if save_image and save_path is None:
        raise ValueError(
            "When 'save_image' is True, you should also set 'save_path'.")
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time_cuda=[]
    for data in data_loader:
        torch.cuda.synchronize()
        t1=time.time()
        with torch.no_grad():
            result = model(
                test_mode=True,
                save_image=save_image,
                save_path=save_path,
                iteration=iteration,
                **data)
        torch.cuda.synchronize()
        t2=time.time()
        print('----------',t2-t1)
        time_cuda.append((t2-t1))
        results.append(result)
        if empty_cache:
            torch.cuda.empty_cache()
        if rank == 0:
            # get batch size
            for _, v in data.items():
                if isinstance(v, torch.Tensor):
                    batch_size = v.size(0)
                    break
            for _ in range(batch_size * world_size):
                prog_bar.update()
    print('############', np.array(time_cuda).mean())
    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def multi_gpu_test_s1_yuv(model,data_loader,tmpdir=None,
                   gpu_collect=False,save_image=False,
                   save_path=None,empty_cache=False, 
                   bin_folder=None, write_stream=None, cal_ssim=False):
    if save_image and save_path is None:
        raise ValueError(
            "When 'save_image' is True, you should also set 'save_path'.")
    model.eval()
    results = {}
    psnr_all, ssim_all, msssim_all,bit_all,pix_all = [],[],[],[],[]
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for frame_idx, data in enumerate(data_loader):
        gt=data['gt']
        pic_height, pic_width = gt.shape[2], gt.shape[3]
        # pad if necessary
        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, 16)
        gt_padded = F.pad(gt,(padding_l, padding_r, padding_t, padding_b),mode="replicate",)
        bin_path = os.path.join(bin_folder, f"{frame_idx}.bin") if write_stream else None

        with torch.no_grad():
            result = model(gt_padded, test_mode=True)
 
        x_hat = F.pad(result, (-padding_l, -padding_r, -padding_t, -padding_b))#yuv444
        # psnr,y_rec,uv_rec = PSNR(x_hat, gt.to(x_hat.device))
        # breakpoint()
        # psnr= calc_psnr(np.clip(x_hat.cpu().numpy()[0],0,1), gt.numpy()[0], data_range=1)
        psnr= 0
 
        if cal_ssim:
            msssim = ms_ssim(x_hat, gt.to(x_hat.device), data_range=1).item()
        else:
            msssim = 0.
 
        psnr_all.append(psnr)

        # save image
        if save_image:
            meta=(data["meta"].data)[0] 
            save_path_dir=os.path.basename(meta[0]['gt_path']).split(".")[0]
            img_name = f"{psnr:.4f}"
            if cal_ssim:
                img_name+=f"_{msssim:.4f}"
 
            save_path_img=os.path.join(save_path,save_path_dir,img_name+'.png')
            # save_path_img=os.path.join(save_path,save_path_dir,img_name+f'_{str(class_img)}'+'.png')
            
            # x_hat_rgb=ycbcr2rgb(x_hat)
            # mmcv.imwrite(tensor2img(x_hat_rgb), save_path_img)

            # x_hat_rgb = ycbcr420_to_rgb(y_rec,uv_rec, order=1).transpose(1, 2, 0)
            # x_hat_rgb = np.clip(np.rint(x_hat_rgb * 255), 0, 255).astype(np.uint8)
            # x_hat_rgb = x_hat_rgb[:,:,[2,1,0]]
            # mmcv.imwrite(x_hat_rgb, save_path_img)
       

            x_hat_rgb = x_hat.cpu().numpy()[0].transpose(1, 2, 0)
            x_hat_rgb = np.clip(np.rint(x_hat_rgb * 255), 0, 255).astype(np.uint8)
            x_hat_rgb = x_hat_rgb[:,:,[2,1,0]]
            mmcv.imwrite(x_hat_rgb, save_path_img)# save bgr
            
 
        if empty_cache:
            torch.cuda.empty_cache()
        if rank == 0:
            # get batch size
            for _, v in data.items():
                if isinstance(v, torch.Tensor):
                    batch_size = v.size(0)
                    break
            for _ in range(batch_size * world_size):
                prog_bar.update()
    # breakpoint()
    results["psnr"]=np.mean(psnr_all)

 
    # collect results from all ranks
    # if gpu_collect:
    #     results = collect_results_gpu(results, len(dataset))
    # else:
    #     results = collect_results_cpu(results, len(dataset), tmpdir)
    print('----------------',results["psnr"])
    return results
