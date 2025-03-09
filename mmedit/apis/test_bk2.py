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
# def PSNR(input1, input2):
#     mse = torch.mean((input1 - input2) ** 2)
#     psnr = 20 * torch.log10(1 / torch.sqrt(mse))
#     return psnr.item()
# with open('/home/v-huiminzeng/MLMM/video/imagenet_class.txt','r') as class_file:
#     class_all=class_file.readlines()
#     class_file.close()

def calc_psnr(img1, img2, data_range=255):
    '''
    img1 and img2 are arrays with same shape
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean(np.square(img1 - img2))
    if mse > 1e-10:
        psnr = 10 * np.log10(data_range * data_range / mse)
    else:
        psnr = 999.9
    return psnr

def PSNR(input1, input2):
    yuv_rec = input1.squeeze(0).cpu().numpy()
    y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
    y_rec = y_rec[0, :, :]
    u_rec = uv_rec[0, :, :]
    v_rec = uv_rec[1, :, :]

    yuv = input2.squeeze(0).cpu().numpy()
    y, uv = ycbcr444_to_420(yuv)
    y = y[0, :, :]
    u = uv[0, :, :]
    v = uv[1, :, :]


    psnr_y = calc_psnr(y, y_rec, data_range=1)
    psnr_u = calc_psnr(u, u_rec, data_range=1)
    psnr_v = calc_psnr(v, v_rec, data_range=1)
    psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
    return psnr,y_rec,uv_rec

def single_gpu_test(model,
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


def multi_gpu_test(model,data_loader,tmpdir=None,
                   gpu_collect=False,save_image=False,
                   save_path=None,empty_cache=False, 
                   bin_folder=None, write_stream=None,q_all=None,rate_idx=None,cal_ssim=False):
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
        lq, gt =data['lq'],data['gt']
        pic_height, pic_width = lq.shape[2], lq.shape[3]
        # pad if necessary
        breakpoint()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, 16)
        lq_padded = F.pad(lq,(padding_l, padding_r, padding_t, padding_b),mode="replicate",)
        gt_padded = F.pad(gt,(padding_l, padding_r, padding_t, padding_b),mode="replicate",)

        with torch.no_grad():
            result = model(lq=lq_padded, gt=gt_padded, q_index=q_all[rate_idx], test_mode=True)
        # class_img =  int(model.module.labels.cpu().numpy())
        

        x_hat = F.pad(result["x_hat"], (-padding_l, -padding_r, -padding_t, -padding_b))#yuv444
        psnr,y_rec,uv_rec = PSNR(x_hat, gt_padded.to(x_hat.device))
        if cal_ssim:
            msssim = ms_ssim(x_hat, gt_padded.to(x_hat.device), data_range=1).item()
        else:
            msssim = 0.
 
        psnr_all.append(psnr)
        # ssim_all.append(float(result["ssim"].cpu()))
        ssim_all.append(0)
        msssim_all.append(msssim)
        bit_all.append(float(result["bit"].cpu()))
        pix_all.append(pic_height*pic_width)
        bpp=float(result["bpp"].cpu())


    
        # # save image
        # if save_image:
        #     meta=(data["meta"].data)[0] 
        #     save_path_dir=os.path.basename(meta[0]['gt_path']).split(".")[0]
        #     img_name = f"{rate_idx}_{bpp:.4f}_{psnr:.4f}"
        #     if cal_ssim:
        #         img_name+=f"_{msssim:.4f}"
 
        #     save_path_img=os.path.join(save_path,save_path_dir,img_name+'.png')
        #     # save_path_img=os.path.join(save_path,save_path_dir,img_name+f'_{str(class_img)}'+'.png')
            
        #     # x_hat_rgb=ycbcr2rgb(x_hat)
        #     # mmcv.imwrite(tensor2img(x_hat_rgb), save_path_img)

        #     x_hat_rgb = ycbcr420_to_rgb(y_rec,uv_rec, order=1).transpose(1, 2, 0)
        #     x_hat_rgb = np.clip(np.rint(x_hat_rgb * 255), 0, 255).astype(np.uint8)
        #     x_hat_rgb = x_hat_rgb[:,:,[2,1,0]]
        #     mmcv.imwrite(x_hat_rgb, save_path_img)

        # save image
        if save_image:
            meta=(data["meta"].data)[0] 
            save_path_dir,img_name =meta[0]['gt_path'].split("/")[-2],meta[0]['gt_path'].split("/")[-1]
            img_name = f"{img_name.split('.')[0]}_{rate_idx}_{bpp:.4f}_{psnr:.4f}"
            if cal_ssim:
                img_name+=f"_{msssim:.4f}"
 
            save_path_img=os.path.join(save_path,save_path_dir,img_name+'.png')
            # save_path_img=os.path.join(save_path,save_path_dir,img_name+f'_{str(class_img)}'+'.png')
            
            # x_hat_rgb=ycbcr2rgb(x_hat)
            # mmcv.imwrite(tensor2img(x_hat_rgb), save_path_img)

            x_hat_rgb = ycbcr420_to_rgb(y_rec,uv_rec, order=1).transpose(1, 2, 0)
            x_hat_rgb = np.clip(np.rint(x_hat_rgb * 255), 0, 255).astype(np.uint8)
            x_hat_rgb = x_hat_rgb[:,:,[2,1,0]]
            mmcv.imwrite(x_hat_rgb, save_path_img)
 
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
    results["ssim"]=np.mean(ssim_all)
    results["msssim"]=np.mean(msssim_all)
    results["bpp"]=np.sum(bit_all)/np.sum(pix_all)
 
    # collect results from all ranks
    # if gpu_collect:
    #     results = collect_results_gpu(results, len(dataset))
    # else:
    #     results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results in cpu mode.

    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None

    Returns:
        list: Ordered results.
    """

    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # synchronizes all processes to make sure tmpdir exist
    dist.barrier()
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    # synchronizes all processes for loading pickle file
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None

    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results in gpu mode.

    It encodes results to gpu tensors and use gpu communication for results
    collection.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.

    Returns:
        list: Ordered results.
    """

    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank != 0:
        return None

    part_list = []
    for recv, shape in zip(part_recv_list, shape_list):
        part_list.append(pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    return ordered_results

 