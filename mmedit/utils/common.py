# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
import numpy as np

def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json(frame_num, frame_pixel_num, test_time, frame_types, bits, psnrs, ssims,
                      psnrs_y=None, psnrs_u=None, psnrs_v=None,
                      ssims_y=None, ssims_u=None, ssims_v=None, verbose=False):
    include_yuv = psnrs_y is not None
    if include_yuv:
        assert psnrs_u is not None
        assert psnrs_v is not None
        assert ssims_y is not None
        assert ssims_u is not None
        assert ssims_v is not None
    i_bits = 0
    i_psnr = 0
    i_psnr_y = 0
    i_psnr_u = 0
    i_psnr_v = 0
    i_ssim = 0
    i_ssim_y = 0
    i_ssim_u = 0
    i_ssim_v = 0
    p_bits = 0
    p_psnr = 0
    p_psnr_y = 0
    p_psnr_u = 0
    p_psnr_v = 0
    p_ssim = 0
    p_ssim_y = 0
    p_ssim_u = 0
    p_ssim_v = 0
    i_num = 0
    p_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            i_bits += bits[idx]
            i_psnr += psnrs[idx]
            i_ssim += ssims[idx]
            i_num += 1
            if include_yuv:
                i_psnr_y += psnrs_y[idx]
                i_psnr_u += psnrs_u[idx]
                i_psnr_v += psnrs_v[idx]
                i_ssim_y += ssims_y[idx]
                i_ssim_u += ssims_u[idx]
                i_ssim_v += ssims_v[idx]
        else:
            p_bits += bits[idx]
            p_psnr += psnrs[idx]
            p_ssim += ssims[idx]
            p_num += 1
            if include_yuv:
                p_psnr_y += psnrs_y[idx]
                p_psnr_u += psnrs_u[idx]
                p_psnr_v += psnrs_v[idx]
                p_ssim_y += ssims_y[idx]
                p_ssim_u += ssims_u[idx]
                p_ssim_v += ssims_v[idx]

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = i_num
    log_result['p_frame_num'] = p_num
    log_result['ave_i_frame_bpp'] = i_bits / i_num / frame_pixel_num
    log_result['ave_i_frame_psnr'] = i_psnr / i_num
    log_result['ave_i_frame_msssim'] = i_ssim / i_num
    if include_yuv:
        log_result['ave_i_frame_psnr_y'] = i_psnr_y / i_num
        log_result['ave_i_frame_psnr_u'] = i_psnr_u / i_num
        log_result['ave_i_frame_psnr_v'] = i_psnr_v / i_num
        log_result['ave_i_frame_msssim_y'] = i_ssim_y / i_num
        log_result['ave_i_frame_msssim_u'] = i_ssim_u / i_num
        log_result['ave_i_frame_msssim_v'] = i_ssim_v / i_num
    if verbose:
        log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
        log_result['frame_psnr'] = psnrs
        log_result['frame_msssim'] = ssims
        log_result['frame_type'] = frame_types
        if include_yuv:
            log_result['frame_psnr_y'] = psnrs_y
            log_result['frame_psnr_u'] = psnrs_u
            log_result['frame_psnr_v'] = psnrs_v
            log_result['frame_msssim_y'] = ssims_y
            log_result['frame_msssim_u'] = ssims_u
            log_result['frame_msssim_v'] = ssims_v
    log_result['test_time'] = test_time
    if p_num > 0:
        total_p_pixel_num = p_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = p_bits / total_p_pixel_num
        log_result['ave_p_frame_psnr'] = p_psnr / p_num
        log_result['ave_p_frame_msssim'] = p_ssim / p_num
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = p_psnr_y / p_num
            log_result['ave_p_frame_psnr_u'] = p_psnr_u / p_num
            log_result['ave_p_frame_psnr_v'] = p_psnr_v / p_num
            log_result['ave_p_frame_msssim_y'] = p_ssim_y / p_num
            log_result['ave_p_frame_msssim_u'] = p_ssim_u / p_num
            log_result['ave_p_frame_msssim_v'] = p_ssim_v / p_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_psnr'] = 0
        log_result['ave_p_frame_msssim'] = 0
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = 0
            log_result['ave_p_frame_psnr_u'] = 0
            log_result['ave_p_frame_psnr_v'] = 0
            log_result['ave_p_frame_msssim_y'] = 0
            log_result['ave_p_frame_msssim_u'] = 0
            log_result['ave_p_frame_msssim_v'] = 0
    log_result['ave_all_frame_bpp'] = (i_bits + p_bits) / (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (i_psnr + p_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (i_ssim + p_ssim) / frame_num
    if include_yuv:
        log_result['ave_all_frame_psnr_y'] = (i_psnr_y + p_psnr_y) / frame_num
        log_result['ave_all_frame_psnr_u'] = (i_psnr_u + p_psnr_u) / frame_num
        log_result['ave_all_frame_psnr_v'] = (i_psnr_v + p_psnr_v) / frame_num
        log_result['ave_all_frame_msssim_y'] = (i_ssim_y + p_ssim_y) / frame_num
        log_result['ave_all_frame_msssim_u'] = (i_ssim_u + p_ssim_u) / frame_num
        log_result['ave_all_frame_msssim_v'] = (i_ssim_v + p_ssim_v) / frame_num

    return log_result


# below code is only used in training
def get_latest_checkpoint_path(dir_cur, key_model=None):
    files = os.listdir(dir_cur)
    all_best_checkpoints = []
    if key_model is None:
        for file in files:
            if file[-4:] == '.tar' and 'ckpt' in file:
                all_best_checkpoints.append(os.path.join(dir_cur, file))
        if len(all_best_checkpoints) > 0:
            return max(all_best_checkpoints, key=os.path.getmtime)
    else:
        for file in files:
            if (file[-4:] == '.tar' and 'ckpt' in file) and (key_model in file):
                all_best_checkpoints.append(os.path.join(dir_cur, file))
        if len(all_best_checkpoints) > 0:
            return max(all_best_checkpoints, key=os.path.getmtime)

    return 'not_exist'


def get_latest_status_path(dir_cur):
    files = os.listdir(dir_cur)
    all_status_files = []
    for file in files:
        if 'status_epo' in file in file:
            all_status_files.append(os.path.join(dir_cur, file))
    all_status_files.sort(key=lambda x: os.path.getmtime(x))
    if len(all_status_files) > 2:
        return [all_status_files[-2], all_status_files[-1]]
    return all_status_files


def ddp_sync_state_dict(task_id, to_sync, rank, device, device_ids):
    # to_sync is model or optimizer, which supports state_dict() and load_state_dict()
    ckpt_path = os.path.join("/dev", "shm", f"train_{task_id}_tmp", "tmp.ckpt")
    if rank == 0:
        print(f"sync model with {ckpt_path}")
        torch.save(to_sync.state_dict(), ckpt_path)
    dist.barrier(device_ids=device_ids)
    to_sync.load_state_dict(torch.load(ckpt_path, map_location=device))
    dist.barrier(device_ids=device_ids)
    if rank == 0:
        os.remove(ckpt_path)
    dist.barrier(device_ids=device_ids)


def avg_per_rate(x, B, anchor_num):
    y = x.reshape((anchor_num, B))
    return torch.sum(y, dim=1) / B


def generate_str(x):
    # print(x)
    s = ''
    for a in x:
        s += f'{a.item():.5f} '
    return s


def remove_nan_grad(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.nan_to_num_(0.0, 0.0, 0.0)


def process_test_epoch(test_epochs, total_epochs):
    # -1 means testing all the remaining epochs
    if -1 in test_epochs:
        if len(test_epochs) > 1:
            last_test_epoch = test_epochs[-2]
        else:
            last_test_epoch = -1
        for i in range(1, total_epochs):
            test_epochs.append(last_test_epoch + i)
    return test_epochs


def start_train(cuda_idx, task_id, train_fun, args):
    if cuda_idx is not None:
        cuda_device = ','.join([str(s) for s in cuda_idx])
        print("task", task_id, "device", cuda_device)
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        os.environ['PYTORCH_KERNEL_CACHE_PATH'] = '/dev/shm/kernel_cache'
        os.makedirs('/dev/shm/kernel_cache', exist_ok=True)

    use_ddp = False
    if torch.cuda.device_count() > 0 and len(cuda_idx) > 1:
        use_ddp = True

    if not use_ddp:
        train_fun(-1, args)
    else:
        tmp_path = os.path.join("/dev", "shm", f"train_{task_id}_tmp")
        create_folder(tmp_path, True)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355 + task_id)
        world_size = len(cuda_idx)
        mp.spawn(train_fun, nprocs=world_size, args=(args,), join=True)


def save_ckpt(net, epoch, args, key_model=None):
    if key_model is not None:
        cur_checkpoint_name = f"{args.save_dir}/ckpt_epo{epoch}{key_model}.pth.tar"
        pre_checkpoint_name = f"{args.save_dir}/ckpt_epo{epoch-1}{key_model}.pth.tar"
    else:
        cur_checkpoint_name = f"{args.save_dir}/ckpt_epo{epoch}.pth.tar"
        pre_checkpoint_name = f"{args.save_dir}/ckpt_epo{epoch-1}.pth.tar"

    print(f"task {args.task_id} save model epoch {epoch}")
    save_dict = {
        "epoch": epoch,
        "state_dict": net.state_dict(),
    }
    torch.save(save_dict, cur_checkpoint_name)

def partial_save_ckpt(net, epoch, args, key_model=None):
    if key_model is not None:
        cur_checkpoint_name = f"{args.save_dir}/ckpt_epo{epoch}{key_model}.pth.tar"
        pre_checkpoint_name = f"{args.save_dir}/ckpt_epo{epoch-1}{key_model}.pth.tar"
    else:
        cur_checkpoint_name = f"{args.save_dir}/ckpt_epo{epoch}.pth.tar"
        pre_checkpoint_name = f"{args.save_dir}/ckpt_epo{epoch-1}.pth.tar"

    print(f"task {args.task_id} save model epoch {epoch}")
    
    save_dict=net.state_dict()
    if hasattr(net,'dec'):
        if hasattr(net.dec,'dec_2'):
            for name in list(save_dict.keys()):
                if name.startswith('dec.dec_2'):
                    del save_dict[name]
  

    save_dict_all = {
        "epoch": epoch,
        "state_dict": save_dict,
    }
    torch.save(save_dict_all, cur_checkpoint_name)

    if (epoch - 1) % args.ckpt_per_epoch != 0 and (epoch - 1) not in args.benchmark_test_epoch:
        if os.path.exists(pre_checkpoint_name):
            os.remove(pre_checkpoint_name)
    return cur_checkpoint_name


def auto_test(rank, epoch, args, curr_ckpt, intra_only=False, writer=None, is_final=False,
              yuv420=False, ssim=False):
    need_test = False
    if args.benchmark_test_anchor is not None and epoch in args.benchmark_test_epoch:
        need_test = True
        torch.cuda.empty_cache()
        if rank >= 0 and not is_final:
            dist.barrier(device_ids=[rank])

    if rank <= 0 and need_test:
        cuda_idx = ' '.join([str(s) for s in args.cuda_idx])
        output_json_path = os.path.join(args.save_dir, f'bmk_test_epo_{epoch}.json')
        skip_test = False
        if os.path.exists(output_json_path):
            skip_test = True
        command_line = (" python test_video.py"
                        f" --is_rt {args.is_rt} --rate_num 4"
                        f" --test_config {args.benchmark_test_config}"
                        f" --force_root_path {args.benchmark_data_path} --yuv420 {yuv420}"
                        f" --calc_ssim {ssim}"
                        f" --cuda 1 --cuda_idx {cuda_idx} -w {len(args.cuda_idx)}"
                        f" --write_stream 0 --output_path {output_json_path}")
        if intra_only:
            command_line += (f" --model_path_i {curr_ckpt}"
                             f" --force_intra_period 1 --force_intra 1 --force_frame_num 1")
        else:
            model_path_i = args.model_path_i
            command_line += (f" --model_path_i {model_path_i}"
                             f" --model_path_p {curr_ckpt}"
                             f" --force_intra_period {args.benchmark_test_intra_period}")
        if args.with_noise:
            command_line += (f" --with_noise")
        if args.with_skip:
            command_line += (f" --with_skip")
            
        if not skip_test:
            print(command_line)
            os.system(command_line)

        anchor_json = args.benchmark_test_anchor
        if is_final:
            output_path = os.path.join(args.save_dir, 'bd_rate_final.txt')
        else:
            output_path = os.path.join(args.save_dir, 'bd_rate.txt')
        metrics = 'psnr'
        if ssim:
            metrics = 'msssim'
        command_line = ("python ./compare_rd_video.py --compare_between class"
                        " --base_method anchor"
                        f" --log_paths anchor {anchor_json}"
                        f" test_epo{epoch} {output_json_path}"
                        f" --output_path {output_path}"
                        " --auto_test 1"
                        " --plot_rd_curve 0 --plot_path ./test_room/figs/"
                        f" --distortion_metrics {metrics}")  # also supports msssim msssim_log
        print(command_line)
        os.system(command_line)

        dataset_num = 4 if yuv420 else 5
        bd_rates = get_average_bd_rates(output_path, dataset_num)
        if len(bd_rates) == 0:
            return
        epoch = bd_rates[-1][0]
        curr_bd_rate = bd_rates[-1][1]
        best_bd_rate = curr_bd_rate
        for bd_rate in bd_rates:
            if bd_rate[1] < best_bd_rate:
                best_bd_rate = bd_rate[1]
        if writer is not None and not is_final:
            write_bd_rate_to_tensorboard(epoch, curr_bd_rate, best_bd_rate, writer)
        if is_final:
            bd_rate_file_path = os.path.join(args.save_dir, 'best_bd_rate_final.txt')
        else:
            bd_rate_file_path = os.path.join(args.save_dir, 'best_bd_rate.txt')
        write_bd_rate_to_file(epoch, curr_bd_rate, best_bd_rate, bd_rate_file_path)


def get_current_device(rank, cuda_idx):
    print(f"cuda_is_available {torch.cuda.is_available()} device_count {torch.cuda.device_count()}")

    if rank >= 0:
        device = f"cuda:{rank}"
    elif cuda_idx is not None and len(cuda_idx) == 1 and torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"device_type: {device}")
    return device


def get_average_bd_rates(log_path, dataset_num=5):
    with open(log_path) as fp:
        lines = fp.readlines()
    results = []
    found = -1
    for line in lines:
        if found > 0:
            found -= 1
            continue
        if found == 0:
            found = -1
            words = line.split()
            if len(words) < dataset_num + 1:
                continue
            epoch = int(words[0][8:])
            bd_rates = 0
            for i in range(1, dataset_num + 1):
                bd_rate = float(words[i])
                bd_rates += bd_rate
            avg_bd_rate = bd_rates / dataset_num
            result = (epoch, avg_bd_rate)
            results.append(result)
        if "all frame" in line:
            found = 1
    return results


def write_bd_rate_to_tensorboard(epoch, curr_bd_rate, best_bd_rate, writer):
    writer.add_scalar("curr_bd_rate", curr_bd_rate, epoch)
    writer.add_scalar("best_bd_rate", best_bd_rate, epoch)


def write_bd_rate_to_file(epoch, curr_bd_rate, best_bd_rate, output_file_path):
    with open(output_file_path, "a") as f:
        line = (f"epoch: {epoch}, "
                f"curr_bd_rate: {curr_bd_rate:.2f}, "
                f"best_bd_rate: {best_bd_rate:.2f}\n")
        f.write(line)


def get_optimizer(parameters, print_info):
    try:
        import apex
        optimizer = apex.optimizers.FusedAdam(parameters, lr=1e-4,
                                              bias_correction=False, weight_decay=0.01)
        if print_info:
            print("use optimizer from apex.")
    except Exception:
        optimizer = optim.AdamW(parameters, lr=1e-4)
        if print_info:
            print("use optimizer from pytorch.")
    return optimizer


def get_clip_grad_norm_func(print_info):
    try:
        from apex.contrib.clip_grad import clip_grad_norm_
        func = clip_grad_norm_
        if print_info:
            print("use clip_grad_norm_ from apex.")
    except Exception:
        func = torch.nn.utils.clip_grad_norm_
        if print_info:
            print("use clip_grad_norm_ from pytorch.")
    return func

 


def get_training_lambdas(lmbdas, rate_num, qp_num, is_rand_rate, device):
    all_lmbdas = np.linspace(np.log(lmbdas[0]), np.log(lmbdas[1]), qp_num)
    all_lmbdas = np.exp(all_lmbdas)

    if is_rand_rate:
        return all_lmbdas, None

    idx0 = 0
    idx1 = int((qp_num - 1) * 1 / 3 + 0.5)
    idx2 = int((qp_num - 1) * 2 / 3 + 0.5)
    idx3 = qp_num - 1

    if rate_num == 4:
        training_lmbdas = [all_lmbdas[idx0], all_lmbdas[idx1], all_lmbdas[idx2], all_lmbdas[idx3]]
        training_lmbdas = [torch.tensor(training_lmbdas, dtype=torch.float32, device=device)]
        q_indexes = [[idx0, idx1, idx2, idx3]]
    elif rate_num == 2:
        training_lmbdas_0 = [all_lmbdas[idx0], all_lmbdas[idx1]]
        training_lmbdas_1 = [all_lmbdas[idx2], all_lmbdas[idx3]]
        training_lmbdas = [torch.tensor(training_lmbdas_0, dtype=torch.float32, device=device),
                           torch.tensor(training_lmbdas_1, dtype=torch.float32, device=device)]
        q_indexes = [[idx0, idx1], [idx2, idx3]]
    else:
        assert rate_num == 1
        training_lmbdas_0 = [all_lmbdas[idx0]]
        training_lmbdas_1 = [all_lmbdas[idx1]]
        training_lmbdas_2 = [all_lmbdas[idx2]]
        training_lmbdas_3 = [all_lmbdas[idx3]]
        training_lmbdas = [torch.tensor(training_lmbdas_0, dtype=torch.float32, device=device),
                           torch.tensor(training_lmbdas_1, dtype=torch.float32, device=device),
                           torch.tensor(training_lmbdas_2, dtype=torch.float32, device=device),
                           torch.tensor(training_lmbdas_3, dtype=torch.float32, device=device)]
        q_indexes = [[idx0], [idx1], [idx2], [idx3]]

    return training_lmbdas, q_indexes
