# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmedit.utils import str2bool
from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
from mmedit.utils import setup_multi_processes
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--is_rt", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--ec_thread", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--force_zero_thres', type=float, default=None, required=False)
    parser.add_argument("--stream_part_i", type=int, default=1)
    parser.add_argument("--stream_part_p", type=int, default=1)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--q_indexes_i', type=int, nargs="+")
    parser.add_argument('--q_indexes_p', type=int, nargs="+")
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    parser.add_argument('--reset_interval', type=int, default=32, required=False)
    parser.add_argument('--test_config', type=str, default="./test_cfg/zhm_rgb.json",required=False)
    parser.add_argument('--yuv420', type=str2bool, default=False, required=False)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument('--float16', type=str2bool, default=False)
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--cuda_idx', type=int, nargs="+", help='GPU indexes to use')
    parser.add_argument('--cal_msssim', type=str2bool, default=False, required=False)
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--stream_path', type=str, default="./out_bin")
    parser.add_argument('--json_path', type=str, default=None)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--lq_folder', type=str, default=None)
    parser.add_argument('--gt_folder', type=str, default=None)
    parser.add_argument('--noise_type', type=int, default=None)
    parser.add_argument('--specified_key', type=str, default=None)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)
    # used by bitrate matching
    parser.add_argument('--match_br_final_recon_path', type=str, default='decoded_frames')
    parser.add_argument('--match_br_result_recon_path', type=str, default='match_bitrate')
    parser.add_argument('--match_br_encode_status_path', type=str, default='encode_status')
    parser.add_argument('--copy_src_file', type=str2bool, default=False, required=False)
    parser.add_argument('--with_noise',action='store_true',help='test images instead of yuv videos')
    parser.add_argument('--with_skip',action='store_true',help='test images instead of yuv videos')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    if args.test_dataset is not None:
        cfg.data.test.gt_folder=args.test_dataset
    if args.lq_folder is not None:
        cfg.data.test.lq_folder=args.lq_folder
    if args.gt_folder is not None:
        cfg.data.test.gt_folder=args.gt_folder
    if args.specified_key is not None:
        cfg.data.test.specified_key=[str(args.specified_key)]
    if args.noise_type is not None:
        cfg.data.test.pipeline[4].degrade_type=args.noise_type
        
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    i_frame_q_scale_enc, i_frame_q_scale_dec = model.generator.get_q_scales_from_ckpt(args.checkpoint)
    get_qp_num = model.generator.get_qp_num()

    args.save_image = args.save_path is not None
    empty_cache = cfg.get('empty_cache', False)
    if not distributed:
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model,
            data_loader,
            save_path=args.save_path,
            save_image=args.save_image)
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

        device_id = torch.cuda.current_device()
        _ = load_checkpoint(model, args.checkpoint,map_location=lambda storage, loc: storage.cuda(device_id))

        rate_num = args.rate_num
        
        print("q_scale_enc in intra ckpt: ", end='')
        for q in i_frame_q_scale_enc:
            print(f"{q:.3f}, ", end='')
        print()
        print("q_scale_dec in intra ckpt: ", end='')
        for q in i_frame_q_scale_dec:
            print(f"{q:.3f}, ", end='')
        print()
        q_indexes_i = []
        if args.q_indexes_i is not None:
            assert len(args.q_indexes_i) == rate_num
            q_indexes_i = args.q_indexes_i
        else:
            assert rate_num >= 2 and rate_num <= get_qp_num
            for i in np.linspace(0, get_qp_num - 1, num=rate_num):
                q_indexes_i.append(int(i+0.5))

        print(f"testing {rate_num} rates, using q_indexes: ", end='')
        for q in q_indexes_i:
            print(f"{q}, ", end='')
        print()

        results_all={}

        for rate_idx in range(rate_num):
            bin_folder = os.path.join(args.stream_path, str(rate_idx))
 
            outputs = multi_gpu_test(model,data_loader,args.tmpdir,
                args.gpu_collect,save_image=args.save_image,
                save_path=args.save_path,empty_cache=empty_cache,
                bin_folder=bin_folder, write_stream=args.write_stream,q_all=q_indexes_i,rate_idx=rate_idx,
                cal_ssim=args.cal_msssim
                )
 
            results_all[str(rate_idx)]=outputs
    
    if rank == 0:
        print("\n")
        for idx in results_all.keys():
            print(f"bpp:{results_all[idx]['bpp']:.4f} / psnr:{results_all[idx]['psnr']:.4f} / ssim:{results_all[idx]['ssim']:.4f} / msssim:{results_all[idx]['msssim']:.4f}")
     
    # TODO: here: dump json file from outputs --json_path
    if not args.json_path is None:
        results_dir =os.path.abspath(os.path.dirname(args.json_path))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with open(args.json_path,'w') as json_file:
            for idx in results_all.keys():
                json_file.write(f"bpp:{results_all[idx]['bpp']:.4f} / psnr:{results_all[idx]['psnr']:.4f} / ssim:{results_all[idx]['ssim']:.4f} / msssim:{results_all[idx]['msssim']:.4f}\n")
            json_file.close()
 
    # if rank == 0 and 'eval_result' in outputs[0]:
    #     print('')
    #     # print metrics
    #     stats = dataset.evaluate(outputs)
    #     for stat in stats:
    #         print('Eval-{}: {}'.format(stat, stats[stat]))

    #     # save result pickle
    #     if args.out:
    #         print('writing results to {}'.format(args.out))
    #         mmcv.dump(outputs, args.out)


if __name__ == '__main__':
    main()
