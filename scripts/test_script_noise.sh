#!/usr/bin/env bash

 
# ./tools/dist_test.sh  configs/noise/all_in_one_L.py  ckpt/all_in_one_L_noise.pth  1      \
# "  --lq_folder data/all-in-one-test  --gt_folder data/all-in-one-test  \
# --rate_num  8     --cal_msssim True  --noise_type 15  \
# --save-path ./results/all_in_one_L/Kodak_24_sgm15  \
# --json_path  ./results/all_in_one_L/Kodak_24_sgm15.json  "

 
 
 
# ./tools/dist_test.sh  configs/noise/all_in_one_S.py  ckpt/all_in_one_S_noise.pth  1     \
# "  --lq_folder data/all-in-one-test  --gt_folder data/all-in-one-test  \
# --rate_num  8    --cal_msssim True   --noise_type 15  \
# --save-path ./results/all_in_one_S/Kodak_24_sgm15  \
# --json_path  ./results/all_in_one_S/Kodak_24_sgm15.json  "


pip install pytorch_msssim scipy mmengine timm==0.6.7
cd /data/zenghuimin/code/codec-dev-release/
 
./tools/dist_test.sh  configs/noise/all_in_one_S.py   all_in_one_S_noise.pth  1     \
"  --lq_folder data/all-in-one-test  --gt_folder data/all-in-one-test  \
--rate_num  8    --cal_msssim True   --noise_type 15  \
--save-path ./results/all_in_one_S/Kodak_24_sgm15  \
--json_path  ./results/all_in_one_S/Kodak_24_sgm15.json  "

