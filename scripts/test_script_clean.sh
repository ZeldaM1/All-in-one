#!/usr/bin/env bash

 



./tools/dist_test.sh  configs/noise/all_in_one_L.py  ckpt/all_in_one_L_noise.pth  1      \
"  --lq_folder data/all-in-one-test  --gt_folder data/all-in-one-test  \
--rate_num  8    --cal_msssim True  --noise_type 0  \
--save-path ./results/all_in_one_L/Kodak_24_sgm0  \
--json_path  ./results/all_in_one_L/Kodak_24_sgm0.json  "



 
./tools/dist_test.sh  configs/noise/all_in_one_S.py  ckpt/all_in_one_S_noise.pth  1     \
"  --lq_folder data/all-in-one-test  --gt_folder data/all-in-one-test  \
--rate_num  8    --cal_msssim True  --noise_type 0  \
--save-path ./results/all_in_one_S/Kodak_24_sgm0  \
--json_path  ./results/all_in_one_S/Kodak_24_sgm0.json  "

 