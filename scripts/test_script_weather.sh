#!/usr/bin/env bash

./tools/dist_test.sh configs/weather/all_in_one_L.py  \
  ./ckpt/all_in_one_L_weather.pth  1   \
  " --specified_key SOTS_outdoor   --gt_folder  data/all-in-one-test  \
  --save-path ./results/all_in_one_L/SOTS_outdoor  \
  --json_path  ./results/all_in_one_L/SOTS_outdoor.json  \
  --rate_num  8  --cal_msssim True "



 
./tools/dist_test.sh configs/weather/all_in_one_S.py  \
  ./ckpt/all_in_one_S_weather.pth  1   \
  " --specified_key SOTS_outdoor   --gt_folder  data/all-in-one-test  \
  --save-path ./results/all_in_one_S/SOTS_outdoor  \
  --json_path  ./results/all_in_one_S/SOTS_outdoor.json  \
  --rate_num  8  --cal_msssim True "



 