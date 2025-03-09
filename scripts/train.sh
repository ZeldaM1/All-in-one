#!/usr/bin/env bash

./tools/dist_train_iter.sh   configs/weather/all_in_one_L.py 4  '--train_dataset data/all-in-one-train --test_dataset data/all-in-one-test --exp_name all_in_one_L_weather ' 

./tools/dist_train_iter.sh   configs/weather/all_in_one_S.py 4  '--train_dataset data/all-in-one-train --test_dataset data/all-in-one-test --exp_name all_in_one_S_weather ' 


./tools/dist_train_iter.sh   configs/noise/all_in_one_L.py 4  '--train_dataset data/all-in-one-train --test_dataset data/all-in-one-test --exp_name all_in_one_L_noise ' 

./tools/dist_train_iter.sh   configs/noise/all_in_one_S.py 4  '--train_dataset data/all-in-one-train --test_dataset data/all-in-one-test --exp_name all_in_one_S_noise ' 

  