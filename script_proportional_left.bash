#!/bin/bash
for n_simulations in 10 20 50 100
do
for max_time in 10 50 100 200 500 1000 5000
do
python main_proportional_left.py --config=./configs/proportional/config_1.json --n_simulations=$n_simulations --max_time=$max_time --seed_offset=0
done
done