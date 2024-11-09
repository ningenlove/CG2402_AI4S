#!/bin/bash
mkdir -p 'user/'

################
# epochs = 20000, milestone = [5500, 10000, 14000, 17500], depth = 1, width = 30
# python3 -u main-work1.py  --image 'Images/Burgers/simulation_2/'  --result 'Result/Burgers/simulation_2/'  --checkpoint 'Checkpoints/Burgers/simulation_2/' > 'user/Exp2_log_2.txt'

# python3 -u main-work1.py  --image 'Images/Burgers/simulation_0/'  --result 'Result/Burgers/simulation_0/'  --checkpoint 'Checkpoints/Burgers/simulation_0/' > 'user/Exp2_log_0.txt'

# python3 -u main-work1.py  --image 'Images/Burgers/simulation_1/'  --result 'Result/Burgers/simulation_1/'  --checkpoint 'Checkpoints/Burgers/simulation_1/' > 'user/Exp2_log_1.txt'


##############################################
# epochs = 13000, milestone = [4000, 7500, 10000, 12000], depth = 1, width = 30
python3 -u main-work1.py  --image 'Images/Burgers/simulation_12/'  --result 'Result/Burgers/simulation_12/'  --checkpoint 'Checkpoints/Burgers/simulation_12/' > 'user/Exp2_log_12.txt'

python3 -u main-work1.py  --image 'Images/Burgers/simulation_10/'  --result 'Result/Burgers/simulation_10/'  --checkpoint 'Checkpoints/Burgers/simulation_10/' > 'user/Exp2_log_0.txt'

python3 -u main-work1.py  --image 'Images/Burgers/simulation_11/'  --result 'Result/Burgers/simulation_11/'  --checkpoint 'Checkpoints/Burgers/simulation_11/' > 'user/Exp2_log_1.txt'
