python tdmpc2/single_imitation_train.py task=walker-walk agent_class=reinforce_pred
# python tdmpc2/train.py task=walker-walk
# python tdmpc2/dialectic_train.py task=walker-walk

# $ python train.py task=mt80 model_size=48 batch_size=1024
# $ python train.py task=mt30 model_size=317 batch_size=1024
# $ python train.py task=dog-run steps=7000000
# $ python train.py task=walker-walk obs=rgb


# Task Observation dim Action dim Sparse? New?
# Acrobot Swingup 6 1 N N
# Cartpole Balance 5 1 N N
# Cartpole Balance Sparse 5 1 Y N
# Cartpole Swingup 5 1 N N
# Cartpole Swingup Sparse 5 1 Y N
# Cheetah Jump 17 6 N Y
# Cheetah Run 17 6 N N
# Cheetah Run Back 17 6 N Y
# Cheetah Run Backwards 17 6 N Y
# Cheetah Run Front 17 6 N Y
# Cup Catch 8 2 Y N
# Cup Spin 8 2 N Y
# Dog Run 223 38 N N
# Dog Trot 223 38 N N
# Dog Stand 223 38 N N
# Dog Walk 223 38 N N
# Finger Spin 9 2 Y N
# Finger Turn Easy 12 2 Y N
# Finger Turn Hard 12 2 Y N
# Fish Swim 24 5 N N
# Hopper Hop 15 4 N N
# Hopper Hop Backwards 15 4 N Y
# Hopper Stand 15 4 N N
# Humanoid Run 67 24 N N
# Humanoid Stand 67 24 N N
# Humanoid Walk 67 24 N N
# Pendulum Spin 3 1 N Y
# Pendulum Swingup 3 1 N N
# Quadruped Run 78 12 N N
# Quadruped Walk 78 12 N N
# Reacher Easy 6 2 Y N
# Reacher Hard 6 2 Y N
# Reacher Three Easy 8 3 Y Y
# Reacher Three Hard 8 3 Y Y
# Walker Run 24 6 N N
# Walker Run Backwards 24 6 N Y
# Walker Stand 24 6 N N
# Walker Walk 24 6 N N
# Walker Walk Backwards 24 6 N Y
