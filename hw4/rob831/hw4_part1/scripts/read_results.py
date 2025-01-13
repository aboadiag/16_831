import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# RUN COMMAND:
# !python3 "hw4/rob831/scripts/read_results.py"


# List of event files to include
experiment_files = [
    "hw4_q2_obstacles_singleiteration_obstacles-hw4_part1-v0_13-11-2024_20-25-15",
    "hw4_ q3_obstacles _obstacles-hw4_part1-v0_13-11-2024_23-35-23",
    "hw4_q3_reacher_reacher-hw4_part1-v0_14-11-2024_00-57-59",
    "hw4_q3_cheetah_cheetah-hw4_part1-v0_14-11-2024_01-43-08",
    "hw4_q4_reacher_horizon5 _reacher-hw4_part1-v0_14-11-2024_03-05-37",
    "hw4_q4_reacher_horizon15 _reacher-hw4_part1-v0_14-11-2024_03-21-36",
    "hw4_q4_reacher_horizon30 _reacher-hw4_part1-v0_14-11-2024_03-44-43",
    "hw4_q4_reacher_numseq100_reacher-hw4_part1-v0_14-11-2024_16-00-37",
    "hw4_ q4_reacher_numseq1000_reacher-hw4_part1-v0_14-11-2024_16-21-27",
    "hw4_q4_reacher_ensemble1_reacher-hw4_part1-v0_14-11-2024_18-45-53",
    "hw4_q4_reacher_ensemble3_reacher-hw4_part1-v0_14-11-2024_18-50-27",
    "hw4_ q4_reacher_ensemble5_reacher-hw4_part1-v0_14-11-2024_19-03-40"
]


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob

    # Specify the log directory
    logdir = "/content/hw_16831/hw4/data"
    eventfile = glob.glob(logdir)[0]

    X, Y = get_section_results(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))