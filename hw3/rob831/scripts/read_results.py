import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np

# Specify the log directory
logdir = "/content/hw_16831/hw3/data"

# List of event files to include
ddqn_experiment_files = [
    "q1_doubledqn_1_LunarLander-v3_29-10-2024_20-47-36",
    "q1_doubledqn_2_LunarLander-v3_29-10-2024_21-21-11",
    "q1_doubledqn_3_LunarLander-v3_29-10-2024_21-42-45"
]

dqn_experiment_files = [
    "q1_dqn1_1_LunarLander-v3_28-10-2024_15-27-48",
    "q1_dqn_2_LunarLander-v3_28-10-2024_16-02-41",
    "q1_dqn_3_LunarLander-v3_28-10-2024_16-24-12"
]

def get_section_results(file):
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        if len(X) > 120:
            break
    return X, Y

dd_all_X = []
dd_all_Y = []

dq_all_X = []
dq_all_Y = []

# Read and aggregate results from each specified experiment
for experiment in ddqn_experiment_files:
    eventfiles = glob.glob(os.path.join(logdir, f'**/{experiment}/events.out.tfevents.*'), recursive=True)
    if eventfiles:
        eventfile = eventfiles[0]
        print(f"Using log file: {eventfile}")
        X, Y = get_section_results(eventfile)
        dd_all_X.append(X)
        dd_all_Y.append(Y)

for experiment in dqn_experiment_files:
    eventfiles = glob.glob(os.path.join(logdir, f'**/{experiment}/events.out.tfevents.*'), recursive=True)
    if eventfiles:
        eventfile = eventfiles[0]
        print(f"Using log file: {eventfile}")
        X, Y = get_section_results(eventfile)
        dq_all_X.append(X)
        dq_all_Y.append(Y)

# Calculate average returns and standard deviation across experiments for DDQN
dd_average_X = []
dd_average_Y = []
dd_std_Y = []

max_length_ddqn = min(len(y) for y in dd_all_Y)
for i in range(max_length_ddqn):
    valid_returns = [y[i] for y in dd_all_Y if len(y) > i]
    dd_average_Y.append(np.mean(valid_returns))
    dd_std_Y.append(np.std(valid_returns))
    dd_average_X.append(dd_all_X[0][i])  # Use X values from the first experiment

# For DQN
dq_average_X = []
dq_average_Y = []
dq_std_Y = []

max_length_dqn = min(len(y) for y in dq_all_Y)
for i in range(max_length_dqn):
    valid_returns = [y[i] for y in dq_all_Y if len(y) > i]
    dq_average_Y.append(np.mean(valid_returns))
    dq_std_Y.append(np.std(valid_returns))
    dq_average_X.append(dq_all_X[0][i])  # Use X values from the first experiment

# Plotting average returns with error bars
plt.figure(figsize=(10, 6))
plt.plot(dd_average_X, dd_average_Y, label='DDQN Avg Returns', color='blue')
plt.fill_between(dd_average_X, np.array(dd_average_Y) - np.array(dd_std_Y), 
                 np.array(dd_average_Y) + np.array(dd_std_Y), color='blue', alpha=0.2)

plt.plot(dq_average_X, dq_average_Y, label='DQN Avg Returns', color='orange')
plt.fill_between(dq_average_X, np.array(dq_average_Y) - np.array(dq_std_Y), 
                 np.array(dq_average_Y) + np.array(dq_std_Y), color='orange', alpha=0.2)

plt.xlabel('Train Steps')
plt.ylabel('Average Return')
plt.title('Average Training Results Across Experiments with Error Bars')
plt.legend()
plt.grid(True)
plt.savefig('Combo_AvgReturns4_with_error_bars.png')
print('successfully saved!')

# import glob
# import os
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import matplotlib.pyplot as plt

# # Specify the log directory
# logdir = "/content/hw_16831/hw3/data"

# # List of event files to include
# ddqn_experiment_files = [
#     "q1_doubledqn_1_LunarLander-v3_29-10-2024_20-47-36",
#     "q1_doubledqn_2_LunarLander-v3_29-10-2024_21-21-11",
#     "q1_doubledqn_3_LunarLander-v3_29-10-2024_21-42-45"
# ]

# dqn_experiment_files = [
#     "q1_dqn1_1_LunarLander-v3_28-10-2024_15-27-48",
#     "q1_dqn_2_LunarLander-v3_28-10-2024_16-02-41",
#     "q1_dqn_3_LunarLander-v3_28-10-2024_16-24-12"
# ]
# def get_section_results(file):
#     X = []
#     Y = []
#     for e in tf.train.summary_iterator(file):
#         for v in e.summary.value:
#             if v.tag == 'Train_EnvstepsSoFar':
#                 X.append(v.simple_value)
#             elif v.tag == 'Train_AverageReturn':
#                 Y.append(v.simple_value)
#         if len(X) > 120:
#             break
#     return X, Y

# dd_all_X = []
# dd_all_Y = []

# dq_all_X = []
# dq_all_Y = []

# # Read and aggregate results from each specified experiment
# for experiment in ddqn_experiment_files:
#     eventfiles = glob.glob(os.path.join(logdir, f'**/{experiment}/events.out.tfevents.*'), recursive=True)
#     if eventfiles:
#         eventfile = eventfiles[0]
#         print(f"Using log file: {eventfile}")
#         X, Y = get_section_results(eventfile)
#         dd_all_X.append(X)
#         dd_all_Y.append(Y)

# for experiment in dqn_experiment_files:
#     eventfiles = glob.glob(os.path.join(logdir, f'**/{experiment}/events.out.tfevents.*'), recursive=True)
#     if eventfiles:
#         eventfile = eventfiles[0]
#         print(f"Using log file: {eventfile}")
#         X, Y = get_section_results(eventfile)
#         dq_all_X.append(X)
#         dq_all_Y.append(Y)


# # Calculate average returns across experiments --> DDQN
# dd_average_X = []
# dd_average_Y = []
# dq_average_X = []
# dq_average_Y = []

# max_length_ddqn = min(len(y) for y in dd_all_Y)
# for i in range(max_length_ddqn):
#     total_return = 0
#     count = 0
#     for y in dd_all_Y:
#         if len(y) > i:
#             total_return += y[i]
#             count += 1
#     dd_average_Y.append(total_return / count if count > 0 else 0)
#     dd_average_X.append(dd_all_X[0][i])  # Use X values from the first experiment

# # for DQN
# max_length_dqn = min(len(y) for y in dq_all_Y)
# for i in range(max_length_dqn):
#     total_return = 0
#     count = 0
#     for y in dq_all_Y:
#         if len(y) > i:
#             total_return += y[i]
#             count += 1
#     dq_average_Y.append(total_return / count if count > 0 else 0)
#     dq_average_X.append(dq_all_X[0][i])  # Use X values from the first experiment

# # labels=['DDQN Avg Returns', 'DQN Avg Returns']

# # Plotting average returns
# plt.figure(figsize=(10, 6))
# plt.plot(dd_average_X, dd_average_Y, label='DDQN Avg Returns', color='blue')
# plt.plot(dq_average_X, dq_average_Y, label='DQN Avg Returns', color='orange')
# plt.xlabel('Train Steps')
# plt.ylabel('Average Return')
# plt.title('Average Training Results Across Experiments')
# plt.legend()
# plt.grid(True)
# plt.savefig('Combo_AvgReturns4.png')
# print('successfully saved!')

# ################# DRAFT 3333333333333333333333333333333333333333333333333

# # # Assuming all X lists are the same length
# # max_length = min(len(x) for x in dd_all_X)
# # for i in range(max_length):
# #     avg_return = sum(y[i] for y in dd_all_Y if len(y) > i) / len(dd_all_Y)
# #     dd_average_X.append(dd_all_X[0][i])  # Use the X values from the first experiment
# #     dd_average_Y.append(avg_return)


# # # Assuming all X lists are the same length
# # max_length = min(len(x) for x in dq_all_X)
# # for i in range(max_length):
# #     avg_return = sum(y[i] for y in dq_all_Y if len(y) > i) / len(dq_all_Y)
# #     dq_average_X.append(dq_all_X[0][i])  # Use the X values from the first experiment
# #     dq_average_Y.append(avg_return)
# # Assuming all X lists are the same length

# # import glob
# # import os
# # import tensorflow.compat.v1 as tf
# # tf.disable_v2_behavior()
# # import matplotlib.pyplot as plt

# # # Enable inline plotting
# # # %matplotlib inline

# # # Specify the log directory
# # logdir = "/content/hw_16831/hw3/data"

# # # List of event files to include
# # experiment_files = [
# #     "q1_doubledqn_3_LunarLander-v3_29-10-2024_21-42-45",
# #     "q1_doubledqn_3_LunarLander-v3_29-10-2024_22-15-46",
# #     "q1_doubledqn_3_LunarLander-v3_29-10-2024_23-44-52"
# # ]

# # def get_section_results(file):
# #     X = []
# #     Y = []
# #     for e in tf.train.summary_iterator(file):
# #         for v in e.summary.value:
# #             if v.tag == 'Train_EnvstepsSoFar':
# #                 X.append(v.simple_value)
# #             elif v.tag == 'Train_AverageReturn':
# #                 Y.append(v.simple_value)
# #         if len(X) > 120:
# #             break
# #     return X, Y

# # all_X = []
# # all_Y = []

# # # Read and aggregate results from each specified experiment
# # for experiment in experiment_files:
# #     eventfiles = glob.glob(os.path.join(logdir, f'**/{experiment}/events.out.tfevents.*'), recursive=True)
# #     if eventfiles:
# #         eventfile = eventfiles[0]
# #         print(f"Using log file: {eventfile}")
# #         X, Y = get_section_results(eventfile)
# #         all_X.append(X)
# #         all_Y.append(Y)

# # # Calculate average returns across experiments
# # average_X = []
# # average_Y = []

# # # Assuming all X lists are the same length
# # max_length = min(len(x) for x in all_X)
# # for i in range(max_length):
# #     avg_return = sum(y[i] for y in all_Y if len(y) > i) / len(all_Y)
# #     average_X.append(all_X[0][i])  # Use the X values from the first experiment
# #     average_Y.append(avg_return)

# # # Plotting average returns
# # plt.figure(figsize=(10, 6))
# # plt.plot(average_X, average_Y, label='Average Return across Experiments')
# # plt.xlabel('Train Steps')
# # plt.ylabel('Average Return')
# # plt.title('Average Training Results Across Experiments')
# # plt.legend()
# # plt.grid(True)
# # plt.show()


# # # # import argparse
# # # import glob
# # # import os
# # # import tensorflow.compat.v1 as tf
# # # tf.disable_v2_behavior()
# # # import matplotlib.pyplot as plt


# # # # Directly specify the log directory
# # # logdir = "/content/hw_16831/hw3/data"

# # # # Find event files in the specified directory and subdirectories
# # # eventfiles = glob.glob(os.path.join(logdir, '**', '*.tfevents.*'), recursive=True)

# # # if not eventfiles:
# # #     print(f"No log files found in {logdir}. Please check the directory.")
# # #     exit(1)

# # # eventfile = eventfiles[0]  # Use the first found event file
# # # print(f"Using log file: {eventfile}")

# # # def get_section_results(file):
# # #     """
# # #         requires tensorflow==1.12.0
# # #     """
# # #     X = []
# # #     Y = []
# # #     for e in tf.train.summary_iterator(file):
# # #         for v in e.summary.value:
# # #             if v.tag == 'Train_EnvstepsSoFar':
# # #                 X.append(v.simple_value)
# # #             elif v.tag == 'Train_AverageReturn':
# # #                 Y.append(v.simple_value)
# # #         if len(X) > 120:
# # #             break
# # #     return X, Y

# # # # if __name__ == '__main__':
# # # #     # parser = argparse.ArgumentParser()
# # # #     # parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
# # # #     # args = parser.parse_args()

# # # #     # logdir = os.path.join(args.logdir, 'events*')
# # # #     # eventfile = glob.glob(logdir)[0]

# # # X, Y = get_section_results(eventfile)
    
# # # for i, (x, y) in enumerate(zip(X, Y)):
# # #         print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
# # #         plt.plot(X, Y, label=os.path.basename(eventfile))  # Use the filename as the label

# # # # Plot the results
# # # plt.figure(figsize=(10, 6))  # Optional: Set the size of the plot
# # # plt.plot(X, Y, label=os.path.basename(eventfile))  # Plot X vs. Y
# # # plt.xlabel('Train Steps')  # Label for x-axis
# # # plt.ylabel('Average Return')  # Label for y-axis
# # # plt.title('Training Results')  # Title of the plot
# # # plt.legend()  # Show legend
# # # plt.grid(True)  # Optional: Show grid for better readability
# # # plt.show()  # Display the plot
