# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:28:36 2024
CME FOR MOTILITY 
@author: Nasser
"""
#%% Imports and initial settings

# for compatibility with Python 2 and 3
from __future__ import division, unicode_literals, print_function
import trackpy as tp
from scipy.optimize import curve_fit
from scipy.stats import t, linregress

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
# Enable %matplotlib qt for pop-out plots in Spyder
%matplotlib qt
# Update matplotlib parameters for consistent font usage
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Latin Modern Roman",
    "font.sans-serif": ["Helvetica"]
})
#    "font.family": "Latin Modern Roman",

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rc('figure', figsize=(8, 8))
plt.rc('image', cmap='gray')

# Experiment conditions
cell = r'Ac 30010 ' # cell line
cond = r'  '# extra important condition. i.e date control or 0.4% O2

# Define conditions constants linked to experiment
# mag = 1.634  # µm/PIXEL
# dt = 30  # SECONDS
betamax = 1.75
betamin = 0.55
betaminn = str(betamin)
betamaxx = str(betamax)
# filtpc = 16.1

# Define path
str_path = (r'')
path = Path(str_path)

def read_experiment_parameters(folder):
    params_path = Path(folder) / 'fitted_parameters.csv'
    params_df = pd.read_csv(params_path)
    dt = float(params_df[params_df['Parameter'] == 'dt (s)']['Value'])
    mag = float(params_df[params_df['Parameter'] == 'resolution (um/pix)']['Value'])
    return dt, mag

params = read_experiment_parameters(str_path)
dt = params[0]
mag= params[1]


trajectories = pd.read_csv(path / 'trajectories.csv')

# Add these lines at the beginning of script to create the 'final' and 'segments' subfolders
path1 = path / 'CME'
path1.mkdir(parents=True, exist_ok=True)


#%% #%% CME ANALYSIS

# def compute_cme(trajectory_segment):
#     start_pos = np.array([trajectory_segment.iloc[0]['x'], trajectory_segment.iloc[0]['y']])
#     end_pos = np.array([trajectory_segment.iloc[-1]['x'], trajectory_segment.iloc[-1]['y']])
#     displacement = np.linalg.norm(end_pos - start_pos)
    
#     # Calculate total path length
#     path_length = 0
#     for i in range(1, len(trajectory_segment)):
#         current_pos = np.array([trajectory_segment.iloc[i]['x'], trajectory_segment.iloc[i]['y']])
#         prev_pos = np.array([trajectory_segment.iloc[i-1]['x'], trajectory_segment.iloc[i-1]['y']])
#         path_length += np.linalg.norm(current_pos - prev_pos)
    
#     # Calculate CME
#     cme = displacement / path_length if path_length > 0 else 0
#     return cme

# # Specify the total time t in minutes
# total_time_minutes = 10  # Example: compute CME over 10 minutes
# total_time_seconds = total_time_minutes * 60
# time_steps = int(total_time_seconds / dt)

# # Initialize a list to store CME data
# cme_data = []

# # Group by particle and compute CME for each valid segment of the trajectory
# for particle, group in trajectories.groupby('particle'):
#     num_segments = len(group) // time_steps
#     for segment in range(num_segments):
#         start_index = segment * time_steps
#         end_index = start_index + time_steps
#         trajectory_segment = group.iloc[start_index:end_index]
#         cme = compute_cme(trajectory_segment)
#         cme_data.append({'particle': particle, 'CME': cme, 'segment': segment + 1})

# # Create a dataframe from the CME data
# cme_df = pd.DataFrame(cme_data)

# # Remove rows with CME value of 0
# cme_df = cme_df[cme_df['CME'] != 0]

# # Calculate mean and SEM of CME
# cme_mean = cme_df['CME'].mean()
# cme_sem = cme_df['CME'].sem()

# # Create a summary dataframe
# summary_df = pd.DataFrame({
#     'CME Mean': [cme_mean],
#     'CME SEM': [cme_sem],
#     'Total Time (minutes)': [total_time_minutes]
# })

# # Save CME data and summary to an Excel file
# output_path = path1 / 'CME_results.xlsx'
# with pd.ExcelWriter(output_path) as writer:
#     cme_df.to_excel(writer, sheet_name='CME Data', index=False)
#     summary_df.to_excel(writer, sheet_name='CME Summary', index=False)

# print(f"CME analysis complete. Results saved to {output_path}")


# #%% Iterate over total time minutes from 1 to 20 minutes 

# '''
# iterate over 20 minutes (from 1 minute to 20 minutes)
# store the resulting average CME and its SEM, along with the corresponding time into an excel sheet called CME_time
# then plot and show the relation of how increasing time over which CME is computed affects the final average CME value 
# '''

# # Function to compute CME for a given trajectory segment over a specified time t in minutes
# def compute_cme(trajectory_segment):
#     start_pos = np.array([trajectory_segment.iloc[0]['x'], trajectory_segment.iloc[0]['y']])
#     end_pos = np.array([trajectory_segment.iloc[-1]['x'], trajectory_segment.iloc[-1]['y']])
#     displacement = np.linalg.norm(end_pos - start_pos)
    
#     # Calculate total path length
#     path_length = 0
#     for i in range(1, len(trajectory_segment)):
#         current_pos = np.array([trajectory_segment.iloc[i]['x'], trajectory_segment.iloc[i]['y']])
#         prev_pos = np.array([trajectory_segment.iloc[i-1]['x'], trajectory_segment.iloc[i-1]['y']])
#         path_length += np.linalg.norm(current_pos - prev_pos)
    
#     # Calculate CME
#     cme = displacement / path_length if path_length > 0 else 0
#     return cme

# # Initialize a list to store CME data over different times
# cme_time_data = []

# # Iterate over total time minutes from 1 to 20 minutes
# for total_time_minutes in range(1, 21):
#     total_time_seconds = total_time_minutes * 60
#     time_steps = int(total_time_seconds / dt)
    
#     # Initialize a list to store CME data for this specific time
#     cme_data = []

#     # Group by particle and compute CME for each valid segment of the trajectory
#     for particle, group in trajectories.groupby('particle'):
#         num_segments = len(group) // time_steps
#         for segment in range(num_segments):
#             start_index = segment * time_steps
#             end_index = start_index + time_steps
#             trajectory_segment = group.iloc[start_index:end_index]
#             cme = compute_cme(trajectory_segment)
#             if cme != 0:
#                 cme_data.append(cme)
    
#     # Calculate mean and SEM of CME for this specific time
#     if cme_data:
#         cme_mean = np.mean(cme_data)
#         cme_sem = np.std(cme_data) / np.sqrt(len(cme_data))
#     else:
#         cme_mean = np.nan
#         cme_sem = np.nan
    
#     # Store the results
#     cme_time_data.append({'Total Time (minutes)': total_time_minutes, 'CME Mean': cme_mean, 'CME SEM': cme_sem})

# # Create a dataframe from the CME time data
# cme_time_df = pd.DataFrame(cme_time_data)

# # Save CME time data to an Excel file
# output_path = path1 / 'CME_time.xlsx'
# cme_time_df.to_excel(output_path, index=False)

# print(f"CME time analysis complete. Results saved to {output_path}")


# #%% Plot the relation of how increasing time over which CME is computed affects the final average CME value

# plt.figure(figsize=(10, 6))
# plt.errorbar(cme_time_df['Total Time (minutes)'], cme_time_df['CME Mean'], yerr=cme_time_df['CME SEM'], fmt='-o', capsize=5)
# plt.xlabel('Total Time (minutes)')
# plt.ylabel('Average CME')
# plt.title('Effect of Increasing Time on Average CME Value')
# plt.grid(True)
# plt.tight_layout()

# # Save the plot
# plot_output_path = path1 / 'CME_vs_time.png'
# plt.savefig(plot_output_path, dpi=200)

# # Show the plot
# plt.show()


#%% approach 2
# Precompute all pairwise distances in each trajectory group
trajectory_distances = {}
trajectory_positions = {}  # To store positions for displacement calculation
for particle, group in trajectories.groupby('particle'):
    positions = group[['x', 'y']].to_numpy()
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    trajectory_distances[particle] = distances
    trajectory_positions[particle] = positions  # Store positions

# Function to compute CME for a given trajectory segment
def compute_cme(start_idx, end_idx, distances, positions):
    start_pos = positions[start_idx]
    end_pos = positions[end_idx]
    displacement = np.linalg.norm(end_pos - start_pos)
    path_length = distances[start_idx:end_idx].sum()
    cme = displacement / path_length if path_length > 0 else 0
    return cme

# Initialize a list to store CME data over different times
cme_time_data = []

# Iterate over total time minutes from 1 to 20 minutes
for total_time_minutes in range(1, 61):
    total_time_seconds = total_time_minutes * 60
    time_steps = int(total_time_seconds / dt)
    
    # Initialize a list to store CME data for this specific time
    cme_data = []

    # Compute CME for each valid segment of the trajectory
    for particle, distances in trajectory_distances.items():
        positions = trajectory_positions[particle]
        num_segments = len(distances) // time_steps
        for segment in range(num_segments):
            start_index = segment * time_steps
            end_index = start_index + time_steps
            if end_index >= len(positions):  # Ensure we don't go out of bounds
                continue
            cme = compute_cme(start_index, end_index, distances, positions)
            if cme != 0:
                cme_data.append(cme)
    
    # Calculate mean and SEM of CME for this specific time
    if cme_data:
        cme_mean = np.mean(cme_data)
        cme_sem = np.std(cme_data) / np.sqrt(len(cme_data))
    else:
        cme_mean = np.nan
        cme_sem = np.nan
    
    # Store the results
    cme_time_data.append({'Total Time (minutes)': total_time_minutes, 'CME Mean': cme_mean, 'CME SEM': cme_sem})

# Create a dataframe from the CME time data
cme_time_df = pd.DataFrame(cme_time_data)

# Save CME time data to an Excel file
output_path = path1 / 'CME_time_optimized.xlsx'
cme_time_df.to_excel(output_path, index=False)

print(f"CME time analysis complete. Results saved to {output_path}")

#%% Plot the relation of how increasing time over which CME is computed affects the final average CME value

fig, ax = plt.subplots(figsize=(6, 5))

# Plot the data with error bars
ax.errorbar(cme_time_df['Total Time (minutes)'], cme_time_df['CME Mean'],
            fmt='b.', alpha=0.7, linewidth=5, capsize=5, markersize=10, zorder=1)

# Setting labels and title
ax.set_xlabel(r'$t_{window}$ [min]', fontsize=16)
ax.set_ylabel(r'$\overline{CME}$', fontsize=16)

# Set font sizes for ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.xlim(left=0,right=60)
plt.ylim(bottom=0,top=1)

# Adding grid for better visualization
plt.grid(True)

# Reverse the order of handles and labels for the legend
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], fontsize=12, loc="best")

# Ensure a tight layout
plt.tight_layout()

# Save the plot
plot_output_path = path1 / 'CME_vs_time_optimized.png'
plt.savefig(plot_output_path, dpi=200)

# Show the plot
plt.show()

#%% Compute and plot the second derivative of average CME to find the inflection point
#%% Compute and plot the second derivative of average CME

# Function to compute the second derivative
def compute_second_derivative(y_values, x_values):
    second_derivative = np.gradient(y_values, x_values)
    return second_derivative

# Compute the second derivative of the average CME
cme_mean = cme_time_df['CME Mean'].to_numpy()
time_values = cme_time_df['Total Time (minutes)'].to_numpy()
derivative = compute_second_derivative(cme_mean, time_values)

# Add the second derivative to the dataframe
cme_time_df['Derivative'] = derivative

# Save the updated dataframe to an Excel file
output_path = path1 / 'CME_time_with_derivatives.xlsx'
cme_time_df.to_excel(output_path, index=False)

# Plot the second derivative to visualize the inflection points
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(time_values, derivative, 'r.', linewidth=2, markersize=10)

# Setting labels and title
ax.set_xlabel(r'$t_{window}$ [min]', fontsize=16)
ax.set_ylabel(r'$\frac{d}{dt} \overline{\mathrm{CME}}$', fontsize=16)

# Set font sizes for ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(left=0, right=60)
plt.ylim(bottom=min(derivative), top=0.025)

# Adding grid for better visualization
plt.grid(True)

# Ensure a tight layout
plt.tight_layout()

# Save the plot
plot_output_path = path1 / 'CME_derivative.png'
plt.savefig(plot_output_path, dpi=200)

# Show the plot
plt.show()


#%% iterate through all folders and take mean 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base directory to search
base_dir = r' '

# Directory to save the concatenated results
save_dir = r' '
os.makedirs(save_dir, exist_ok=True)

# Initialize list to store dataframes
dataframes = []
folder_names = []

# Traverse the directory tree
for root, dirs, files in os.walk(base_dir):
    if 'final2' in dirs:
        final2_path = os.path.join(root, 'final2')
        file_path = os.path.join(final2_path, 'CME_time_with_derivatives.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df['Folder Path'] = root
            dataframes.append(df)
            folder_names.append(root)

# Concatenate all dataframes
if dataframes:
    all_data = pd.concat(dataframes, ignore_index=True)
    # Save concatenated dataframe
    concatenated_path = os.path.join(save_dir, 'CME_persistence_all_data.xlsx')
    all_data.to_excel(concatenated_path, index=False)

    # Calculate mean and SEM for each time point for CME Mean
    mean_sem_cme_df = all_data.groupby('Total Time (minutes)').agg({
        'CME Mean': ['mean', 'sem']
    }).reset_index()
    mean_sem_cme_df.columns = ['Total Time (minutes)', 'CME Mean', 'CME SEM']

    # Calculate mean and SEM for each time point for Derivative
    mean_sem_derivative_df = all_data.groupby('Total Time (minutes)').agg({
        'Derivative': ['mean', 'sem']
    }).reset_index()
    mean_sem_derivative_df.columns = ['Total Time (minutes)', 'Derivative Mean', 'Derivative SEM']

    # Save mean and SEM dataframes
    mean_sem_cme_path = os.path.join(save_dir, 'CME_persistence_mean_sem.xlsx')
    mean_sem_cme_df.to_excel(mean_sem_cme_path, index=False)

    mean_sem_derivative_path = os.path.join(save_dir, 'CME_persistence_derivative_mean_sem.xlsx')
    mean_sem_derivative_df.to_excel(mean_sem_derivative_path, index=False)

    print(f"Number of valid experiments: {len(set(folder_names))}")
    print(f"Data concatenated and saved to {concatenated_path}")
    print(f"Mean and SEM calculated and saved to {mean_sem_cme_path}")
    print(f"Derivative Mean and SEM calculated and saved to {mean_sem_derivative_path}")

    # Plot the relation of how increasing time over which CME is computed affects the final average CME value
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the data with error bars and larger dots
    ax.errorbar(mean_sem_cme_df['Total Time (minutes)'], mean_sem_cme_df['CME Mean'], 
                yerr=mean_sem_cme_df['CME SEM'], fmt='b.', alpha=0.7, linewidth=3, 
                capsize=5, label=r'$N_{exp}=9$', markersize=10, zorder=1)

    # Setting labels and title
    ax.set_xlabel(r'$t$ [min]', fontsize=20)
    ax.set_ylabel(r'$\overline{\mathrm{CME}}$', fontsize=20)

    # Set font sizes for ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim(left=0, right=60)
    plt.ylim(bottom=0, top=1)

    # Adding grid for better visualization
    plt.grid(True)

    # Add legend
    ax.legend(fontsize=16, loc="best")

    # Ensure a tight layout
    plt.tight_layout()

    # Save the plot
    plot_output_path = os.path.join(save_dir, 'CME_vs_time_optimized.png')
    plt.savefig(plot_output_path, dpi=200)

    # Show the plot
    plt.show()

    # Plot the relation of how increasing time over which CME is computed affects the derivative of CME
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the data with error bars and larger dots
    ax.errorbar(mean_sem_derivative_df['Total Time (minutes)'], mean_sem_derivative_df['Derivative Mean'], 
                yerr=mean_sem_derivative_df['Derivative SEM'], fmt='r.', alpha=0.7, linewidth=3, 
                capsize=5, label=r'$N_{exp}=9$', markersize=10, zorder=1)

    # Setting labels and title
    ax.set_xlabel(r'$t$ [min]', fontsize=20)
    ax.set_ylabel(r'$\frac{d}{dt} \overline{\mathrm{CME}}$', fontsize=20)

    # Set font sizes for ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim(left=0, right=60)
    plt.ylim(bottom=min(mean_sem_derivative_df['Derivative Mean'] - mean_sem_derivative_df['Derivative SEM']),
             top=max(mean_sem_derivative_df['Derivative Mean'] + mean_sem_derivative_df['Derivative SEM']))

    # Adding grid for better visualization
    plt.grid(True)

    # Add legend
    ax.legend(fontsize=16, loc="best")

    # Ensure a tight layout
    plt.tight_layout()

    # Save the plot
    plot_output_path = os.path.join(save_dir, 'Derivative_vs_time_optimized.png')
    plt.savefig(plot_output_path, dpi=200)

    # Show the plot
    plt.show()

else:
    print("No valid 'final2' folders found with the 'CME_time_with_derivatives.xlsx' file.")
