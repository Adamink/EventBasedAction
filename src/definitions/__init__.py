import os
from os.path import join
import yaml

with open(join(os.path.dirname(os.path.abspath(__file__)),'../../configs/settings.yaml'), 'r') as f:
    settings = yaml.safe_load(f)
    project_dir = settings['project_dir']
    data_dir = settings['data_dir']

# projects
src_dir = join(project_dir, 'src/')
checkpoints_dir = join(project_dir, 'checkpoints/')
figures_dir = join(project_dir, 'figures/')
results_dir = join(project_dir, 'results/')
log_dir = results_dir 
stat_dir = join(project_dir, 'results/stat/')
config_dir = join(project_dir, 'configs/')

# data
vicon_dir = join(data_dir, 'Vicon_data/')
p_mat_dir = join(data_dir, 'P_matrices/')
pose_groundtruth_dir = join(data_dir, 'pose_groundtruth/')
skeleton_groundtruth_dir = join(data_dir, 'skeleton_groundtruth/')
pointcloud_dir = join(data_dir, 'nocut_subsample_sliding_windows/')
raw_events = join(data_dir, 'matlab_output/344x260raw/')
pose_7500_dir = join(data_dir, 'matlab_output/h5_dataset_7500_events/344x260/')
event_high_frame_dir = join(data_dir, 'event_high_frame/')
action_dir = join(data_dir, 'matlab_action/h5_dataset_7500_events/344x260/')
pose7500numpy_dir = join(data_dir, 'pose7500numpy/')

new_pose_7500_dir = join(data_dir, 'matlab_official/h5_dataset_7500_events/344x260/')
new_pose_numpy_dir = join(data_dir, 'new_pose_numpy/')
