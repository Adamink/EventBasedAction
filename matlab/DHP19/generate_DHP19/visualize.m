data_pth = '/mnt/data1/wuxiao/DHP19/matlab_output/h5_dataset_7500_events/346x260/';
output_pth = '/mnt/data1/wuxiao/DHP19/matlab_output/h5_dataset_7500_events/percam/';
addpath data_pth
numSubjects=17;
numSessions=5;
for subj = 1:numSubjects
    subj_string=sprintf('S%d', subj);
    sessionPath=fullfile(data_pth, subj_string);
    for sess =1:numSessions
        sessString = sprintf('session%d', sess
load S1_session1_mov1_raw_raw;
index = cam==2;
x_axis = X(index);
y_axis = y(index);
t = timeStamp(index);

n = 50000;
scatter3(x_axis(1:n), t(1:n), y_axis(1:n), 0.5, t(1:n));
xlabel("X");
ylabel("Time");
zlabel("Y");