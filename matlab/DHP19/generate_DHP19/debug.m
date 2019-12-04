rootCodeFolder = '/mnt/hdd/wuxiao/winter/matlab/DHP19/'; % root directory of the git repo.
rootDataFolder = '/mnt/hdd/wuxiao/DHP19/'; % root directory of the data downloaded from resiliosync.
% outDatasetFolder = '/mnt/data1/wuxiao/DHP19/matlab_output';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cameras number and resolution. Constant for DHP19.
nbcam = 4;
sx = 344;
sy = 260;

%%%%%%%%%%% PARAMETERS: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Average num of events per camera, for constant count frames.
eventsPerFrame = 7500; 

% Flag and sizes for subsampling the original DVS resolution.
% If no subsample, keep (sx,sy) original img size.
do_subsampling = false;
reshapex = sx;
reshapey = sy;

% Flag to save accumulated recordings.
saveHDF5 = true;

% Flag to convert labels
convert_labels = true;

save_log_special_events = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hot pixels threshold (pixels spiking above threshold are filtered out).
thrEventHotPixel = 1*10^4;

% Background filter: events with less than dt (us) to neighbors pass through.
dt = 70000;

%%% Masks for IR light in the DVS frames.
% Mask 1
xmin_mask1 = 780;
xmax_mask1 = 810;
ymin_mask1 = 115;
ymax_mask1 = 145;
% Mask 2
xmin_mask2 = 346*3 + 214;
xmax_mask2 = 346*3 + 221;
ymin_mask2 = 136;
ymax_mask2 = 144;

%%% Paths     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = datetime('now','Format','yyyy_MM_dd''_''HHmmss');
%
DVSrecFolder = fullfile(rootDataFolder,'DVS_movies/');
viconFolder = fullfile(rootDataFolder,'Vicon_data/');

% output directory where to save files after events accumulation.
out_folder_append = ['h5_dataset_',num2str(eventsPerFrame),'_events'];

addpath(fullfile(rootCodeFolder, 'read_aedat/'));
addpath(fullfile(rootCodeFolder, 'generate_DHP19/'));

% Setup output folder path, according to accumulation type and spatial resolution.
% outputFolder = fullfile(outDatasetFolder, out_folder_append,[num2str(reshapex),'x',num2str(reshapey)]);

% log_path = fullfile(outDatasetFolder, out_folder_append);
% log_file = sprintf('%s/log_generation_%sx%s_%s.log',log_path,num2str(reshapex),num2str(reshapey), t);


aedatPath = '/mnt/hdd/wuxiao/DHP19/DVS_movies/S1/session1/mov1.aedat';
movementsPath = '/mnt/hdd/wuxiao/DHP19/DVS_movies/S1/session1';
% labelPath = '/mnt/data1/wuxiao/DHP19/Vicon_data/S1_1_1.mat';
% fileID = fopen('debug.log', 'w');
% out_file = './debug'
movString = 'mov1';
aedat = ImportAedat([movementsPath '/'], strcat(movString, '.aedat'));
% XYZPOS = load(labelPath);
events = int64(aedat.data.polarity.timeStamp);
%%% conditions on special events %%%
try
    specialEvents = int64(aedat.data.special.timeStamp);
    numSpecialEvents = length(specialEvents);

    if numSpecialEvents == 0
        % the field aedat.data.special does not exist
        % for S14_5_3. There are no other cases.
        error('special field is there but is empty');
        
    elseif numSpecialEvents == 1
        
        if (specialEvents-min(events)) > (max(events)-specialEvents)
            % The only event is closer to the end of the recording.
            stopTime = specialEvents;
            startTime = floor(stopTime - n);
        else
            startTime = specialEvents;
            stopTime = floor(startTime + n);
        end
        
        
    elseif (numSpecialEvents == 2) || (numSpecialEvents == 4)
        % just get the minimum value, the others are max 1
        % timestep far from it.
        special = specialEvents(1); %min(specialEvents);
        
        %%% special case, for S14_1_1 %%%
        % if timeStamp overflows, then get events only
        % until the overflow.
        if events(end) < events(1)
            startTime = special;
            stopTime = max(events);
        
        
        %%% regular case %%%
        else
            if (special-events(1)) > (events(end)-special)
                % The only event is closer to the end of the recording.
                stopTime = special;
                startTime = floor(stopTime - n);
            else
                startTime = special;
                stopTime = floor(startTime + n);
            end
        end 
        
    elseif (numSpecialEvents == 3) || (numSpecialEvents == 5)
        % in this case we have at least 2 distant special
        % events that we consider as start and stop.
        startTime = specialEvents(1);
        stopTime = specialEvents(end);
        
    elseif numSpecialEvents > 5
        % Two recordings with large number of special events.
        % Corrupted recordings, skipped.
         
    end
    
catch 
    % if no special field exists, get first/last regular
    % events (not tested).
    startTime = events(1); 
    stopTime = events(end); 
    
end % end try reading special events



