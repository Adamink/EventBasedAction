function [] = FilterAndKeepEvents(...
    fileID, ... % log file
    aedat, events, eventsPerFullFrame, ...
    startTime, stopTime, fileName, ...
    XYZPOS, sx, sy, nbcam, thrEventHotPixel, dt, ...
    xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, ... % 1st mask coordinates
    xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2, ... % 2nd mask coordinates
    do_subsampling, reshapex, reshapey, ...
    saveHDF5, convert_labels)

startTime = uint32(startTime);
stopTime  = uint32(stopTime);

% Extract and filter events from aedat
[startIndex, stopIndex, pol, X, y, cam, timeStamp] = ...
extract_from_aedat(...
                aedat, events, ...
                startTime, stopTime, ...
                sx, sy, nbcam, ...
                thrEventHotPixel, dt, ...
                xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, ...
                xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2);

% DVSfilenameh5 = strcat(fileName,'.h5');
%l = length(timeStamp);
%IMovie = zeros(length(timeStamp), 5);
%for i=1:l
%    IMovie(i, 0) = cam(i);
%    IMovie(i, 1) = 
%end
%DVSfilenameh5 = strcat(fileName, '_raw', '.h5');
%h5create(DVSfilenameh5, '/DVS', [l, 5]);
DVSfilenamemat = strcat(fileName, '_raw', '.mat')
save(DVSfilenamemat, 'startIndex');
save(DVSfilenamemat, 'stopIndex','-append');
save(DVSfilenamemat, 'startTime','-append');
save(DVSfilenamemat, 'stopTime','-append');
save(DVSfilenamemat, 'pol','-append');
save(DVSfilenamemat, 'X','-append');
save(DVSfilenamemat, 'y','-append');
save(DVSfilenamemat, 'cam','-append');
save(DVSfilenamemat, 'timeStamp','-append');
end
