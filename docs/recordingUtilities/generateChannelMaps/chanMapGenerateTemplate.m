%% chan map 
%% example 2.0 4-shank in an horzontal stripe 
%  list of channel indices (and give
% an index to dead channels too). chanMap(1) is the row in the raw binary
% file for the first channel. chanMap(385) = flipper in my case

chanMap = 1:384;

% the first thing Kilosort does is reorder the data with data = data(chanMap, :).
% Now we declare which channels are "connected" in this normal ordering, 
% meaning not dead or used for non-ephys data

connected = true(384, 1); 
%connected(385) = 0;%flipper channel

% now we define the horizontal (x) and vertical (y) coordinates of these
% 34 channels. For dead or nonephys channels the values won't matter. Again
% I will take this information from the specifications of the probe. These
% are in um here, but the absolute scaling doesn't really matter in the
% algorithm. 

xcoords = repmat([0,32, 250,250+32,500,500+32,750, 750+32],[1,384/8])';
%xcoords(385) = NaN;
ycoords =  sort(repmat([2865:-15: 2865-(384/8*15)+15], [1,8])');
%ycoords(385) = NaN;
% Often, multi-shank probes or tetrodes will be organized into groups of
% channels that cannot possibly share spikes with the rest of the probe. This helps
% the algorithm discard noisy templates shared across groups. In
% this case, we set kcoords to indicate which group the channel belongs to.
% In our case all channels are on the same shank in a single group so we
% assign them all to group 1. 

kcoords =repmat([1,1,2,2,3,3,4,4],[1,384/8])';
%kcoords(385) = NaN;
% at this point in Kilosort we do data = data(connected, :), ycoords =
% ycoords(connected), xcoords = xcoords(connected) and kcoords =
% kcoords(connected) and no more channel map information is needed (in particular
% no "adjacency graphs" like in KlustaKwik). 
% Now we can save our channel map for the eMouse. 
% kcoords is used to forcefully restrict templates to channels in the same
% channel group. An option can be set in the master_file to allow a fraction 
% of all templates to span more channel groups, so that they can capture shared 
% noise across all channels. This option is

% ops.criterionNoiseChannels = 0.2; 

% if this number is less than 1, it will be treated as a fraction of the total number of clusters

% if this number is larger than 1, it will be treated as the "effective
% number" of channel groups at which to set the threshold. So if a template
% occupies more than this many channel groups, it will not be restricted to
% a single channel group. 
% would be good to also save the sampling frequency here
fs = 30000; 


save('chanMapNP2_4Shank_bottRow.mat', 'chanMap', 'connected', 'xcoords', 'ycoords', 'kcoords', 'fs')

%% example 2.0 4-shank, using only shank 3
%  list of channel indices (and give
% an index to dead channels too). chanMap(1) is the row in the raw binary
% file for the first channel. chanMap(385) = flipper in my case

chanMap = 1:385;

% the first thing Kilosort does is reorder the data with data = data(chanMap, :).
% Now we declare which channels are "connected" in this normal ordering, 
% meaning not dead or used for non-ephys data

connected = true(385, 1); 
connected(385) = 0;%flipper channel

% now we define the horizontal (x) and vertical (y) coordinates of these
% 34 channels. For dead or nonephys channels the values won't matter. Again
% I will take this information from the specifications of the probe. These
% are in um here, but the absolute scaling doesn't really matter in the
% algorithm. 

xcoords = repmat([750,750+32],[1,384/2])';
xcoords(385) = NaN;
ycoords =  sort(repmat([2865:-15: 2865-(384/2*15)+15], [1,2])');
ycoords(385) = NaN;
% Often, multi-shank probes or tetrodes will be organized into groups of
% channels that cannot possibly share spikes with the rest of the probe. This helps
% the algorithm discard noisy templates shared across groups. In
% this case, we set kcoords to indicate which group the channel belongs to.
% In our case all channels are on the same shank in a single group so we
% assign them all to group 1. 

kcoords =repmat([4,4],[1,384/2])';
kcoords(385) = NaN;
% at this point in Kilosort we do data = data(connected, :), ycoords =
% ycoords(connected), xcoords = xcoords(connected) and kcoords =
% kcoords(connected) and no more channel map information is needed (in particular
% no "adjacency graphs" like in KlustaKwik). 
% Now we can save our channel map for the eMouse. 
% kcoords is used to forcefully restrict templates to channels in the same
% channel group. An option can be set in the master_file to allow a fraction 
% of all templates to span more channel groups, so that they can capture shared 
% noise across all channels. This option is

% ops.criterionNoiseChannels = 0.2; 

% if this number is less than 1, it will be treated as a fraction of the total number of clusters

% if this number is larger than 1, it will be treated as the "effective
% number" of channel groups at which to set the threshold. So if a template
% occupies more than this many channel groups, it will not be restricted to
% a single channel group. 
% would be good to also save the sampling frequency here
fs = 30000; 


save(fullfile(dropboxPath, 'spikeSorting', 'chanMapNP2_4Shank_bottRow_shank3_flipper.mat'), 'chanMap', 'connected', 'xcoords', 'ycoords', 'kcoords', 'fs')

%% example 2.0 4-shank, horizontal stripe using shanks 0 and 1 
chanMap = 1:385;

% the first thing Kilosort does is reorder the data with data = data(chanMap, :).
% Now we declare which channels are "connected" in this normal ordering, 
% meaning not dead or used for non-ephys data

connected = true(385, 1); 
connected(385) = 0;%flipper channel

% now we define the horizontal (x) and vertical (y) coordinates of these
% 34 channels. For dead or nonephys channels the values won't matter. Again
% I will take this information from the specifications of the probe. These
% are in um here, but the absolute scaling doesn't really matter in the
% algorithm. 

xcoords = repmat([0,32, 250,250+32],[1,384/4])';
xcoords(385) = NaN;
ycoords =  sort(repmat([2865:-15: 2865-(384/4*15)+15], [1,4])');
ycoords(385) = NaN;
% Often, multi-shank probes or tetrodes will be organized into groups of
% channels that cannot possibly share spikes with the rest of the probe. This helps
% the algorithm discard noisy templates shared across groups. In
% this case, we set kcoords to indicate which group the channel belongs to.
% In our case all channels are on the same shank in a single group so we
% assign them all to group 1. 

kcoords =repmat([1,1,2,2],[1,384/4])';
kcoords(385) = NaN;
% at this point in Kilosort we do data = data(connected, :), ycoords =
% ycoords(connected), xcoords = xcoords(connected) and kcoords =
% kcoords(connected) and no more channel map information is needed (in particular
% no "adjacency graphs" like in KlustaKwik). 
% Now we can save our channel map for the eMouse. 
% kcoords is used to forcefully restrict templates to channels in the same
% channel group. An option can be set in the master_file to allow a fraction 
% of all templates to span more channel groups, so that they can capture shared 
% noise across all channels. This option is

% ops.criterionNoiseChannels = 0.2; 

% if this number is less than 1, it will be treated as a fraction of the total number of clusters

% if this number is larger than 1, it will be treated as the "effective
% number" of channel groups at which to set the threshold. So if a template
% occupies more than this many channel groups, it will not be restricted to
% a single channel group. 
% would be good to also save the sampling frequency here
fs = 30000; 


save(fullfile(dropboxPath, 'spikeSorting', 'chanMapNP2_4Shank_bottRow_shank01_flipper.mat'), 'chanMap', 'connected', 'xcoords', 'ycoords', 'kcoords', 'fs')



%% example 2.0 4-shank, horizontal stripe using shanks 2 and 41 
chanMap = 1:385;

% the first thing Kilosort does is reorder the data with data = data(chanMap, :).
% Now we declare which channels are "connected" in this normal ordering, 
% meaning not dead or used for non-ephys data

connected = true(385, 1); 
connected(385) = 0;%flipper channel

% now we define the horizontal (x) and vertical (y) coordinates of these
% 34 channels. For dead or nonephys channels the values won't matter. Again
% I will take this information from the specifications of the probe. These
% are in um here, but the absolute scaling doesn't really matter in the
% algorithm. 

xcoords = repmat([500,500+32, 750,750+32],[1,384/4])';
xcoords(385) = NaN;
ycoords =  sort(repmat([2865:-15: 2865-(384/4*15)+15], [1,4])');
ycoords(385) = NaN;
% Often, multi-shank probes or tetrodes will be organized into groups of
% channels that cannot possibly share spikes with the rest of the probe. This helps
% the algorithm discard noisy templates shared across groups. In
% this case, we set kcoords to indicate which group the channel belongs to.
% In our case all channels are on the same shank in a single group so we
% assign them all to group 1. 

kcoords =repmat([3,3,4,4],[1,384/4])';
kcoords(385) = NaN;
% at this point in Kilosort we do data = data(connected, :), ycoords =
% ycoords(connected), xcoords = xcoords(connected) and kcoords =
% kcoords(connected) and no more channel map information is needed (in particular
% no "adjacency graphs" like in KlustaKwik). 
% Now we can save our channel map for the eMouse. 
% kcoords is used to forcefully restrict templates to channels in the same
% channel group. An option can be set in the master_file to allow a fraction 
% of all templates to span more channel groups, so that they can capture shared 
% noise across all channels. This option is

% ops.criterionNoiseChannels = 0.2; 

% if this number is less than 1, it will be treated as a fraction of the total number of clusters

% if this number is larger than 1, it will be treated as the "effective
% number" of channel groups at which to set the threshold. So if a template
% occupies more than this many channel groups, it will not be restricted to
% a single channel group. 
% would be good to also save the sampling frequency here
fs = 30000; 


save(fullfile(dropboxPath, 'spikeSorting', 'chanMapNP2_4Shank_bottRow_shank23_flipper.mat'), 'chanMap', 'connected', 'xcoords', 'ycoords', 'kcoords', 'fs')


