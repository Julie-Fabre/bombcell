animal = 'JF067';
day = '2022-02-13';
site = 1;
experiment = 1;
recording =[];
verbose = false; % display load progress and some info figures
load_parts.cam=false;
load_parts.imaging=false;
load_parts.ephys=true;
isSpikeGlx=1;
loadClusters = 0;
[ephysAPfile,aa] = AP_cortexlab_filenameJF(animal,day,experiment,'ephys_ap',site,recording);
AP_load_experimentJF; % -> get stimOn_times, spike_times_timeline
ephysap_path = AP_cortexlab_filenameJF(animal,day,experiment,'ephys_ap',site);
param.rawFolder = [ephysap_path, '/..'];
param.nChannels = 385;
bc_getRawMemMap;

cCount = cumsum(repmat(500, 128, 1), 1);

lick_times_aligned_samples = int32((spike_times_timeline /  co(2) - co(1)) * 30000);

t = lick_times_aligned_samples(103) + [int32(-3000):int32(3000)];

thisMemMap = double(memMapData(100:110, reshape(t, [], 1)))+double(cCount(100:110));
figure()
LinePlotReducer(@plot, double(reshape(t, [], 1)), thisMemMap,'k');
hold on;
tS = lick_times_aligned_samples(106) + [int32(-42):int32(42)];
thisMemMapS = double(memMapData(100:110, reshape(tS, [], 1)))+double(cCount(100:110));
plot(double(reshape(tS, [], 1)), thisMemMapS,'r')
%% make so you flip through xx channels + flip through stim times 
