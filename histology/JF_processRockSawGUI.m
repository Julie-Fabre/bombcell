%% REGISTER ROCKSAW PROCESSED BRAINS TO ALLEN ATLAS 
% dependancies: 
% - loadtiff 
% - matlabelastix
% - uigetfile 
% JF 2021/06/08
%% to dos
% - btter auto brightness contrast scaling
% all colors 

%% params / loading 
myPaths; 
animal = 'JF070';
% get in 'AP' format: histology_ccf.mat with tv_slices, av_slices,
% plane_ap, plane_ml, plane_dv 
allen_atlas_path = [allenAtlasPath 'allenCCF'];
tv = readNPY([allen_atlas_path, filesep, 'template_volume_10um.npy']);
%tv = permute(tv, [2,1,3]);
av = readNPY([allen_atlas_path, filesep, 'annotation_volume_10um_by_index.npy']);
%av = permute(av, [2,1,3]);
st = loadStructureTreeJF([allen_atlas_path, filesep, 'structure_tree_safe_2017.csv']);
bregma = [540,0,570];
slice_path = [extraHDPath,animal,'/slices'];

%% Open dialog box, select images to register + template and load 
[filename,filepath]=uigetfile([brainsawPath '*.tif'],'Select autofluorescence channel imaged brain'); %select green channel
[filenameref,filepathref]=uigetfile({[allenAtlasPath '*.tif']},'Select reference'); %template.tif 
greenChannel = loadtiff([filepath, filename]);
redChannel = loadtiff([filepath, filename(1:end-11), '1_red.tif']);
allenAtlas = loadtiff([filepathref, filenameref]);
allenAtlas10um = readNPY([allenAtlasPath 'allenCCF' filesep 'template_volume_10um.npy']);
allenAtlas10um = flipud(rot90(permute(allenAtlas10um, [3,2,1])));

%% crop template to fit image to register 
StackSlider2(greenChannel,allenAtlas10um,2);

cropAllenLimits = [203, 1317]; 
cropImgLimits = [1, 465]; 
allenAtlas10um = allenAtlas10um(:,:,cropAllenLimits(1):cropAllenLimits(2));
greenChannel = greenChannel(:,:,cropImgLimits(1):cropImgLimits(2));
redChannel = redChannel(:,:,cropImgLimits(1):cropImgLimits(2));

%% register, get transform, apply
extraHDPath = '/media/julie/Elements'
dd=dir([extraHDPath filesep animal '\Substack*']);
cd(regParamsPath)
params = {'01_ARA_affine.txt',...
    '02_ARA_bspline.txt'};

[reg,elastixLog] = elastix(squeeze(greenChannel),allenAtlas10um(1:2:end, 1:2:end, 1:2:end),[],params);%green channel - don't use of teto mouse 
saveastiff(reg, [extraHDPath filesep animal '/regGreenChan.tiff']); 

%get red channel 
[regRed,elastixLog2] = transformix(redChannel,elastixLog);%red channel 
saveastiff(regRed, [extraHDPath filesep animal '/regRedChan.tiff']); 

histology_ccf=struct;
atlas2histology_tform = cell(size(reg,3),1);
for iSlice = 1:size(reg,3)
    histology_ccf(iSlice).plane_ap = 1320 - (1320 - cropAllenLimits(2)) - repmat((iSlice-1)*2, [size(reg,1), size(reg,2)]);%other way? %bregma ?? 
    histology_ccf(iSlice).plane_ml = repmat(1:2:1140, [size(reg,1),1]);
    histology_ccf(iSlice).plane_dv = repmat(1:2:800, [ size(reg,2),1])';
    histology_ccf(iSlice).tv_slices = squeeze(tv((iSlice-1)*2 + cropAllenLimits(1),:,:));
    histology_ccf(iSlice).av_slices = squeeze(av((iSlice-1)*2 + cropAllenLimits(1),:,:));
    atlas2histology_tform{iSlice} = [1 0 0; 0 1 0; 0 0 1]; % no scaling 
end
mkdir([extraHDPath filesep animal '/slices'])
save([extraHDPath filesep animal '/slices/histology_ccf.mat'], 'histology_ccf','-v7.3')
save([extraHDPath filesep animal '/slices/atlas2histology_tform.mat'], 'atlas2histology_tform')

%% draw probes 
regRed = loadtiff([extraHDPath filesep animal '/regRedChan.tiff']);
AP_get_probe_histologyJF(tv, av, st, slice_path,'rocksaw',regRed); % shift+1 = 11, alt+1 = 21, ctrl+1=31, .. 
set(gcf, 'Color', 'black')
im_path = [extraHDPath filesep animal];

%% align ephys depth
%% BLUE, ORANGE, YELLOW, PURPLE, GREEN , LIGHT BLUE, RED correspondance (manual):
probe2ephys = struct;
probe2ephys.animal = animal;
probe2ephys(1).day = 1;
probe2ephys(1).site = 1;
probe2ephys(1).shank = 4;

probe2ephys(2).day = 1;
probe2ephys(2).site = 1;
probe2ephys(2).shank = 3;

probe2ephys(3).day =1;
probe2ephys(3).site = 1;
probe2ephys(3).shank = 2;

probe2ephys(4).day = 1;
probe2ephys(4).site = 1;
probe2ephys(4).shank = 1;

probe2ephys(5).day = 3;
probe2ephys(5).site = 1;
probe2ephys(5).shank = NaN;

probe2ephys(6).day = 6;
probe2ephys(6).site = 1;
probe2ephys(6).shank = NaN;

probe2ephys(7).day = 7;
probe2ephys(7).site = 1;
probe2ephys(7).shank = NaN;

probe2ephys(8).day = 5;
probe2ephys(8).site = 2;
probe2ephys(8).shank = 4;

probe2ephys(9).day = 5;
probe2ephys(9).site = 2;
probe2ephys(9).shank = 3;

probe2ephys(10).day = 5;
probe2ephys(10).site = 2;
probe2ephys(10).shank = 2;

probe2ephys(11).day = 5;
probe2ephys(11).site = 2;
probe2ephys(11).shank = 1;

probe2ephys(12).day = 4;
probe2ephys(12).site = 2;
probe2ephys(12).shank = 4;

probe2ephys(13).day = 4;
probe2ephys(13).site = 2;
probe2ephys(13).shank = 3;

probe2ephys(14).day = 4;
probe2ephys(14).site = 2;
probe2ephys(14).shank = 2;

probe2ephys(15).day = 4;
probe2ephys(15).site = 2;
probe2ephys(15).shank = 1;

probe2ephys(16).day = 2;
probe2ephys(16).site = 2;
probe2ephys(16).shank = 4;

probe2ephys(17).day = 2;
probe2ephys(17).site = 2;
probe2ephys(17).shank = 3;

probe2ephys(18).day = 2;
probe2ephys(18).site = 2;
probe2ephys(18).shank = 2;

probe2ephys(19).day = 2;
probe2ephys(19).site = 2;
probe2ephys(19).shank = 1;

probe2ephys(20).day = 3;
probe2ephys(20).site = 2;
probe2ephys(20).shank = 4;

probe2ephys(21).day = 3;
probe2ephys(21).site = 2;
probe2ephys(21).shank = 3;

probe2ephys(22).day = 3;
probe2ephys(22).site = 2;
probe2ephys(22).shank = 2;

probe2ephys(23).day = 3;
probe2ephys(23).site = 2;
probe2ephys(23).shank = 1;

probe2ephys(24).day = 1;
probe2ephys(24).site = 2;
probe2ephys(24).shank = 4;

probe2ephys(25).day = 1;
probe2ephys(25).site = 2;
probe2ephys(25).shank = 3;

probe2ephys(26).day = 1;
probe2ephys(26).site = 2;
probe2ephys(26).shank = 2;

probe2ephys(27).day = 1;
probe2ephys(27).site = 2;
probe2ephys(27).shank = 1;

probe2ephys(28).day = 4;
probe2ephys(28).site = 2;
probe2ephys(28).shank = 3;

probe2ephys(29).day = 4;
probe2ephys(29).site = 2;
probe2ephys(29).shank = 4;

save([im_path, '/probe2ephys.mat'], 'probe2ephys')

%% align ephys
% Align histology to electrophysiology %% NOTE: YOU HAVE TO DO THESE ONE BY
% ONE RN (NOT IN LOOP) FOR THINGS TO SAVE PROPERLY
im_path = [extraHDPath filesep animal];
load([im_path, '/probe2ephys.mat'])
load([im_path, '/slices/probe_ccf.mat'])
dontAnalyze = 0;
iProbe=0
iProbe =9
%for iProbe = 1:size(probe2ephys, 2)
keep st probe2ephys tv av animal iProbe slice_path im_path
    use_probe = iProbe;
    corona = 0;
    protocol = 'oice'; %protocol common to all sites and days
    experiments = AP_find_experimentsJF(animal, protocol, protocol);
    experiments = experiments([experiments.ephys]);
    curr_day = probe2ephys(iProbe).day;
    if isfield(probe2ephys, 'shank')
        curr_shank = probe2ephys(iProbe).shank;
    else
        curr_shank = NaN;
    end
    %if ~isempty(curr_day)
    day = experiments(curr_day).day;
    experiment = experiments(curr_day).experiment; % experiment number
    if length(experiment)>1
        experiment = experiment(2);
    end
    %experiment = experiment(probe2ephys(iProbe).site);
    verbose = false; % display load progress and some info figures
    load_parts.cam = false;
    load_parts.imaging = false;
    load_parts.ephys = true;
    site = probe2ephys(iProbe).site;
    
    %experiment=[];
    
    lfp_channel = 'all'; % load lfp 
    loadClusters = 0;
    recording = [];
    [ephysAPfile,aa] = AP_cortexlab_filenameJF(animal,day,experiment,'ephys_ap',site,recording);
    isSpikeGlx = contains(ephysAPfile, '_g');%spike glx (2.0 probes) or open ephys (3A probes)? 
    if isSpikeGlx
         [ephysKSfile,~] = AP_cortexlab_filenameJF(animal,day,experiment,'ephys',site,recording);
        if isempty(dir([ephysKSfile filesep 'sync.mat']))
            sync = syncFT(ephysAPfile, 385, ephysKSfile);
        end
        loadLFP=0;
    else
        loadLFP=1;
    end
    loadLFP=0;
    recording=[];
    experiment=1
    
    AP_load_experimentJF;

    %if dontAnalyze == 0
        %AP_cellrasterJF({stimOn_times}, {stimIDs})
        
        
        AP_cellrasterJF({stimOn_times,wheel_move_time,signals_events.responseTimes(n_trials(1):n_trials(end))',signals_events.responseTimes(n_trials(1):n_trials(end))'}, ...
            {trial_conditions(:,1),trial_conditions(:,2), ...
            trial_conditions(:,3),trial_outcome});
        lfp=nan;
        AP_align_probe_histologyJF(st, slice_path, ...
            spike_times, spike_templates, template_depths, spike_xdepths,template_xdepths,...
            lfp, lfp_channel_positions, lfp_channel_xpositions, ...
            use_probe,isSpikeGlx, curr_shank);
    else
        experiment = experiments(curr_day).experiment; % experiment number
        experiment = experiment(probe2ephys(iProbe).site) + 1;
        verbose = false; % display load progress and some info figures
        load_parts.cam = false;
        load_parts.imaging = false;
        load_parts.ephys = true;
        site = probe2ephys(iProbe).site;
        lfp_channel = 'all';
        try
            AP_load_experimentJF;
            AP_align_probe_histologyJF(st, slice_path, ...
                spike_times, spike_templates, template_depths, ...
                lfp, channel_positions(:, 2), ...
                use_probe);
        catch
            experiment = experiments(curr_day).experiment; % experiment number
            experiment = experiment(probe2ephys(iProbe).site) + 2;
            verbose = false; % display load progress and some info figures
            load_parts.cam = false;
            load_parts.imaging = false;
            load_parts.ephys = true;
            site = probe2ephys(iProbe).site;
            lfp_channel = 'all';
            AP_load_experimentJF;
            AP_align_probe_histologyJF(st, slice_path, ...
                spike_times, spike_templates, template_depths, ...
                lfp, channel_positions(:, 2), ...
                use_probe);
        end
    end
    %end

    %     indexes = probe_ccf .* 10;
    %     av(indexes)
    %     av(round(probe_ccf(:, 1)), probe_ccf(:, 2), probe_ccf(:, 3))
%end

%% save on server

im_path = [extraHDPath filesep animal];
locationHisto = ['/home/netshare/tempserver/', animal, '/Histology']; % copy files over to local disk
mkdir([locationHisto, '/processed/']);
copyfile(im_path, [locationHisto, '/processed/']);
