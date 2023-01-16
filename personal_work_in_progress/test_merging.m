% feature view, run bdscan on non-time projections, plot unit amplitude x
% time, waveforms, CCG 

test_data_location = '/media/julie/Elements/test_dataset/acute';
manually_split_units = [286,287; 290,291; 294,295; 304,305; 307,306;...
    321,322; 324,325; 336,337; 326,327; 329,330; 334,333;338,339;340,341;342,343;344,345;356,357];

pc_features = readNPY([test_data_location, filesep, 'pc_features.npy']);
pc_features_ind = readNPY([test_data_location, filesep, 'pc_feature_ind.npy']);
spike_amplitudes = readNPY([test_data_location, filesep, 'amplitudes.npy']);
spike_templates = readNPY([test_data_location, filesep, 'spike_clusters.npy']);
spike_times = readNPY([test_data_location, filesep, 'spike_times.npy']);
waveforms = readNPY([test_data_location, filesep, '_phy_spikes_subset.waveforms.npy']);
spike_waveforms = readNPY([test_data_location, filesep, '_phy_spikes_subset.spikes.npy']);
% param.rawFolder = test_data_location;
% param.nChannels = 385;
% param.nRawSpikesToExtract = 100;
% 
% [rawWaveformsFull, rawWaveformsPeakChan] = bc_extractRawWaveformsFast(param, ...
%     spike_times, spike_templates, 0, 1);

for iSplit = 1:size(manually_split_units,1)
    unit1 = manually_split_units(iSplit,1);
    unit2 = manually_split_units(iSplit,2);
    figure();
    
    % Feature view (exlcuding time)                     
    subplot(2,3,1)
    scatter(pc_features(spike_templates==unit1,1,1),...
        pc_features(spike_templates==unit1,2,1), 2, 'filled');
    hold on;
    scatter(pc_features(spike_templates==unit2,1,1),...
        pc_features(spike_templates==unit2,2,1), 2, 'filled');
   

    subplot(2,3,2)
    scatter(pc_features(spike_templates==unit1,1,1),...
        pc_features(spike_templates==unit1,1,2), 2, 'filled');
    hold on;
    scatter(pc_features(spike_templates==unit2,1,1),...
        pc_features(spike_templates==unit2,1,2), 2, 'filled');
    
    subplot(2,3,3)
    scatter(pc_features(spike_templates==unit1,1,1),...
        pc_features(spike_templates==unit1,2,2), 2, 'filled');
    hold on;
    scatter(pc_features(spike_templates==unit2,1,1),...
        pc_features(spike_templates==unit2,2,2), 2, 'filled');

    subplot(2,3,4)
    scatter(pc_features(spike_templates==unit1,2,1),...
        pc_features(spike_templates==unit1,1,2), 2, 'filled'); hold on;
    scatter(pc_features(spike_templates==unit2,2,1),...
        pc_features(spike_templates==unit2,1,2), 2, 'filled');
    
    subplot(2,3,5)
    scatter(pc_features(spike_templates==unit1,2,1),...
        pc_features(spike_templates==unit1,2,2), 2, 'filled'); hold on;
    scatter(pc_features(spike_templates==unit2,2,1),...
        pc_features(spike_templates==unit2,2,2), 2, 'filled');
    
    subplot(2,3,6)
    scatter(pc_features(spike_templates==unit1,1,2),...
        pc_features(spike_templates==unit1,2,2), 2, 'filled'); hold on;
     scatter(pc_features(spike_templates==unit2,1,2),...
        pc_features(spike_templates==unit2,2,2), 2, 'filled');


end
 


% try to cluster 
views_for_cluster = [1,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1];
view_indexes = [1,1,2,1;1,1,1,2];
for iSplit = 1:size(manually_split_units,1)
    %iSplit=iSplit+1
    unit1 = manually_split_units(iSplit,1);
    unit2 = manually_split_units(iSplit,2);
    

    split_matrix = [pc_features(spike_templates==unit1,...
        view_indexes(views_for_cluster(iSplit),1),view_indexes(views_for_cluster(iSplit),2)), ...
        pc_features(spike_templates==unit1,...
        view_indexes(views_for_cluster(iSplit),3),view_indexes(views_for_cluster(iSplit),4));...
        pc_features(spike_templates==unit2,...
        view_indexes(views_for_cluster(iSplit),1),view_indexes(views_for_cluster(iSplit),2)), ...
        pc_features(spike_templates==unit2,...
        view_indexes(views_for_cluster(iSplit),3),view_indexes(views_for_cluster(iSplit),4))];
    
    figure();
    
    % Feature view (exlcuding time)                     
    subplot(1,2,1)
    title('manual clustering'); hold on;
    scatter(pc_features(spike_templates==unit1,...
        view_indexes(views_for_cluster(iSplit),1),view_indexes(views_for_cluster(iSplit),2)), ...
        pc_features(spike_templates==unit1,...
        view_indexes(views_for_cluster(iSplit),3),view_indexes(views_for_cluster(iSplit),4)), 5, 'filled');
    hold on;
      scatter(pc_features(spike_templates==unit2,...
        view_indexes(views_for_cluster(iSplit),1),view_indexes(views_for_cluster(iSplit),2)), ...
        pc_features(spike_templates==unit2,...
        view_indexes(views_for_cluster(iSplit),3),view_indexes(views_for_cluster(iSplit),4)), 5, 'filled');
  xlabel('Ch1, PC1')
  ylabel('Ch2, PC1')

    subplot(1,2,2)
    title('DBSCAN clustering'); hold on;
    idx = dbscan(split_matrix,1,130);
    gscatter(split_matrix(:,1),split_matrix(:,2),idx);
xlabel('Ch1, PC1')
  ylabel('Ch2, PC1')


end
 
