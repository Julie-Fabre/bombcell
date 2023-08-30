function qMetric = bc_saveQMetrics(param, qMetric, forGUI, savePath)
% JF, Reformat and save ephys properties
% ------
% Inputs
% ------
% paramEP: matlab structure defining extraction and classification parameters 
%   (see bc_ephysProperties Values for required fields
%   and suggested starting values)
% ephysProperties: matlab structure computed in the main loop of
%   bc_computeAllEphysProperties
% savePath: character array defining the path where you want to save your
%   quality metrics and parameters 
% ------
% Outputs
% ------
% ephysProperties: reformated ephysProperties structure into a table array

if ~exist(savePath, 'dir')
    mkdir(fullfile(savePath))
end

% save parameters
if ~istable(param)
    parquetwrite([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')], struct2table(param))
end
% save quality metrics
if param.saveMatFileForGUI
    save(fullfile(savePath, 'templates.qualityMetricDetailsforGUI.mat'), 'forGUI', '-v7.3')
end

% save fraction refractory period violations for all different tauR times
parquetwrite([fullfile(savePath, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet')], array2table(qMetric.fractionRPVs))
qMetric.fractionRPVs_estimatedTauR = arrayfun(@(x) qMetric.fractionRPVs(x, qMetric.RPV_tauR_estimate(x)), 1:size(qMetric.fractionRPVs,1));
qMetric = rmfield(qMetric, 'fractionRPVs');

% save the rest of quality metrics and fraction refractory period
% violations for each unit's estimated tauR
% make sure everything is a double first
FNames = fieldnames(qMetric);
for fid = 1:length(FNames)
    eval(['qMetric.', FNames{fid}, '=double(qMetric.', FNames{fid}, ');'])
end
qMetricArray = double(squeeze(reshape(table2array(struct2table(qMetric, 'AsArray', true)), size(qMetric.maxChannels, 2), ...
    length(fieldnames(qMetric)))));
qMetricTable = array2table(qMetricArray);
qMetricTable.Properties.VariableNames = fieldnames(qMetric);

parquetwrite([fullfile(savePath, 'templates._bc_qMetrics.parquet')], qMetricTable)

% overwrite qMetric with the table, to be consistent with it for next steps
% of the pipeline
qMetric = qMetricTable;

% optionally, also save output as a .tsv file
if isfield(param,'saveAsTSV') % ensure back-compatibility if users have a previous version of param
    fieldsToSave = {'percentageSpikesMissing_gaussian', ...
        'presenceRatio', 'maxDriftEstimate', 'nPeaks', 'nTroughs', 'isSomatic','waveformDuration_peakTrough', 'spatialDecaySlope','waveformBaselineFlatness',...
        'signalToNoiseRatio','fractionRPVs_estimatedTauR' };
    fieldsRename = {'%_spikes_missing', 'presence_ratio', 'max_drift', 'n_peaks', 'n_troughs', ...
        'is_somatic','waveform_dur', 'spatial_decay_slope','wv_baseline_flatness',...
        'SNR','frac_RPVs' };
    if param.saveAsTSV == 1 
        cluster_id_vector = qMetric.clusterID - 1; % from bombcell to phy nomenclature 
        if isfield(param,'ephysKilosortPath')
            saveTSV_path = param.ephysKilosortPath;
        else
            saveTSV_path = savePath;
        end
        for fid = 1:length(fieldsToSave)
            cluster_table = table(cluster_id_vector, qMetric.(fieldsToSave{fid}), 'VariableNames', {'cluster_id', fieldsRename{fid}});
            writetable(cluster_table,[saveTSV_path filesep 'cluster_' fieldsRename{fid} '.tsv'],'FileType', 'text','Delimiter','\t');  
        end
    end
end

end