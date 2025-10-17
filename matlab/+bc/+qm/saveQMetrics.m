function qMetric = saveQMetrics(param, qMetric, forGUI, savePath, medianSpikeDepth, timeBins)
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

% check quality metric and apram names
[qMetric, param] = bc.qm.prettify_names(qMetric, param);

% Save full drift information
if param.computeDrift
    parquetwrite([fullfile(savePath, 'templates._bc_medianSpikeDepth.parquet')], array2table(medianSpikeDepth))
    parquetwrite([fullfile(savePath, 'time_chunks._bc_medianSpikeDepth.parquet')], array2table(timeBins))
end
% Get ratios
invalid_peaks = (qMetric.troughToPeak2Ratio> param.minTroughToPeak2Ratio_nonSomatic | ...
                            qMetric.mainPeak_before_width > param.minWidthFirstPeak_nonSomatic | ...
                            qMetric.mainTrough_width > param.minWidthMainTrough_nonSomatic);
qMetric.peak1ToPeak2Ratio(invalid_peaks) = 0;

if ~exist(savePath, 'dir')
    mkdir(fullfile(savePath))
end

% save parameters
if ~istable(param)
    if ~isfield(param,'ephysKilosortPath') || isempty(param.ephysKilosortPath)
        param.ephysKilosortPath = 'NaN';
    end

    if ~isfield(param,'gain_to_uV') || isempty(param.gain_to_uV)
        param.gain_to_uV = 'NaN';
    end
    parquetwrite([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')], struct2table(param, 'AsArray', true))
end
% save quality metrics
if param.saveMatFileForGUI
    save(fullfile(savePath, 'templates.qualityMetricDetailsforGUI.mat'), 'forGUI', '-v7.3')
end

% compute the waveform ratios 
invalid_peaks = (qMetric.troughToPeak2Ratio > param.minTroughToPeak2Ratio_nonSomatic | ...
                            qMetric.mainPeak_before_width > param.minWidthFirstPeak_nonSomatic | ...
                            qMetric.mainTrough_width > param.minWidthMainTrough_nonSomatic);
qMetric.peak1ToPeak2Ratio(invalid_peaks) = 0;

% save fraction refractory period violations for all different tauR times
parquetwrite([fullfile(savePath, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet')], array2table(qMetric.fractionRPVs))
qMetric.fractionRPVs_estimatedTauR = arrayfun(@(x) qMetric.fractionRPVs(x, qMetric.RPV_window_index(x)), 1:size(qMetric.fractionRPVs,1));
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

% Additionally, save all metrics to a single CSV file
writetable(qMetricTable, fullfile(savePath, 'templates._bc_qMetrics_all.csv'));

% overwrite qMetric with the table, to be consistent with it for next steps
% of the pipeline
qMetric = qMetricTable;


% optionally, also save output as a .tsv file
if isfield(param,'saveAsTSV') % ensure back-compatibility if users have a previous version of param 

    fieldsToSave = {'percentageSpikesMissing_gaussian', ...
        'presenceRatio', 'maxDriftEstimate', 'nPeaks', 'nTroughs','waveformDuration_peakTrough', 'spatialDecaySlope','waveformBaselineFlatness',...
        'signalToNoiseRatio','fractionRPVs_estimatedTauR', 'scndPeakToTroughRatio', 'peak1ToPeak2Ratio', 'mainPeakToTroughRatio', 'isolationDistance', 'Lratio', 'silhouetteScore',...
        'useTheseTimesStart', 'useTheseTimesStop'};
    fieldsRename = {'percentage_spikes_missing', 'presence_ratio', 'max_drift', 'n_peaks', 'n_troughs','waveform_duration', 'spatial_decay_slope', 'wv_baseline_flatness',...
        'SNR','frac_RPVs', 'peak_2_to_trough', 'peak_1_to_peak_2', 'peak_main_to_trough', 'isolation_distance', 'L-ratio', ...
        'silhouette_score', 'good_chunk_start', 'good_chunk_stop'};
    
    indicesToUse = [1:13];
    if param.computeDistanceMetrics 
        indicesToUse = [indicesToUse, 14:16];
    end
    if param.computeDrift
        indicesToUse = [indicesToUse, 17,18];
    end
    if param.computeTimeChunks
        indicesToUse = [indicesToUse, 19,20];
    end
    if param.saveAsTSV == 1 
        cluster_id_vector = qMetric.phy_clusterID; % from bombcell to phy nomenclature 
        if isfield(param,'ephysKilosortPath') && ~strcmp(param.ephysKilosortPath, 'NaN') && ~isempty(param.ephysKilosortPath)
            saveTSV_path = param.ephysKilosortPath;
        else
            saveTSV_path = savePath;
        end
        for fid = 1:length(fieldsToSave)
            if ismember(fid, indicesToUse)
                cluster_table = table(cluster_id_vector, qMetric.(fieldsToSave{fid}), 'VariableNames', {'cluster_id', fieldsRename{fid}});
                writetable(cluster_table,[saveTSV_path filesep 'cluster_' fieldsRename{fid} '.tsv'],'FileType', 'text','Delimiter','\t');  
            else
                if exist([saveTSV_path filesep 'cluster_' fieldsRename{fid} '.tsv'], 'file')
                    delete([saveTSV_path filesep 'cluster_' fieldsRename{fid} '.tsv'])
                end
            end
        end
    end
end

end