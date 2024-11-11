function [param, qMetric, fractionRPVs_allTauR] = loadSavedMetrics(savePath, saveTSV)
% JF, Load saved quality metrics
% ------
% Inputs
% ------
% 
% ------
% Outputs
% ------
qMetric = parquetread(fullfile(savePath, 'templates._bc_qMetrics.parquet'));
fractionRPVs_allTauR = parquetread([fullfile(savePath, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet')]);
param = parquetread([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')]);

% optionally, also save output as a .tsv file
if nargin < 2
    saveTSV = 0; 
end
if isfield(param,'saveAsTSV') % ensure back-compatibility if users have a previous version of param 

    fieldsToSave = {'percentageSpikesMissing_gaussian', ...
        'presenceRatio', 'maxDriftEstimate', 'nPeaks', 'nTroughs','waveformDuration_peakTrough', 'spatialDecaySlope','waveformBaselineFlatness',...
        'signalToNoiseRatio','fractionRPVs_estimatedTauR', 'scndPeakToTroughRatio', 'peak1ToPeak2Ratio', 'mainPeakToTroughRatio', 'isoD', 'Lratio', 'silhouetteScore',...
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