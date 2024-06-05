function bc_updateSpatialDecaySlope(kilosortSavePath, qMetricSavePath)

% load in quality metrics 
[~, qMetric] = bc_loadSavedMetrics(qMetricSavePath); 

% load in relevant kilosort files 
[~, spikeTemplates, templateWaveforms, ~, ~, ...
    ~, ~] = bc_loadEphysData(kilosortSavePath);

uniqueTemplates = unique(spikeTemplates);

% update spatial decay value
for iUnit = 1:size(uniqueTemplates,1)

    thisUnit = uniqueTemplates(iUnit);

    qMetric.spatialDecaySlope(iUnit) = qMetric.spatialDecaySlope(iUnit)./max(max(templateWaveforms(thisUnit, :,:)));
end

% save new metrics 
parquetwrite([fullfile(qMetricSavePath, 'templates._bc_qMetrics.parquet')], qMetric)
end