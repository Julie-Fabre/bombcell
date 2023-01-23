function bc_saveQMetrics(param, qMetric, forGUI, savePath)
% JF, Save quality metrics
% ------
% Inputs
% ------
% 
% ------
% Outputs
% ------
if ~exist(savePath, 'dir')
    mkdir(fullfile(savePath))
end

% save parameters
parquetwrite([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')], struct2table(param, 'AsArray', 1))

% save quality metrics 
if param.saveAsMat
    save(fullfile(savePath, 'templates.qualityMetricDetailsforGUI.mat'), 'forGUI', '-v7.3')
end

if param.saveAsParquet
    % save fraction refractory period violations for all different tauR times 
    writeNPY(qMetric.fractionRPVs, [fullfile(savePath, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.npy')])
    qMetric.fractionRPVs = qMetric.fractionRPVs(qMetric.RPV_tauR_estimate)';

    % save the rest of quality metrics and fraction refractory period
    % violations for each unit's estimated tauR
    parquetwrite([fullfile(savePath, 'templates._bc_qMetrics.parquet')], struct2table(param, 'AsArray', 1))

end
end