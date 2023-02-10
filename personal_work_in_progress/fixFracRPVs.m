% fix previously computed frac rpvs 

qMetric = parquetread([fullfile(savePath, 'templates._bc_qMetrics.parquet')]);
frpv = table2array(parquetread([fullfile(savePath, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet')]));
tauR_est = qMetric.RPV_tauR_estimate;
qMetric.fractionRPVs_estimatedTauR = arrayfun(@(x) frpv(x, tauR_est(x,1)), 1:size(frpv,1))';
parquetwrite([fullfile(savePath, 'templates._bc_qMetrics.parquet')], qMetric);
