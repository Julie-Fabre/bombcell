if ~isempty(dir(fullfile(savePath, 'qMetric*.mat')))
    load(fullfile(savePath, 'qMetric.mat'))
elseif ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')))
    qMetric = parquetread(fullfile(savePath, 'templates._bc_qMetrics.parquet'));
end

param = parquetread([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')]);
