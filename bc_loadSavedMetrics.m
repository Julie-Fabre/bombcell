if ~isempty(dir(fullfile(savePath, 'qMetric*.mat')))
    load(fullfile(savePath, 'qMetric.mat'))
    load(fullfile(savePath, 'param.mat'))
    if ~isempty(dir(fullfile(savePath, 'templates._jf_qMetrics.parquet')))
        parquetread(fullfile(savePath, 'templates._jf_qMetrics.parquet'))
    end
end