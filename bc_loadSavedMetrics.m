if ~isempty(dir(fullfile(savePath, 'qMetric*.mat')))
    load(fullfile(savePath, 'qMetric.mat'))
    if ~isempty(dir(fullfile(savePath, 'templates._jf_qMetrics.parquet')))
        qMetric = parquetread(fullfile(savePath, 'templates._jf_qMetrics.parquet'));
    end
    try
        load(fullfile(savePath, 'param.mat'))
    catch
        param = parquetread([fullfile(savePath, '_jf_parameters._jf_qMetrics.parquet')]);
    end
    
end