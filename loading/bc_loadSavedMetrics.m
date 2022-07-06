if ~isempty(dir(fullfile(savePath, 'qMetric*.mat')))
    load(fullfile(savePath, 'qMetric.mat'))
    load(fullfile(savePath, 'param.mat'))
else
    
end
T = struct2table(qMetric, 'AsArray', true);
bc_writetable(T,'tabledata2.htsv','Delimiter','\t')
readtable('tabledata2.htsv')
%writeNPY(qMetric,'tabledata2.npy')
