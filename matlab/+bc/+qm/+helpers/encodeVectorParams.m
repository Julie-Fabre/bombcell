function param = encodeVectorParams(param)
% ENCODEVECTORPARAMS parquetwrite cannot store a field holding more than one
% value (e.g. tauR_values), so numeric vectors are saved as their mat2str
% text, e.g. '[0.001 0.00125]'. bc.qm.helpers.decodeVectorParams reverses this.
fields = fieldnames(param);
for iField = 1:numel(fields)
    value = param.(fields{iField});
    if isnumeric(value) && numel(value) > 1
        param.(fields{iField}) = mat2str(double(value(:)'), 17);
    end
end
end
