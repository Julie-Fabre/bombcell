function param = decodeVectorParams(param)
% DECODEVECTORPARAMS Turn numeric vectors that bc.qm.helpers.encodeVectorParams
% saved as text (e.g. '[0.001 0.00125]') back into numbers.
fields = fieldnames(param);
for iField = 1:numel(fields)
    value = param.(fields{iField});
    if iscell(value) && isscalar(value)
        value = value{1};
    end
    if (ischar(value) || (isstring(value) && isscalar(value))) && ...
            ~isempty(regexp(char(value), '^\[[-+0-9.eE ;]*\]$', 'once'))
        param.(fields{iField}) = str2num(char(value)); %#ok<ST2NM> only digits pass the regexp
    end
end
end
