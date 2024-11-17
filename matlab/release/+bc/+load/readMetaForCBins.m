function [meta] = readMetaForCBins(binName,aporlfp)

if nargin<2
    aporlfp='ap';
end
% Create the matching metafile name
file = dir(fullfile(binName,['*' aporlfp '.meta']));

if isempty(file)
    file = dir(fullfile(binName,['*.meta']));
end

% Parse ini file into cell entries C{1}{i} = C{2}{i}
fid = fopen(fullfile(file(1).folder, file(1).name), 'r');
% -------------------------------------------------------------
%    Need 'BufSize' adjustment for MATLAB earlier than 2014
%    C = textscan(fid, '%[^=] = %[^\r\n]', 'BufSize', 32768);
C = textscan(fid, '%[^=] = %[^\r\n]');
% -------------------------------------------------------------
fclose(fid);

% New empty struct
meta = struct();

% Convert each cell entry into a struct entry
for i = 1:length(C{1})
    tag = C{1}{i};
    if tag(1) == '~'
        % remake tag excluding first character
        tag = sprintf('%s', tag(2:end));
    end
    meta = setfield(meta, tag, C{2}{i});
end
end % ReadMeta