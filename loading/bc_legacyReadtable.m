function t = bc_legacyReadtable(filename,args)
%legacy implementation of READTABLE

% Copyright 2019-2021 The MathWorks, Inc.

try
    %Make readtable and writetable accept strings (shallow)
    [filename, args{:}] = convertStringsToChars(filename, args{:});
    
    pnames = {'FileType','VariableNamingRule'};
    dflts =  {       [] ,'modify'};
    [fileType,vnr,supplied,otherArgs] = matlab.internal.datatypes.parseArgs(pnames, dflts, args{:});
    
    if supplied.VariableNamingRule
        vnr = validatestring(vnr,["modify","preserve"]);
        otherArgs(end+(1:2)) = {'preserveVariableNames',(vnr=="preserve")};
    end

    if ~supplied.FileType
        [~,~,fx] = fileparts(filename);
        switch lower(fx)
        case {'.txt' '.dat' '.csv' '.tsv' '.htsv'}, fileType = 'text';
        case {'.xls' '.xlsx' '.xlsb' '.xlsm' '.xltm' '.xltx' '.ods'}, fileType = 'spreadsheet';
        case '', fileType = 'text';
        case cellstr([matlab.io.internal.FileExtensions.XMLExtensions,...
            matlab.io.internal.FileExtensions.HTMLExtensions,...
            matlab.io.internal.FileExtensions.WordDocumentExtensions])
            error(message('MATLAB:readtable:LegacyFormatFileTypes'));
        otherwise
            error(message('MATLAB:readtable:UnrecognizedFileExtension',fx));
        end
    elseif ~ischar(fileType) && ~(isstring(fileType)&&isscalar(fileType))
        error(message('MATLAB:textio:textio:InvalidStringProperty','FileType'));
    else
        fileTypes = {'text' 'spreadsheet'};
        itype = find(strncmpi(fileType,fileTypes,strlength(fileType)));
        if isempty(itype)
            if any(strcmpi(fileType,{'xml','html','xhtml','worddocument'}))
                error(message('MATLAB:readtable:LegacyFormatFileTypes'));
            end
            error(message('MATLAB:readtable:UnrecognizedFileType',fileType));
        elseif ~isscalar(itype)
            error(message('MATLAB:readtable:AmbiguousFileType',fileType));
        end
        fileType = fileTypes{itype};
    end

    % readTextFile and readXLSFile will add an extension if need be, no need to add one here.
    remote2Local = matlab.io.internal.vfs.stream.RemoteToLocal(filename);
    filename = remote2Local.LocalFileName;

    switch lower(fileType)
    case 'text'
        t = matlab.io.internal.readTextFile(filename,otherArgs);
    case 'spreadsheet'
        idx = [];
        for i = 1:2:(numel(otherArgs)-1)
            if strcmpi(otherArgs{i},"Format") && strcmpi(otherArgs{i+1},'auto')
                idx = [idx;i;i+1];  %#ok<AGROW>
            end
        end
        otherArgs(idx) = []; % Remove 'Format','auto' from the list
        t = matlab.io.internal.readXLSFile(filename,otherArgs);
    end
catch ME
    throwAsCaller(ME)
end
