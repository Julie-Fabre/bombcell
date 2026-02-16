function saveMethodsText(param, savePath, varargin)
% bc.qm.saveMethodsText - Save methods text and BibTeX to files.
%
% Writes a .txt file containing the methods paragraph and formatted
% references, and a .bib file containing BibTeX entries.
%
% ------
% Inputs
% ------
% param : struct
%   The BombCell parameter structure (from bc.qm.qualityParamValues).
% savePath : char or string
%   Path to the output .txt file. A .bib file will be created alongside
%   with the same base name.
%
% Optional name-value pairs (passed to bc.qm.generateMethodsText):
%   'citationStyle' : char, 'inline' (default) or 'numbered'
%   'qualityMetrics' : struct or table, default []
%
% ------
% Example
% ------
%   param = bc.qm.qualityParamValues(ephysMetaDir, rawFile, ksPath);
%   bc.qm.saveMethodsText(param, '/path/to/bombcell_methods.txt');
%   % Creates: bombcell_methods.txt and bombcell_methods.bib

    [text, refs, bib] = bc.qm.generateMethodsText(param, varargin{:});

    sep = repmat('=', 1, 70);

    % Write .txt file
    fid = fopen(savePath, 'w');
    if fid == -1
        error('Could not open file for writing: %s', savePath);
    end
    fprintf(fid, 'METHODS\n%s\n%s\n\n', sep, text);
    fprintf(fid, 'REFERENCES\n%s\n', sep);
    for i = 1:numel(refs)
        fprintf(fid, '  %s\n', refs{i});
    end
    fclose(fid);

    % Write .bib file
    [folder, name, ~] = fileparts(savePath);
    bibPath = fullfile(folder, [name '.bib']);
    fid = fopen(bibPath, 'w');
    if fid == -1
        error('Could not open file for writing: %s', bibPath);
    end
    for i = 1:numel(bib)
        fprintf(fid, '%s\n\n', bib{i});
    end
    fclose(fid);

    fprintf('Methods text saved to: %s\n', savePath);
    fprintf('BibTeX entries saved to: %s\n', bibPath);

end
