function printMethodsText(param, varargin)
% bc.qm.printMethodsText - Print generated methods text and references to the command window.
%
% ------
% Inputs
% ------
% param : struct
%   The BombCell parameter structure (from bc.qm.qualityParamValues).
%
% Optional name-value pairs (passed to bc.qm.generateMethodsText):
%   'citationStyle' : char, 'inline' (default) or 'numbered'
%   'qualityMetrics' : struct or table, default []
%
% ------
% Example
% ------
%   param = bc.qm.qualityParamValues(ephysMetaDir, rawFile, ksPath);
%   bc.qm.printMethodsText(param);
%   bc.qm.printMethodsText(param, 'citationStyle', 'numbered');

    [text, refs, bib] = bc.qm.generateMethodsText(param, varargin{:});

    sep = repmat('=', 1, 70);
    fprintf('\n%s\nMETHODS\n%s\n', sep, sep);
    fprintf('%s\n', text);
    fprintf('\n%s\nREFERENCES\n%s\n', sep, sep);
    for i = 1:numel(refs)
        fprintf('  %s\n', refs{i});
    end
    fprintf('\n%s\nBIBTEX\n%s\n', sep, sep);
    for i = 1:numel(bib)
        fprintf('%s\n\n', bib{i});
    end

end
