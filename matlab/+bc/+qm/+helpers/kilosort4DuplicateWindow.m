function duplicateWindow_s = kilosort4DuplicateWindow(kilosortPath, defaultMs)
% KILOSORT4DUPLICATEWINDOW Kilosort 4's duplicate spike window
% (settings duplicate_spike_ms) in seconds, read from the kilosort4.log it
% saves alongside its output. Falls back to Kilosort 4's default of 0.25 ms
% if the log cannot be read, and returns 0 if it was written by a Kilosort 4
% version that does not remove duplicate spikes.
% (The python version reads ops.npy, which holds the same settings.)
if nargin < 2
    defaultMs = 0.25;
end
duplicateWindow_s = defaultMs / 1000;
if isempty(kilosortPath) || (isstring(kilosortPath) && ismissing(kilosortPath))
    return;
end
logFile = fullfile(char(kilosortPath), 'kilosort4.log');
if ~isfile(logFile)
    return;
end
try
    logText = fileread(logFile);
catch
    return;
end
tokens = regexp(logText, '''duplicate_spike_ms''\s*:\s*([0-9.eE+-]+)', 'tokens');
if ~isempty(tokens)
    duplicateWindow_s = str2double(tokens{end}{1}) / 1000; % last run in the log
else
    duplicateWindow_s = 0; % a Kilosort 4 version without duplicate removal
end
end
