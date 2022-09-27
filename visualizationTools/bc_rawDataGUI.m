
%% raw data GUI

function rawDataGuiHandle = bc_rawDataGUI(memMapData, ephysData)

% if isempty(memMapData)
%     display('Extract data on the fly using python. You need Matlab version 2022a or higher, and have it pointed to the correct (Anaconda) environment')
% end

%% set up dynamic figure
rawDataGuiHandle = figure('color', 'w');
set(rawDataGuiHandle, 'KeyPressFcn', @KeyPressCb);

%% initial conditions
iCluster = 1;
uniqueTemps = unique(ephysData.spike_templates);

%% plot initial conditions
iChunk = 1;
iChannelChunk = 1;
initializePlot(rawDataGuiHandle)

%% change on keypress
    function KeyPressCb(~, evnt)
        %fprintf('key pressed: %s\n', evnt.Key);
        if strcmpi(evnt.Key, 'leftarrow')
            iChunk = iChunk + 1;
            if iChunk > length(ephysData.spike_times_timeline(ephysData.spike_templates == iCluster))
                iChunk = 1;
            end
            updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, iCluster, ...
                uniqueTemps, iChunk, iChannelChunk);
        elseif strcmpi(evnt.Key, 'rightarrow')
            iChunk = iChunk - 1;
            if iChunk == 0
                iChunk = length(ephysData.spike_times_timeline(ephysData.spike_templates == iCluster));
            end
            updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, iCluster, ...
                uniqueTemps, iChunk, iChannelChunk);
        elseif strcmpi(evnt.Key, 'uparrow')
            iChannelChunk = iChannelChunk + 1;
            if iChannelChunk > floor(384/20)
                iChannelChunk = iChannelChunk - 1;
            end
            updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, iCluster, ...
                uniqueTemps, iChunk, iChannelChunk);

        elseif strcmpi(evnt.Key, 'downarrow')
            iChannelChunk = iChannelChunk - 1;
            if iChannelChunk < 1
                iChannelChunk = iChannelChunk + 1;
            end
            updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, iCluster, ...
                uniqueTemps, iChunk, iChannelChunk);
        end
    end
end

function initializePlot(rawDataGuiHandle)

%% main title

mainTitle = sgtitle('');

%% initialize raw data

rawPlotH = subplot(1, 1, 1);
hold on;
title('Raw unwhitened data')
set(rawPlotH, 'XColor', 'w', 'YColor', 'w')
rawPlotLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
rawSpikeLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'b'), 1:max_n_channels_plot);

%% save all handles
guiData = struct;
% main title
guiData.mainTitle = mainTitle;

% raw data

guiData.rawPlotH = rawPlotH;
guiData.rawPlotLines = rawPlotLines;
guiData.rawSpikeLines = rawSpikeLines;

% upload guiData
guidata(rawDataGuiHandle, guiData);
end

function updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, iCluster, ...
    uniqueTemps, iChunk, iChannelChunk)

guiData = guidata(rawDataGuiHandle);

%% 8. plot raw data

plotSubRaw(guiData.rawPlotH, guiData.rawPlotLines, guiData.rawSpikeLines, memMapData, ephysData, iCluster, uniqueTemps, iChunk, iChannelChunk);


end


function plotSubRaw(rawPlotH, rawPlotLines, rawSpikeLines, memMapData, ephysData, iCluster, uniqueTemps, iChunk, iChannelChunk)
%get the used channels
theseChannels = (iChannelChunk - 1)* 20 + 1: (iChannelChunk )* 20;
% get units with max channel displayed
chanAmps = squeeze(max(ephysData.templates(iCluster, :, :))-min(ephysData.templates(iCluster, :, :)));
maxChan = find(chanAmps == max(chanAmps), 1);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = ((ephysData.channel_positions(:, 1) - maxXC).^2 ...
    +(ephysData.channel_positions(:, 2) - maxYC).^2).^0.5;
chansToPlot = find(chanDistances < 100);

%get spike locations
timeToPlot = 0.1;
pull_spikeT = -40:41;
thisC = uniqueTemps(iCluster);
theseTimesCenter = ephysData.spike_times(ephysData.spike_templates == thisC) ./ ephysData.ephys_sample_rate;
if iChunk < 0
    disp('Don''t do that')
    iChunk = 1;
end
if length(theseTimesCenter) > 10 + iChunk
    firstSpike = theseTimesCenter(iChunk+10) - 0.05; %tenth spike occurance %
else
    firstSpike = theseTimesCenter(iChunk) - 0.05; %first spike occurance
end
% Not sure why this was +10?
theseTimesCenter = theseTimesCenter(theseTimesCenter >= firstSpike);
theseTimesCenter = theseTimesCenter(theseTimesCenter <= firstSpike+timeToPlot);
if ~isempty(theseTimesCenter)
    %theseTimesCenter=theseTimesCenter(1);
    theseTimesFull = theseTimesCenter * ephysData.ephys_sample_rate + pull_spikeT;
    %theseTimesFull=unique(sort(theseTimesFull));
end

cCount = cumsum(repmat(1000, size(chansToPlot, 1), 1), 1);


t = int32(firstSpike*ephysData.ephys_sample_rate):int32((firstSpike + timeToPlot)*ephysData.ephys_sample_rate);
subplot(rawPlotH)
plotidx = int32(firstSpike*ephysData.ephys_sample_rate) ...
    :int32((firstSpike + timeToPlot)*ephysData.ephys_sample_rate);
t(plotidx < 1 | plotidx > size(memMapData, 2)) = [];
plotidx(plotidx < 1 | plotidx > size(memMapData, 2)) = [];
thisMemMap = double(memMapData(chansToPlot, plotidx)) + double(cCount);
for iClear = 1:length(rawSpikeLines)
    set(rawSpikeLines(iClear), 'XData', NaN, 'YData', NaN)
end
if length(rawSpikeLines) < length(chansToPlot)
    rawSpikeLines(end+1:length(chansToPlot)) = rawSpikeLines(end);
    rawPlotLines(end+1:length(chansToPlot)) = rawPlotLines(end);
end
for iChanToPlot = 1:length(chansToPlot)
    set(rawPlotLines(iChanToPlot), 'XData', t, 'YData', thisMemMap(iChanToPlot, :));
    if ~isempty(theseTimesCenter)
        for iTimes = 1:size(theseTimesCenter, 1)
            if ~any(mod(theseTimesFull(iTimes, :), 1))
                set(rawSpikeLines(iChanToPlot), 'XData', theseTimesFull(iTimes, :), 'YData', thisMemMap(iChanToPlot, ...
                    int32(theseTimesFull(iTimes, :))-t(1)));
            end
        end
    end

end


end
