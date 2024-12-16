function [qMetric, param] = prettify_names(qMetric, param)
% Define name mapping


oldNames = {'minTroughToPeakRatio', 'firstPeakRatio', ...
    'minMainPeakToTroughRatio', 'minWidthFirstPeak', ...
    'minWidthMainTrough'};
newNames = {'maxScndPeakToTroughRatio_noise', 'maxPeak1ToPeak2Ratio_nonSomatic', ...
    'maxMainPeakToTroughRatio_nonSomatic', 'minWidthFirstPeak_nonSomatic', ...
    'minWidthMainTrough_nonSomatic'};

if ~isempty(param)
    % Handle param input
    if istable(param)
        % For table format
        for i = 1:length(oldNames)
            if ismember(oldNames{i}, param.Properties.VariableNames)
                param.Properties.VariableNames(strcmp(param.Properties.VariableNames, oldNames{i})) = newNames(i);
            end
        end
    else
        % For struct format
        for i = 1:length(oldNames)
            if isfield(param, oldNames{i})
                param.(newNames{i}) = param.(oldNames{i});
                param = rmfield(param, oldNames{i});
            end
        end
    end
end

if ~isempty(qMetric)
% Handle qMetric input
    if istable(qMetric)
        % Compute additional metrics if mainPeakToTroughRatio doesn't exist
        if ~ismember('mainPeakToTroughRatio', qMetric.Properties.VariableNames)
            qMetric.scndPeakToTroughRatio = abs(qMetric.mainPeak_after_size ./ qMetric.mainTrough_size);
            qMetric.peak1ToPeak2Ratio = abs(qMetric.mainPeak_before_size ./ qMetric.mainPeak_after_size);
            qMetric.mainPeakToTroughRatio = max([qMetric.mainPeak_before_size, qMetric.mainPeak_after_size], [], 2) ./ qMetric.mainTrough_size;
            qMetric.troughToPeak2Ratio = abs(qMetric.mainTrough_size ./ qMetric.mainPeak_before_size);
        end
        
        % Rename columns if they exist
        for i = 1:length(oldNames)
            if ismember(oldNames{i}, qMetric.Properties.VariableNames)
                qMetric.Properties.VariableNames(strcmp(qMetric.Properties.VariableNames, oldNames{i})) = newNames(i);
            end
        end
    else
        % For struct format
        if ~isfield(qMetric, 'mainPeakToTroughRatio')
            qMetric.scndPeakToTroughRatio = abs(qMetric.mainPeak_after_size ./ qMetric.mainTrough_size);
            qMetric.peak1ToPeak2Ratio = abs(qMetric.mainPeak_before_size ./ qMetric.mainPeak_after_size);
            qMetric.mainPeakToTroughRatio = max([qMetric.mainPeak_before_size, qMetric.mainPeak_after_size], [], 2) ./ qMetric.mainTrough_size;
            qMetric.troughToPeak2Ratio = abs(qMetric.mainTrough_size ./ qMetric.mainPeak_before_size);
        end
        
        % Rename fields if they exist
        for i = 1:length(oldNames)
            if isfield(qMetric, oldNames{i})
                qMetric.(newNames{i}) = qMetric.(oldNames{i});
                qMetric = rmfield(qMetric, oldNames{i});
            end
        end
    end
end


end