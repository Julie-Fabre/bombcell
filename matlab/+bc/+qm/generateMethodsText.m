function [methodsText, references, bibtexEntries] = generateMethodsText(param, varargin)
% bc.qm.generateMethodsText - Generate a methods section paragraph from BombCell parameters.
%
% Generates a ready-to-use methods section paragraph for scientific papers,
% with appropriate citations, based on the parameters used to run BombCell.
%
% ------
% Inputs
% ------
% param : struct
%   The BombCell parameter structure (from bc.qm.qualityParamValues).
%
% Optional name-value pairs:
%   'citationStyle' : char, 'inline' (default) or 'numbered'
%       Citation format. 'inline' produces (Author et al., year),
%       'numbered' produces [1], [2], etc.
%   'qualityMetrics' : struct or table, default []
%       If provided, a summary of unit classification counts is appended.
%
% ------
% Outputs
% ------
% methodsText : string
%   The generated methods paragraph.
% references : cell array of strings
%   Formatted reference list for all cited works.
% bibtexEntries : cell array of strings
%   BibTeX entries for all cited works.
%
% ------
% Example
% ------
%   param = bc.qm.qualityParamValues(ephysMetaDir, rawFile, ksPath);
%   [text, refs, bib] = bc.qm.generateMethodsText(param);
%   disp(text);

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'citationStyle', 'inline', @(x) ismember(x, {'inline', 'numbered'}));
    addParameter(p, 'qualityMetrics', [], @(x) isstruct(x) || istable(x) || isempty(x));
    parse(p, varargin{:});
    citationStyle = p.Results.citationStyle;
    qualityMetrics = p.Results.qualityMetrics;

    % =====================================================================
    % Citation database
    % =====================================================================
    citations = struct();

    citations.bombcell.inline = 'Fabre et al., 2023';
    citations.bombcell.full = ['Fabre, J.M.J., van Beest, E.H., Peters, A.J., Carandini, M., ' ...
        '& Harris, K.D. (2023). Bombcell: automated curation and cell classification of ' ...
        'spike-sorted electrophysiology data. Zenodo. https://doi.org/10.5281/zenodo.8172821'];
    citations.bombcell.bibtex = sprintf(['@misc{fabre2023bombcell,\n' ...
        '  author = {Fabre, Julie M. J. and van Beest, Enny H. and Peters, Andrew J. and Carandini, Matteo and Harris, Kenneth D.},\n' ...
        '  title = {Bombcell: automated curation and cell classification of spike-sorted electrophysiology data},\n' ...
        '  year = {2023},\n' ...
        '  publisher = {Zenodo},\n' ...
        '  doi = {10.5281/zenodo.8172821},\n' ...
        '  url = {https://doi.org/10.5281/zenodo.8172821}\n' ...
        '}']);

    citations.hill2011.inline = 'Hill et al., 2011';
    citations.hill2011.full = ['Hill, D.N., Mehta, S.B., & Kleinfeld, D. (2011). Quality metrics ' ...
        'to accompany spike sorting of extracellular signals. Journal of Neuroscience, 31(24), 8699-8705.'];
    citations.hill2011.bibtex = sprintf(['@article{hill2011quality,\n' ...
        '  author = {Hill, Daniel N. and Mehta, Samar B. and Kleinfeld, David},\n' ...
        '  title = {Quality metrics to accompany spike sorting of extracellular signals},\n' ...
        '  journal = {Journal of Neuroscience},\n' ...
        '  volume = {31},\n' ...
        '  number = {24},\n' ...
        '  pages = {8699--8705},\n' ...
        '  year = {2011}\n' ...
        '}']);

    citations.llobet2022.inline = 'Llobet et al., 2022';
    citations.llobet2022.full = ['Llobet, V., Wyngaard, D., Bhatt, D.K., Marx, M., & Bhatt, S. (2022). ' ...
        'A sliding window framework for optimal refractory period estimation. bioRxiv.'];
    citations.llobet2022.bibtex = sprintf(['@article{llobet2022sliding,\n' ...
        '  author = {Llobet, Vincent and Wyngaard, Darik and Bhatt, Devika K. and Marx, Marius and Bhatt, Shashwat},\n' ...
        '  title = {A sliding window framework for optimal refractory period estimation},\n' ...
        '  journal = {bioRxiv},\n' ...
        '  year = {2022}\n' ...
        '}']);

    citations.deligkaris2016.inline = 'Deligkaris et al., 2016';
    citations.deligkaris2016.full = ['Deligkaris, K., Bullmann, T., & Frey, U. (2016). Extracellularly ' ...
        'recorded somatic and neuritic signal shapes and classification algorithms for high-density ' ...
        'microelectrode array electrophysiology. Frontiers in Neuroscience, 10, 421. ' ...
        'https://doi.org/10.3389/fnins.2016.00421'];
    citations.deligkaris2016.bibtex = sprintf(['@article{deligkaris2016extracellularly,\n' ...
        '  author = {Deligkaris, Kosmas and Bullmann, Torsten and Frey, Urs},\n' ...
        '  title = {Extracellularly recorded somatic and neuritic signal shapes and classification algorithms for high-density microelectrode array electrophysiology},\n' ...
        '  journal = {Frontiers in Neuroscience},\n' ...
        '  volume = {10},\n' ...
        '  pages = {421},\n' ...
        '  year = {2016},\n' ...
        '  doi = {10.3389/fnins.2016.00421}\n' ...
        '}']);

    citations.siegle2021.inline = 'Siegle et al., 2021';
    citations.siegle2021.full = ['Siegle, J.H., Jia, X., Durand, S., et al. (2021). Survey of spiking ' ...
        'in the mouse visual system reveals functional hierarchy. Nature, 592, 86-92. ' ...
        'https://doi.org/10.1038/s41586-020-03171-x'];
    citations.siegle2021.bibtex = sprintf(['@article{siegle2021survey,\n' ...
        '  author = {Siegle, Joshua H. and Jia, Xiaoxuan and Durand, S\\''everine and others},\n' ...
        '  title = {Survey of spiking in the mouse visual system reveals functional hierarchy},\n' ...
        '  journal = {Nature},\n' ...
        '  volume = {592},\n' ...
        '  pages = {86--92},\n' ...
        '  year = {2021},\n' ...
        '  doi = {10.1038/s41586-020-03171-x}\n' ...
        '}']);

    % =====================================================================
    % Citation tracker state
    % =====================================================================
    usedKeys = {};

    % =====================================================================
    % Build sections
    % =====================================================================
    sections = {};
    sections{end+1} = introSection();
    sections{end+1} = peakTroughSection();
    sections{end+1} = noiseSection();
    sections{end+1} = muaSection();
    sections{end+1} = goodSection();
    sections{end+1} = nonsomaticSection();

    % Remove empty sections and join
    sections = sections(~cellfun(@isempty, sections));
    methodsText = strjoin(sections, sprintf('\n\n'));

    % Collect references and BibTeX
    references = cell(1, numel(usedKeys));
    bibtexEntries = cell(1, numel(usedKeys));
    for i = 1:numel(usedKeys)
        key = usedKeys{i};
        if strcmp(citationStyle, 'numbered')
            references{i} = sprintf('[%d] %s', i, citations.(key).full);
        else
            references{i} = citations.(key).full;
        end
        bibtexEntries{i} = citations.(key).bibtex;
    end

    % =====================================================================
    % Nested functions: citation helpers
    % =====================================================================
    function citeStr = cite(key)
        % Parenthetical citation: (Author et al., year) or [N]
        if ~ismember(key, usedKeys)
            usedKeys{end+1} = key;
        end
        if strcmp(citationStyle, 'inline')
            citeStr = sprintf('(%s)', citations.(key).inline);
        else
            idx = find(strcmp(usedKeys, key), 1);
            citeStr = sprintf('[%d]', idx);
        end
    end

    function citeStr = textcite(key)
        % Narrative citation: Author et al. (year) or Author et al. [N]
        if ~ismember(key, usedKeys)
            usedKeys{end+1} = key;
        end
        inlineStr = citations.(key).inline;
        parts = strsplit(inlineStr, ', ');
        authorPart = strjoin(parts(1:end-1), ', ');
        yearPart = parts{end};
        if strcmp(citationStyle, 'inline')
            citeStr = sprintf('%s (%s)', authorPart, yearPart);
        else
            idx = find(strcmp(usedKeys, key), 1);
            citeStr = sprintf('%s [%d]', authorPart, idx);
        end
    end

    % =====================================================================
    % Nested functions: section generators
    % =====================================================================
    function text = introSection()
        if isfield(param, 'splitGoodAndMua_NonSomatic') && param.splitGoodAndMua_NonSomatic
            categories = ['noise, non-somatic good, non-somatic multi-unit activity ' ...
                '(MUA), multi-unit activity (MUA), and good single units'];
        else
            categories = ['noise, non-somatic, multi-unit activity (MUA), and good ' ...
                'single units'];
        end
        text = sprintf(['Automated spike sorting quality control was performed using ' ...
            'BombCell %s. Each unit was sequentially evaluated and classified as %s, ' ...
            'based on the quality metrics described below.'], cite('bombcell'), categories);
    end

    function text = peakTroughSection()
        thresh = getParam('minThreshDetectPeaksTroughs', 0.2);
        text = sprintf(['Noise and non-somatic quality metrics were derived from each ' ...
            'unit''s template waveform on its peak channel (the channel with the largest ' ...
            'amplitude). Peaks and troughs were identified using a prominence-based ' ...
            'peak-finding algorithm, with the minimum prominence threshold set to %g ' ...
            'times the waveform''s absolute maximum. The main trough was defined as the ' ...
            'most prominent trough, and peaks were detected separately on either side ' ...
            'of it.'], thresh);
    end

    function text = noiseSection()
        maxNPeaks = getParam('maxNPeaks', 2);
        maxNTroughs = getParam('maxNTroughs', 1);
        minWvDur = getParam('minWvDuration', 100);
        maxWvDur = getParam('maxWvDuration', 1150);
        maxBaseline = getParam('maxWvBaselineFraction', 0.3);
        maxScndPeak = getParam('maxScndPeakToTroughRatio_noise', 0.8);

        if maxNTroughs == 1
            troughWord = 'trough';
        else
            troughWord = 'troughs';
        end

        text = sprintf(['Units were first evaluated for noise-like waveform features. ' ...
            'A unit was classified as noise if any of the following criteria were met: ' ...
            'the waveform contained more than %d peaks or more than %d %s; ' ...
            'the peak-to-trough duration (time between the largest peak and largest ' ...
            'trough) fell outside %d%s%d %ss; the waveform baseline fraction (maximum ' ...
            'absolute value in the baseline period preceding the spike, relative to the ' ...
            'waveform''s absolute peak) exceeded %g; or the ratio of the second peak ' ...
            'amplitude to the main trough amplitude exceeded %g.'], ...
            maxNPeaks, maxNTroughs, troughWord, ...
            minWvDur, char(8211), maxWvDur, char(181), ...
            maxBaseline, maxScndPeak);

        if getParam('computeSpatialDecay', 1)
            if getParam('spDecayLinFit', 0)
                fitType = 'linear';
                nChan = 6;
            else
                fitType = 'exponential';
                nChan = 10;
            end
            if strcmp(fitType, 'exponential')
                article = 'an';
            else
                article = 'a';
            end

            spText = sprintf([' Additionally, the spatial decay of the waveform was ' ...
                'quantified by measuring the absolute peak amplitude on the %d nearest ' ...
                'channels within 33 %sm in the x-dimension, as a function of Euclidean ' ...
                'distance (in %sm) from the peak channel'], nChan, char(181), char(181));

            if getParam('normalizeSpDecay', 1)
                spText = [spText '. Amplitudes were normalized to the peak channel'];
            end
            spText = [spText sprintf(', and %s %s fit was applied to obtain the decay slope', article, fitType)];

            if getParam('spDecayLinFit', 0)
                minSlope = getParam('minSpatialDecaySlope', -0.008);
                spText = [spText sprintf(['. Units with a decay slope exceeding %g were ' ...
                    'classified as noise.'], minSlope)];
            else
                minSlope = getParam('minSpatialDecaySlopeExp', 0.01);
                maxSlope = getParam('maxSpatialDecaySlopeExp', 0.1);
                spText = [spText sprintf(['. Units with a decay slope outside the range ' ...
                    '%g%s%g were classified as noise.'], minSlope, char(8211), maxSlope)];
            end
            text = [text spText];
        end
    end

    function text = muaSection()
        tauRMin = getParam('tauR_valuesMin', 0.002) * 1000;
        tauRMax = getParam('tauR_valuesMax', 0.002) * 1000;
        tauC = getParam('tauC', 0.0001) * 1000;
        maxRPV = getParam('maxRPVviolations', 0.1);
        maxPct = getParam('maxPercSpikesMissing', 20);
        minSpikes = getParam('minNumSpikes', 300);
        minPR = getParam('minPresenceRatio', 0.7);
        binSize = getParam('presenceRatioBinSize', 60);

        text = ['Non-noise units were then assessed for single-unit isolation quality, ' ...
            'and classified as multi-unit activity (MUA) if any of the following criteria ' ...
            'were met.'];

        % RPV
        if getParam('hillOrLlobetMethod', 1)
            rpvMethod = textcite('hill2011');
        else
            rpvMethod = textcite('llobet2022');
        end
        rpvText = sprintf([' The fraction of refractory period violations was estimated ' ...
            'using the method of %s'], rpvMethod);
        if tauRMin == tauRMax
            rpvText = [rpvText sprintf(', with a refractory period of %.1f ms and a censored period of %.1f ms', ...
                tauRMin, tauC)];
        else
            tauRStep = getParam('tauR_valuesStep', 0.0005) * 1000;
            rpvText = [rpvText sprintf([', testing refractory period values from %.1f to ' ...
                '%.1f ms in steps of %.1f ms (with a censored period of %.1f ms), and ' ...
                'selecting the optimal value per unit'], tauRMin, tauRMax, tauRStep, tauC)];
        end
        rpvText = [rpvText sprintf('; units exceeding %.0f%% violations were classified as MUA.', maxRPV * 100)];
        text = [text rpvText];

        % Spikes missing
        text = [text sprintf([' The percentage of missing spikes was estimated by fitting a ' ...
            'Gaussian to each unit''s amplitude distribution and computing the fraction ' ...
            'falling below the detection threshold; units with more than %d%% missing ' ...
            'spikes were classified as MUA.'], maxPct)];

        if getParam('computeTimeChunks', 0)
            delta = getParam('deltaTimeChunk', 360);
            text = [text sprintf([' Both the refractory period violation rate and the ' ...
                'percentage of missing spikes were additionally computed in %d-second ' ...
                'time chunks to identify the most stable recording period for each unit.'], delta)];
        end

        % Spike count
        text = [text sprintf(' Units with fewer than %d total spikes were classified as MUA.', minSpikes)];

        % Presence ratio
        text = [text sprintf([' The presence ratio was computed by dividing the recording into ' ...
            '%d-second bins and calculating the fraction of bins in which the unit''s spike ' ...
            'count exceeded 5%% of the 90th-percentile bin count (inspired by %s); units ' ...
            'with a presence ratio below %g were classified as MUA.'], ...
            binSize, textcite('siegle2021'), minPR)];

        % Amplitude and SNR
        if getParam('extractRaw', 1)
            minAmp = getParam('minAmplitude', 40);
            minSNR = getParam('minSNR', 5);
            nSpikes = getParam('nRawSpikesToExtract', 100);
            rawText = sprintf([' The raw waveform amplitude (peak-to-trough voltage in ' ...
                '%sV) was computed from the mean of %d randomly sampled raw waveforms'], ...
                char(181), nSpikes);
            if getParam('detrendWaveform', 1)
                rawText = [rawText ' after linear de-trending'];
            end
            rawText = [rawText sprintf(['; units with an amplitude below %d %sV were ' ...
                'classified as MUA. The signal-to-noise ratio (SNR) was defined as the ' ...
                'waveform amplitude divided by the standard deviation of the baseline ' ...
                'noise; units with an SNR below %d were classified as MUA.'], ...
                minAmp, char(181), minSNR)];
            text = [text rawText];
        end

        % Drift
        if getParam('computeDrift', 0)
            maxDrift = getParam('maxDrift', 100);
            driftBin = getParam('driftBinSize', 60);
            text = [text sprintf([' Electrode drift was estimated by computing, in %d-second ' ...
                'bins, the spike depth as a weighted center of mass of the first principal ' ...
                'component features across channels. Maximum drift was defined as the ' ...
                'difference between the maximum and minimum median spike depth across bins; ' ...
                'units with drift exceeding %d %sm were classified as MUA.'], ...
                driftBin, maxDrift, char(181))];
        end

        % Distance metrics
        if getParam('computeDistanceMetrics', 0)
            nChan = getParam('nChannelsIsoDist', 4);
            isoMin = getParam('isoDmin', 20);
            lratioMax = getParam('lratioMax', 0.3);
            text = [text sprintf([' Cluster isolation was assessed using principal component ' ...
                'features from the %d nearest channels. Isolation distance (the Mahalanobis ' ...
                'distance to the nearest non-member spike) was required to be at least %d. ' ...
                'The L-ratio (a contamination estimate based on the chi-squared distribution ' ...
                'of Mahalanobis distances) was required to be below %g. Units failing either ' ...
                'criterion were classified as MUA.'], nChan, isoMin, lratioMax)];
        end
    end

    function text = goodSection()
        text = ['All remaining units that passed the above criteria were classified as ' ...
            'good single units.'];

        % Optional unit count summary
        if ~isempty(qualityMetrics)
            try
                if istable(qualityMetrics)
                    if ismember('bc_unitType', qualityMetrics.Properties.VariableNames)
                        unitTypes = qualityMetrics.bc_unitType;
                    else
                        return;
                    end
                elseif isstruct(qualityMetrics)
                    if isfield(qualityMetrics, 'bc_unitType')
                        unitTypes = qualityMetrics.bc_unitType;
                    else
                        return;
                    end
                else
                    return;
                end

                nTotal = numel(unitTypes);
                labels = {'GOOD', 'MUA', 'NOISE', 'NON-SOMA', 'NON-SOMA GOOD', 'NON-SOMA MUA'};
                parts = {};
                for i = 1:numel(labels)
                    if iscell(unitTypes)
                        c = sum(strcmp(unitTypes, labels{i}));
                    else
                        c = sum(unitTypes == string(labels{i}));
                    end
                    if c > 0
                        parts{end+1} = sprintf('%d %s', c, lower(labels{i})); %#ok<AGROW>
                    end
                end
                if ~isempty(parts)
                    text = [text sprintf(' Of %d total units, %s.', nTotal, strjoin(parts, ', '))];
                end
            catch
                % Silently skip if unit type extraction fails
            end
        end
    end

    function text = nonsomaticSection()
        maxMainPTR = getParam('maxMainPeakToTroughRatio_nonSomatic', 0.8);
        maxP1P2 = getParam('maxPeak1ToPeak2Ratio_nonSomatic', 3);
        minT2P2 = getParam('minTroughToPeak2Ratio_nonSomatic', 5);
        minWPeak = getParam('minWidthFirstPeak_nonSomatic', 4);
        minWTrough = getParam('minWidthMainTrough_nonSomatic', 5);

        if getParam('splitGoodAndMua_NonSomatic', 0)
            overrideText = ['those previously classified as good were reclassified as ' ...
                'non-somatic good, and those previously classified as MUA were reclassified ' ...
                'as non-somatic MUA'];
        else
            overrideText = 'their classification was updated to non-somatic';
        end

        text = sprintf(['Finally, all non-noise units (both good and MUA) were evaluated for ' ...
            'non-somatic (e.g., axonal) waveform characteristics %s. A unit was identified ' ...
            'as non-somatic if either of two conditions was met. First, if all of the ' ...
            'following were true simultaneously: the ratio of the main trough amplitude to ' ...
            'the pre-trough peak amplitude was less than %d, the half-prominence width of ' ...
            'the pre-trough peak was less than %d samples, the half-prominence width of the ' ...
            'main trough was less than %d samples, and the ratio of the pre-trough peak ' ...
            'amplitude to the post-trough peak amplitude exceeded %d. Second, independently, ' ...
            'if the ratio of the largest peak amplitude (before or after the trough) to the ' ...
            'main trough amplitude exceeded %g. For units identified as non-somatic, %s.'], ...
            cite('deligkaris2016'), minT2P2, minWPeak, minWTrough, maxP1P2, maxMainPTR, ...
            overrideText);
    end

    % =====================================================================
    % Helper: get parameter with default
    % =====================================================================
    function val = getParam(name, default)
        if isfield(param, name)
            val = param.(name);
        else
            val = default;
        end
    end

end
