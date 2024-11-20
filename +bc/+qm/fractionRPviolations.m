function [RPVrate, RPVfraction, overestimateBool] = fractionRPviolations(theseSpikeTimes, theseAmplitudes, ...
    tauR, param, timeChunks, RPV_tauR_estimate)
% JF, get the estimated fraction of refractory period violation for a unit
% for each timeChunk
% ------
% Inputs
% ------
% theseSpikeTimes: nSpikesforThisUnit × 1 double vector of time in seconds
%   of each of the unit's spikes.
% theseAmplitudes: nSpikesforThisUnit × 1 double vector of the amplitude scaling factor
%   that was applied to the template when extracting that spike
%   , only needed if plotThis is set to true
% tauR: refractory period
% timeChunk: experiment duration time chunks
% param: structure with field s
% - plotThis: boolean, whether to plot ISIs or not
% - tauC: censored period
% - hillOrLlobetMethod
% ------
% Outputs
% ------
% RPVrate estimated false positive rate of the spikes in the given
%   spike train, using the Hill et al. equations or
% Llobet et al. equations
% RPVfraction: fraction of refractory period violations over the total
%   number of spikes.
% overestimateBool: boolean, true if the number of refractory period violations
%    is too high. we then overestimate the fraction of
%    contamination.
% ------
% Reference
% ------
% - Llobet V et al., biorXiv 2023. (see equation 3, page 11)
% - Hill, Daniel N., Samar B. Mehta, and David Kleinfeld.
% "Quality metrics to accompany spike sorting of extracellular signals."
% Journal of Neuroscience 31.24 (2011): 8699-8705:
% r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T , solve for Fp , fraction
% refractory period violatons. 2 factor because rogue spikes can occur before or
% after true spike.
%
% Note: there is a difference between the Hill et al.  paper (which I based
% this function on) and their implementation in UltraMegaSort. They use an
% approximation in UltraMegaSort, which I think was intended to avoid imaginary
% numbers but is not accurate to the solution they provide in the paper.
% Another point: Hill et al's solution is partially incorrect because they used
% an expression from Meunier et al (2003) that assumed contaminating spikes
% came from a single neuron with a refractory period. The correct expression,
% derived in Llobet et al (bioRxiv 2022), accounts for contamination from true
% Poisson processes like electrical noise or multiple nearby neurons. IN
% practice, below a rate of 30, both methods are highly correlated and
% agree.
% Final point: this function assumes a set tauR, but this is likely
% different for different brain regions - IBL/Steinmetz lab nicely take this
% into account here: https://github.com/SteinmetzLab/slidingRefractory

% initialize variables
RPVrate_Hill = ones(length(timeChunks)-1, length(tauR));
RPVrate_Llobet = ones(length(timeChunks)-1, length(tauR));
overestimateBool = nan(length(timeChunks)-1, length(tauR));
RPVfraction = nan(length(timeChunks)-1, length(tauR));

if param.plotDetails
    figure('Color', 'none');
end

for iTimeChunk = 1:length(timeChunks) - 1 %loop through each time chunk
    % spikes in this chunk
    spikeChunk = theseSpikeTimes(theseSpikeTimes >= timeChunks(iTimeChunk) & theseSpikeTimes < timeChunks(iTimeChunk+1));

    % number of spikes in chunk
    N_chunk = length(spikeChunk);

    % chunk duration
    durationChunk = timeChunks(iTimeChunk+1) - timeChunks(iTimeChunk);

    % chunk ISIs
    isisChunk = diff(spikeChunk);

    % total times at which refractory period violations can occur
    for iTauR_value = 1:length(tauR)
        thisSpikeTrain = theseSpikeTimes(theseSpikeTimes >= timeChunks(iTimeChunk) & theseSpikeTimes < timeChunks(iTimeChunk+1));
        nRPVs = sum(diff(thisSpikeTrain) <= tauR(iTauR_value));

        RPVfraction(iTimeChunk, iTauR_value) = nRPVs / N_chunk;

        overestimateBool(iTimeChunk, iTauR_value) = 0;
if param.hillOrLlobetMethod
        k = 2 * (tauR(iTauR_value) - param.tauC) * N_chunk^2;
        T = abs(diff(timeChunks(iTimeChunk:iTimeChunk+1)));
        % a = 2 * (tauR(iTauR_value) - param.tauC) * N_chunk^2 / abs(diff(timeChunks(iTimeChunk:iTimeChunk+1)));
        % observed number of refractory period violations

        if nRPVs == 0 % no observed refractory period violations - this can
            % also be because there are no spikes in this interval - use presence ratio to weed this out
            RPVrate_Hill(iTimeChunk, iTauR_value) = 0;
        else % otherwise solve the equation above
            rts = roots([k, -k, nRPVs * T]);
            RPVrate_Hill(iTimeChunk, iTauR_value) = min(rts);
            if ~isreal(RPVrate_Hill(iTimeChunk, iTauR_value)) % function returns imaginary number if r is too high
                RPVrate_Hill(iTimeChunk, iTauR_value) = nRPVs / (2 * (tauR(iTauR_value) - param.tauC) * (N_chunk - nRPVs));
            end
            if RPVrate_Hill(iTimeChunk, iTauR_value) > 1 % it is nonsense to have a rate >1, the assumptions are failing here
                RPVrate_Hill(iTimeChunk, iTauR_value) = 1;
                overestimateBool(iTimeChunk, iTauR_value) = 1;
            end
        end
else % this method is slower
    N = length(thisSpikeTrain);
    isi_violations_sum = 0;
    
    for i = 1:N
        for j = i+1:N
            isi = thisSpikeTrain(j) - thisSpikeTrain(i);
            if isi <= tauR(iTauR_value) && isi >= param.tauC
                isi_violations_sum = isi_violations_sum + 1;
            end
        end
    end
    underRoot = 1 - (isi_violations_sum * (durationChunk - 2 * N_chunk * param.tauC)) / (N_chunk^2 * (tauR(iTauR_value) - param.tauC));
    if underRoot >= 0
        RPVrate_Llobet(iTimeChunk, iTauR_value) = 1 - sqrt(underRoot);
    else
        RPVrate_Llobet(iTimeChunk, iTauR_value) = 1;
    end
end
    end


    if param.hillOrLlobetMethod
        RPVrate = RPVrate_Hill;
    else
        RPVrate = RPVrate_Llobet;
    end

    if param.plotDetails
        theseISI = diff(theseSpikeTimes);
        subplot(2, length(timeChunks)-1, (length(timeChunks) - 1)+iTimeChunk)
        theseisiclean = isisChunk(theseISI >= param.tauC); % removed duplicate spikes
        [isiProba, edgesISI] = histcounts(theseisiclean*1000, [0:0.5:100]);
        bar(edgesISI(1:end-1)+mean(diff(edgesISI)), isiProba, 'FaceColor', [0, 0.35, 0.71], ...
            'EdgeColor', [0, 0.35, 0.71]); %Check FR
        if iTimeChunk == 1
            xlabel('Interspike interval (ms)')
            ylabel('# of spikes')
        else
            xticks([])
            yticks([])
        end
        ylims = ylim;
        [fr, ~] = histcounts(theseisiclean*1000, [0:0.5:5000]);
        line([0, 10], [nanmean(fr(800:1000)), nanmean(fr(800:1000))], 'Color', [0.86, 0.2, 0.13], 'LineStyle', '--');
        %dummyh = line(nan, nan, 'Linestyle', 'none', 'Marker', 'none', 'Color', 'none');
        if isnan(RPV_tauR_estimate)
            for iTauR_value = [1, length(tauR)]
                line([tauR(iTauR_value) * 1000, tauR(iTauR_value) * 1000], [ylims(1), ylims(2)], 'Color', [0.86, 0.2, 0.13]);
            end
            if length(tauR) == 1
                title({[num2str(RPVrate(iTimeChunk, 1)*100, '%.0f'), '% rpv', newline, ...
                    'frac. rpv=', num2str(RPVfraction(iTimeChunk, 1), '%.3f')]});
            else
                title([num2str(round(RPVrate(iTimeChunk, 1)*100, '%.0f')), '% rpv', newline, ...
                    num2str(round(RPVrate(iTimeChunk, length(tauR))*100, '%.0f')), '% rpv']);
            end

        else
            line([tauR(RPV_tauR_estimate) * 1000, tauR(RPV_tauR_estimate) * 1000], [ylims(1), ylims(2)], 'Color', [0.86, 0.2, 0.13]);
            title({[num2str(RPVrate(iTimeChunk, RPV_tauR_estimate)*100, '%.0f'), '% rpv', newline, ...
                'frac. rpv=', num2str(RPVfraction(iTimeChunk, RPV_tauR_estimate), '%.3f')]});

        end

        %set(gca, 'XScale', 'log')
    end

end

if param.plotDetails
    subplot(2, numel(timeChunks)-1, 1:numel(timeChunks)-1)
    scatter(theseSpikeTimes, theseAmplitudes, 4, [0, 0.35, 0.71], 'filled');
    hold on;
    % chunk lines
    ylims = ylim;
    for iTimeChunk = 1:length(timeChunks)
        line([timeChunks(iTimeChunk), timeChunks(iTimeChunk)], ...
            [ylims(1), ylims(2)], 'Color', [0.7, 0.7, 0.7])
    end
    xlabel('time (s)')
    ylabel(['amplitude scaling', newline, 'factor'])
    if exist('prettify_plot', 'file')
        prettify_plot('FigureColor', 'w')
    else
        warning('https://github.com/Julie-Fabre/prettify-matlab repo missing - download it and add it to your matlab path to make plots pretty')
    end
end


end