function [postSpikeSup_ms, tauRise_ms, tauDecay_ms, refractoryPeriod_ms] = computeACGprop(thisACG, ACGbinSize, ACGduration)

%% post spike suppression
postSpikeSup = find(thisACG(500:1000) >= nanmean(thisACG(600:900))); % nanmean(ephysParams.ACG(iUnit, 900:1000)) also works.
if ~isempty(postSpikeSup)
    postSpikeSup = postSpikeSup(1);
else
    postSpikeSup = NaN;
end
postSpikeSup_ms = postSpikeSup  / ACGbinSize / 1000;

%% tau rise and decay
% Assuming an exponential rise, we fit the ACG to an exponential function
% and estimate tau from the fit parameters
ACGlags = -ACGduration/2:ACGbinSize:ACGduration/2;
[tauRise, ~] = estimateTau(ACGlags, thisACG, true); % True for rising phase
[tauDecay, ~] = estimateTau(ACGlags, thisACG, false); % False for decaying phase
tauRise_ms  = tauRise   * 1000;
tauDecay_ms = tauDecay  * 1000;

%% refractory period
% Assuming the refractory period is the time lag at which the ACG starts to rise
ACGlags_from0 = ACGlags(ACGlags>0);
refractoryPeriod = ACGlags_from0(find(thisACG(ACGlags>0) > (min(thisACG) + std(thisACG)), 1));
if isempty(refractoryPeriod)
    refractoryPeriod_ms = NaN;
else
    refractoryPeriod_ms = refractoryPeriod  *1000;
end


% figure();
% plot(ACGlags, thisACG)

%% internal functions 
% Function to estimate tau rise/decay from ACG
function [tau, fitResult] = estimateTau(lags, autoCorr, isRise)
    % Fit the rising or decaying part of the ACG
    if isRise
        relevantPart = autoCorr(lags >= 0);
        relevantLags = lags(lags >= 0);
    else
        relevantPart = autoCorr(lags <= 0);
        relevantLags = lags(lags <= 0);
    end

    % Exponential fit
    fitFunc = fittype('a*exp(-b*x)', 'independent', 'x');
    [fitResult, ~] = fit(relevantLags', relevantPart', fitFunc, 'StartPoint', [max(relevantPart), 1]);

    % Tau is the inverse of the b parameter in the exponential
    tau = 1 / fitResult.b ;
end

end
