
%% clean commented quality metrics [WIP]
% JF 2020-04-01
% dependancies: readNPY, keep, http://www.fieldtriptoolbox.org/ (for
% ACG/CCG), APscripts

%% QQ to do:
% -add remove duplicate spikes,
% -CAR raw data - subtract other detected spike in rw? or only spikes when no overlap.
% -better way to define chunks: estim. drift + estim. bimodality. (look at
%Marius)
% -why is this so slow? improved by using only str templates, no raw data, no redundancy, faster CCG/ACG function.
% mahal function in ditance metric is what takes longest 12s/cluster vs
% 3s/cluster with and without.. (~18 min total vs 4 min).
% ->add memory pre-allocation, vectorize instead of for-loop, params in
% .csv file. Or just run once, save in local folder and then only load...

%% example dataset - replace with your dataset and its path :)

%% load data
function [qMetric, ephysParams, ephysData, behavData, goodUnits, msn, fsi, tan] = classifyUnitQualityCellTypeJF(save_path, ephys_path, thisAnimal, thisDate, thisExperiment, corona, param, raw, trained )
if param.dontdo && exist([save_path, 'qMetric.mat'], 'file')
    load([save_path, 'qMetric.mat'])
    load([save_path, 'ephysParams.mat'])
    load([save_path, 'param.mat'])
    load([save_path, 'ephysData.mat'])
else
    [ephysData, behavData] = loadEphysDataJF(ephys_path, thisAnimal, thisDate, thisExperiment, corona, trained ); %load data, and keep only 'good' and 'mua' units.
    ephysData.timeChunks = min(ephysData.spike_times_timeline):300:max(ephysData.spike_times_timeline); %5 minute chunks
    ephysData.waveform_t = 1e3 * ((0:size(ephysData.templates, 2) - 1) / ephysData.ephys_sample_rate);
    ephysData.allT = unique(ephysData.spike_templates);
    raw.used_channels_idx = ephysData.channel_map + 1;

    qMetric = struct;
    ephysParams = struct;
    qMetric.drift = struct;
    ephysParams.drift = struct;

    %preallocate memory
    if param.strOnly
        allT = ephysData.allT(ephysData.str_templates);
    else
        allT = ephysData.allT;
    end
    tic

    for iUnit = 1:size(allT, 1)

        %% for all timepoints
        param.drift = 0;
        [qMetric, ephysParams] = qualityMetEphysParamJF(qMetric, ephysParams, ephysData, raw, param, allT, iUnit, ephysData.str_templates, []);

        %% chunk by chunk (optional)
        if param.driftdo

            %% get drift chunk
            timeChunks = ephysData.timeChunks;
            thisRawUnit = allT(iUnit);
            if param.strOnly
                strUnits = find(ephysData.str_templates);
                thisUnit = strUnits(iUnit);
            else
                thisUnit = iUnit;
            end
            for iChunk = 2:numel(timeChunks)
                theseSpikesIdx = ephysData.spike_templates == thisUnit & ephysData.spike_times_timeline >= timeChunks(iChunk-1) ...
                    & ephysData.spike_times_timeline < timeChunks(iChunk);
                theseSpikes = ephysData.spike_times_timeline(theseSpikesIdx);
                theseAmplis = ephysData.template_amplitudes(theseSpikesIdx);
                pc_feature_ind = ephysData.pc_features_ind_full;

                allSpikesIdx = ephysData.spike_templates(ephysData.spike_times_timeline >= timeChunks(iChunk-1) ...
                    & ephysData.spike_times_timeline < timeChunks(iChunk));
                pc_features = ephysData.pc_features(allSpikesIdx, :, :);

                %% percentage spikes missing
                % using gaussian %%QQ add test for this

                try
                    [qMetric.percent_missing_ndtr(iChunk, iUnit), fitOutput] = ampli_fit_prc_missJF(theseAmplis, 0);
                catch
                    qMetric.percent_missing_ndtr(iChunk, iUnit) = NaN;
                end


                % using symetry %%QQ can i really do this?
                qMetric.symSpikesMissing(iChunk, iUnit) = prctgMissingSymetry(theseAmplis);

               
            end

            percent_missing_ndtr = squeeze(qMetric.percent_missing_ndtr(:, iUnit));
            if numel(find(percent_missing_ndtr <= 30)) > 1

                q = find(percent_missing_ndtr <= 30);
                a = diff(q);
                b = find([a; inf] > 1);
                c = diff([0; b]);
                maxC = find(c == max(c));
                if numel(maxC) > 1
                    maxC = maxC(1);
                end
                stopT = b(maxC);
                startT = stopT - c(maxC) + 1;
                useTimeChunk = [timeChunks(startT), timeChunks(stopT)];
                theseAmplisChunk = theseAmplis(theseSpikes >= useTimeChunk(1) & theseSpikes < useTimeChunk(2));
                theseSpikesChunk = theseSpikes(theseSpikes >= useTimeChunk(1) & theseSpikes < useTimeChunk(2));

            elseif numel(find(percent_missing_ndtr <= 30)) == 1

                useTimeChunk = [timeChunks(percent_missing_ndtr <= 30), timeChunks(find(percent_missing_ndtr <= 30)+1)];
                theseAmplisChunk = theseAmplis(theseSpikes >= useTimeChunk(1) & theseSpikes < useTimeChunk(2));
                theseSpikesChunk = theseSpikes(theseSpikes >= useTimeChunk(1) & theseSpikes < useTimeChunk(2));

            else

                useTimeChunk = [0, 0];
                theseAmplisChunk = NaN;
                theseSpikesChunk = NaN;

            end

            qMetric.useTimeChunk(iUnit, :) = useTimeChunk;
            param.drift = 1;
        end

    end
    toc

    % save qMetrics and ephysParams
    save([save_path, 'qMetric.mat'], 'qMetric', '-v7.3');
    save([save_path, 'ephysData.mat'], 'ephysData', '-v7.3');
    save([save_path, 'ephysParams.mat'], 'ephysParams', '-v7.3');
    save([save_path, 'param.mat'], 'param', '-v7.3');
end

%% post

%% label 'good' units : vector the size of unique(ephysData.spike_templates)


goodUnits = qMetric.numSpikes >= param.minNumSpikes & qMetric.waveformRawAmpli >= param.minAmpli & ...
    qMetric.spatialDecayTemp1 >= param.minSpatDeKlowbound & qMetric.fractionRPVchunk <= param.maxRPV & ...
    qMetric.numPeaksTroughsTemp <= param.maxNumPeak & ...
    ephysParams.somatic == param.somaCluster & ephysParams.templateDuration < 800;

%% label celltypes - previous mehod. UINs now added with prop_isi. 
msn = goodUnits & ephysParams.postSpikeSuppression < 40 & ephysParams.templateDuration > param.cellTypeDuration;
fsi = goodUnits & ephysParams.postSpikeSuppression < 40 & ephysParams.templateDuration < param.cellTypeDuration;
tan = goodUnits & ephysParams.postSpikeSuppression >= 40;

%% plot plot plot
if param.plotMetricsCtypes
    %histogram metrics
    figure();
    subplot(4, 4, 1)
    hist(qMetric.numSpikes, 10000);
    xlim([0, max(qMetric.numSpikes)])
    ylabel('# units')
    xlabel('number of spikes')
    makepretty;
    breakxaxis([1000, 0.999 * max(qMetric.numSpikes)])

    subplot(4, 4, 2)
    hist(qMetric.waveformRawAmpli, 70); %use raw otherwise nothing makes sense
    ylabel('# units')
    xlabel('amplitude')
    makepretty;

    subplot(4, 4, 3)
    hist(qMetric.numPeaksTroughsTemp, 20);
    ylabel('# units')
    xlabel('number of troughs/peaks')
    makepretty;

    subplot(4, 4, 4)
    hist(qMetric.spatialDecayTemp1, 100);
    ylabel('# units')
    xlabel('sp. decay')
    makepretty;

    subplot(4, 4, 5)
    hist(qMetric.drift.fractionRPVchunk, 100);
    ylabel('# units')
    xlabel('frac RPV')
    makepretty;

    subplot(4, 4, 6)
    histogram(qMetric.isoD, 20);
    ylabel('# units')
    xlabel('isolation distance')
    makepretty;

    subplot(4, 4, 7)
    histogram(qMetric.Lratio, 2000);
    ylabel('# units')
    xlabel('l-ratio')
    makepretty;

    subplot(4, 4, 8)
    histogram(qMetric.silhouetteScore, 2000);
    ylabel('# units')
    xlabel('silhouette-score')
    makepretty;

    %cell type histogram
    figure();
    scatter(ephysParams.templateDuration(fsi), ephysParams.postSpikeSuppression(fsi), 'b')
    hold on;
    scatter(ephysParams.templateDuration(msn), ephysParams.postSpikeSuppression(msn), 'r')
    hold on;
    scatter(ephysParams.templateDuration(tan), ephysParams.postSpikeSuppression(tan), 'g')

    ylabel('post spike suppression (ms)')
    xlabel('waveform peak-trough (ms)')
    xlim([0, 800])
    makepretty;
    %cell type properites: mean ACG, waveform, cv2, burst, firing rate, max
    %firing rate, propISI>2s

    %waveforms
    figure();

    title('Mean waveforms');
    hold on;
    P(1) = plot(waveform_t, mean(qMetric.waveformUnit(fsi, :)));
    plotshaded(waveform_t, [-std(qMetric.waveformUnit(fsi, :)) + mean(qMetric.waveformUnit(fsi, :)); std(qMetric.waveformUnit(fsi, :)) + mean(qMetric.waveformUnit(fsi, :))], 'b');
    hold on;
    xlim([0, waveform_t(end)])
    makepretty;

    P(2) = plot(waveform_t, mean(qMetric.waveformUnit(msn, :)));
    plotshaded(waveform_t, [-std(qMetric.waveformUnit(msn, :)) + mean(qMetric.waveformUnit(msn, :)); std(qMetric.waveformUnit(msn, :)) + mean(qMetric.waveformUnit(msn, :))], 'r');
    xlim([0, waveform_t(end)])
    makepretty;
    hold on;

    if sum(tan) > 1
        P(3) = plot(waveform_t, mean(qMetric.waveformUnit(tan, :)));
        plotshaded(waveform_t, [-std(qMetric.waveformUnit(tan, :)) + mean(qMetric.waveformUnit(tan, :)); std(qMetric.waveformUnit(tan, :)) + mean(qMetric.waveformUnit(tan, :))], 'g');
        xlim([0, waveform_t(end)])
        makepretty;
    else
        P(3) = plot(waveform_t, qMetric.waveformUnit(tan, :));
        xlim([0, waveform_t(end)])
        makepretty;
    end

    ylabel('Amplitude (\muV)')
    xlabel('Time (ms)')
    legend(P, {'FSI mean +/- std', 'MSN mean +/- std', 'TAN mean +/- std'});
    makepretty;

    %acg
    acg = ephysParams.ACG;
    acgmean = smoothdata(mean(acg(fsi, 501:end)), 'gaussian', [0, 5]);
    acgstd = smoothdata(std(acg(fsi, 501:end)), 'gaussian', [0, 5]);
    figure();
    subplot(131)
    cla();
    title('FSIs');
    hold on;
    plot(0:0.001:0.5, acgmean, 'b')
    hold on;
    plotshaded(0:0.001:0.5, [-acgstd + acgmean; acgstd + acgmean], 'b');
    area(0:0.001:0.5, acgmean, 'FaceColor', 'b');
    xlim([0, 0.5])
    ylim([0, 80])
    ylabel('sp/s')
    xlabel('Time (s)')
    makepretty;

    acgmean = smoothdata(mean(acg(msn, 501:end)), 'gaussian', [0, 5]);
    acgstd = smoothdata(std(acg(msn, 501:end)), 'gaussian', [0, 5]);
    subplot(132)
    cla();
    title('MSNs');
    hold on;
    plot(0:0.001:0.5, acgmean, 'r'); %,'FaceColor','b');
    hold on;
    plotshaded(0:0.001:0.5, [(-acgstd + acgmean); (acgstd + acgmean)], 'r');
    area(0:0.001:0.5, acgmean, 'FaceColor', 'r');
    ylabel('sp/s')
    xlabel('Time (s)')
    xlim([0, 0.5])
    ylim([0, 80])
    makepretty;

    if sum(tan) > 1
        acgmean = smoothdata(mean(acg(tan, 501:end)), 'gaussian', [0, 5]);
        acgstd = smoothdata(std(acg(tan, 501:end)), 'gaussian', [0, 5]);
        subplot(133)
        cla();
        title('TANs');
        hold on;
        plot(0:0.001:0.5, acgmean, 'g'); %,'FaceColor','b');
        hold on;
        plotshaded(0:0.001:0.5, [-acgstd + acgmean; acgstd + acgmean], 'g');
        area(0:0.001:0.5, acgmean, 'FaceColor', 'g');
    else
        acgmean = smoothdata(acg(tan, 501:end), 'gaussian', [0, 5]);
        subplot(133)
        cla();
        title('TANs');
        hold on;
        plot(0:0.001:0.5, acgmean, 'g'); %,'FaceColor','b');
        area(0:0.001:0.5, acgmean, 'FaceColor', 'g');
    end
    ylabel('sp/s')
    xlabel('Time (s)')
    xlim([0, 0.5])
    ylim([0, 80])
    makepretty;
    suptitle('ACG')
    %mean acg

end


% %% figure
% allFSI=find(fsi);
% for iFSI = 1:sum(fsi)
%     figure();
%     thisFSI=allFSI(iFSI);
% P(1) = plot(waveform_t, qMetric.waveformUnit(thisFSI, :));
% end
%
%  allTAN=find(tan);
%  for iTAN = 1:sum(tan)
%      figure();
%      thisTAN=allTAN(iTAN);
%         subplot(1,2,1)
%        P(1) = plot(waveform_t, qMetric.waveformUnit(thisTAN, :));
%        subplot(1,2,2)
%                acgmean = smoothdata(acg(thisTAN, 501:end), 'gaussian', [0, 5]);
%
%         hold on;
%         plot(0:0.001:0.5, acgmean, 'g'); %,'FaceColor','b');
%         area(0:0.001:0.5, acgmean, 'FaceColor', 'g');
%  end
%
% allMSN=find(msn);
% for iMSN = 1:sum(msn)
%     figure();
%     thisMSN=allMSN(iMSN);
% P(1) = plot(waveform_t, qMetric.waveformUnit(thisMSN, :));
% end
end

%% additional: are > 3peak/troughs another cell type, LTS? + can we seperate "FSIs" into FSIs + TH+ neurons ?:hmm:
