function unitClassif = bc_classifyCells(ephysProperties, paramEP, region)
% classify striatal, GPe and cortical cells
unitClassif = cell(size(ephysProperties,1),1);


if ismember(region, {'CP', 'STR', 'Striatum', 'DMS', 'DLS', 'PS'}) %striatum, classification as in Peters et al., Nature, 2021
    unitClassif(ephysProperties.waveformDuration_peakTrough_us > paramEP.templateDuration_CP_threshold &...
        ephysProperties.postSpikeSuppression_ms < paramEP.postSpikeSup_CP_threshold) = {'MSN'};

    unitClassif(ephysProperties.waveformDuration_peakTrough_us <= paramEP.templateDuration_CP_threshold &...
        ephysProperties.proplongISI <= paramEP.propISI_CP_threshold) = {'FSI'};

    unitClassif(ephysProperties.waveformDuration_peakTrough_us > paramEP.templateDuration_CP_threshold &...
        ephysProperties.postSpikeSuppression_ms >= paramEP.postSpikeSup_CP_threshold) = {'TAN'};

    unitClassif(ephysProperties.waveformDuration_peakTrough_us <= paramEP.templateDuration_CP_threshold &...
        ephysProperties.proplongISI > paramEP.propISI_CP_threshold) = {'UIN'};

    figure();
    scatter3(ephysProperties.waveformDuration_peakTrough_us, ephysProperties.postSpikeSuppression_ms, ephysProperties.proplongISI, 4, 'filled'); hold on;
    set(gca, 'YDir', 'reverse' );

elseif ismember(region, {'Ctx', 'Cortex', 'Cortical'}) % cortex, classification as in Peters et al., Cell Reports, 2022
     unitClassif(ephysProperties.waveformDuration_peakTrough_us > paramEP.templateDuration_Ctx_threshold) = {'Wide-spiking'};

     unitClassif(ephysProperties.waveformDuration_peakTrough_us <= paramEP.templateDuration_Ctx_threshold) = {'Narrow-spiking'};


% elseif ismember(region, {'GPe', 'Globus Pallidus external'}) % GPe - work
% in progress  
end

end
