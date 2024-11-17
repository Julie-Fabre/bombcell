import numpy as np
import pandas as pd
import os

def load_bc_results(bc_path):
    """
    Loads saved BombCell results

    Parameters
    ----------
    bc_path : string
        The absolute path to the directory which has the saved BombCell results

    Returns
    -------
    tuple (df, df, df)
        The data frames of hte BombCell results
    """
    #Files
    #BombCell params ML
    param_path = os.path.join(bc_path, '_bc_parameters._bc_qMetrics.parquet')
    if os.path.exists(param_path):
        param = pd.read_parquet(param_path)
    else:
        print('Paramater file not found')


    #BombCell quality metrics
    quality_metrics_path = os.path.join(bc_path, 'templates._bc_qMetrics.parquet')
    if os.path.exists(quality_metrics_path):
        quality_metrics = pd.read_parquet(quality_metrics_path)
    else:
        print('Quality Metrics file not found')


    #BombCell fration RPVS all TauR
    fractions_RPVs_all_taur_path = os.path.join(bc_path, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet')
    if os.path.exists(fractions_RPVs_all_taur_path):
        fractions_RPVs_all_taur = pd.read_parquet(fractions_RPVs_all_taur_path)
    else:
        print('Fraction RPVs all TauR file not found')
        fractions_RPVs_all_taur = None

    return param, quality_metrics, fractions_RPVs_all_taur

def get_quality_unit_type(param, quality_metrics):
    """
    Assign each unit a type based of its' quality metrics.
    unit_type == 0 all noise units
    unit_type == 1 all good units
    unit_type == 2 all mua units
    unit_type == 3 all non-somatic units (if split somatic units its good non-somatic units)
    unit_type == 4 (if split somatic units its mua non-somatic units)


    Parameters
    ----------
    param : df
        The param dataframe from ML BombCell 
    quality_metrics : df
        The quality metrics dataframefrom ML BombCell

    Returns
    -------
    tuple (np array, np array)
        Two array of the unit types one as number the other as strings
    """

    #converting dataframes to dictionary of numpy arrays
    quality_metrics = dict(zip(quality_metrics.columns, quality_metrics.values.T))
    param = dict(zip(param.columns, param.values.T))

    
    #Testing for non-somatic waveforms
    is_non_somatic = np.zeros(quality_metrics['nPeaks'].shape[0])

    is_non_somatic[(quality_metrics['mainTrough_size'] / np.max((quality_metrics['mainPeak_before_size'] , quality_metrics['mainPeak_after_size']), axis = 0)) < param['minTroughToPeakRatio']] = 1 

    is_non_somatic[(quality_metrics['mainPeak_before_size'] / quality_metrics['mainPeak_after_size'])  > param['firstPeakRatio']] = 1

    is_non_somatic[(quality_metrics['mainPeak_before_size'] * param['firstPeakRatio'] > quality_metrics['mainPeak_after_size']) & (quality_metrics['mainPeak_before_width'] < param['minWidthFirstPeak']) \
        & (quality_metrics['mainPeak_before_size'] * param['minMainPeakToTroughRatio'] > quality_metrics['mainTrough_size']) & (quality_metrics['mainTrough_width'] < param['minWidthMainTrough'])] = 1

    #Test all quality metrics
    ## categorise units
    # unit_type == 0 all noise units
    # unit_type == 1 all good units
    # unit_type == 2 all mua units
    # unit_type == 3 all non-somatic units (if split somatic units its good non-somatic units)
    # unit_type == 4 (if split somatic units its mua non-somatic units)

    unit_type = np.full(quality_metrics['nPeaks'].shape[0], np.nan)

    # classify noise
    unit_type[np.isnan(quality_metrics['nPeaks'])] = 0
    unit_type[quality_metrics['nPeaks']  > param['maxNPeaks']] = 0
    unit_type[quality_metrics['nTroughs'] > param['maxNTroughs']] = 0
    unit_type[quality_metrics['waveformDuration_peakTrough'] < param['minWvDuration']] = 0
    unit_type[quality_metrics['waveformDuration_peakTrough'] > param['maxWvDuration']] = 0
    unit_type[quality_metrics['waveformBaselineFlatness'] > param['maxWvBaselineFraction']] = 0
    unit_type[quality_metrics['spatialDecaySlope'] < param['minSpatialDecaySlopeExp']] = 0
    unit_type[quality_metrics['spatialDecaySlope'] > param['maxSpatialDecaySlopeExp']] = 0

    # classify as mua
    #ALL or ANY?
    unit_type[np.logical_and(quality_metrics['percentageSpikesMissing_gaussian'] > param['maxPercSpikesMissing'], np.isnan(unit_type))] = 2
    unit_type[np.logical_and(quality_metrics['nSpikes'] < param['minNumSpikes'] , np.isnan(unit_type))] = 2
    unit_type[np.logical_and(quality_metrics['fractionRPVs_estimatedTauR']> param['maxRPVviolations'], np.isnan(unit_type))] = 2
    unit_type[np.logical_and(quality_metrics['presenceRatio'] < param['minPresenceRatio'] , np.isnan(unit_type))] = 2

    if param['extractRaw'].astype(int) == 1:
        unit_type[np.logical_and(quality_metrics['rawAmplitude'] < param['min_aminAmplitudemplitude'] , np.isnan(unit_type))] = 2
        unit_type[np.logical_and(quality_metrics['signalToNoiseRatio'] < param['minSNR'] , np.isnan(unit_type))] = 2

    if param['computeDrift'].astype(int) == 1:
        unit_type[np.logical_and(quality_metrics['maxDriftEstimate'] > param['maxDrift'] , np.isnan(unit_type))] = 2

    if param['computeDistanceMetrics'].astype(int) == 1 & ~np.isnan(param['isoDmin'].astype(int)):
        unit_type[np.logical_and(quality_metrics['isoD'] > param['isoDmin'] , np.isnan(unit_type))] = 2
        unit_type[np.logical_and(quality_metrics['Lratio'] > param['lratioMax'] , np.isnan(unit_type))] = 2

    unit_type[np.isnan(unit_type)] = 1 # SINGLE SEXY UNIT

    if param['splitGoodAndMua_NonSomatic'].astype(int) == 1:
        unit_type[np.logical_and(is_non_somatic == 1, unit_type == 1)] = 3 # Good non-somatic
        unit_type[np.logical_and(is_non_somatic == 1, unit_type == 2)] = 4 # MUA non-somatic
    else:
        unit_type[np.logical_and(is_non_somatic == 1, unit_type != 0)] = 3 # Good non-somatic

    #Have unit types as strings as well
    unit_type_string = np.full(unit_type.size, '', dtype = object)
    unit_type_string[unit_type == 0] = 'NOISE'
    unit_type_string[unit_type == 1] = 'GOOD'
    unit_type_string[unit_type == 2] = 'MUA'

    if param['splitGoodAndMua_NonSomatic'].astype(int) == 1:
        unit_type_string[unit_type == 3] = 'NON-SOMA GOOD'
        unit_type_string[unit_type == 4] = 'NON-SOMA MUA'
    else:
        unit_type_string[unit_type == 3] = 'NON-SOMA'
    
    return unit_type, unit_type_string
