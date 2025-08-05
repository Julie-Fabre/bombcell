import os, sys
from pathlib import Path
from pprint import pprint 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bombcell as bc


# Replace with your kilosort directory
ks_dir = Path('/data1/hyunwoo/tDNMS_project/data/AC01_ephys/AC01_01232025_g0/spikeinterface_imec1/group_0/kilosort4_output/sorter_output')
#ks_dir = r"X:\hyunwoo\tDNMS_project\data\AC01_ephys\AC01_01232025_g0\spikeinterface_imec1\group_0\kilosort4_output\sorter_output"
#ks_dir = r"X:\hyunwoo\tDNMS_project\data\AC07_ephys\AC07_05222025_g0\spikeinterface\group_0\kilosort4_output\sorter_output"

# Set bombcell's output directory
save_path = Path(ks_dir) / "bombcell"

print(f"Using kilosort directory: {ks_dir}")
## For Neuropixels probes, provide raw and meta files
## Leave 'None' if no raw data. Ideally, your raw data is common-average-referenced and
# the channels are temporally aligned to each other (this can be done with CatGT)
raw_file_path = None#"/home/julie/Dropbox/Example datatsets/JF093_2023-03-09_site1/site1/2023-03-09_JF093_g0_t0_bc_decompressed.imec0.ap.bin" # ks_dir
meta_file_path = None#"/home/julie/Dropbox/Example datatsets/JF093_2023-03-09_site1/site1/2023-03-09_JF093_g0_t0.imec0.ap.meta"
#raw_file_path = r"X:\hyunwoo\tDNMS_project\data\AC01_ephys\AC01_01232025_g0\AC01_01232025_g0_imec0\AC01_01232025_g0_t0.imec0.ap.bin"
#meta_file_path = r"X:\hyunwoo\tDNMS_project\data\AC01_ephys\AC01_01232025_g0\AC01_01232025_g0_imec0\AC01_01232025_g0_t0.imec0.ap.meta"

## Get default parameters - we will see later in the notebook how to assess and fine-tune these
param = bc.get_default_parameters(ks_dir, 
                                  raw_file=raw_file_path,
                                  meta_file=meta_file_path,
                                  kilosort_version=4)

print("Bombcell parameters:")

# for instance, you might to change classification thresholds like: 
# param["maxRPVviolations"] = 0.1
#  or which quality metrics are computed (by default these are not): 
param["computeDistanceMetrics"] = 0
param["computeDrift"] = 0
param["maxWvBaselineFraction"] = 2.0

# MUA
param["maxRPVviolations"] = 30
param["maxPercSpikesMissing"] = 20
param["minNumSpikes"] = 300
param["minPresenceRatio"] = 0.7

# noise
param["minSpatialDecaySlopeExp"] = 0.01
param["maxSpatialDecaySlopeExp"] = 0.1

# non-somatic
param["maxMainPeakToTroughRatio_nonSomatic"] = 0.8

#  or whether the recording is split into time chunks to detemrine "good" time chunks: 
# param["computeTimeChunks"] = 0
# full list in the wiki or in the bc.get_default_parameters function


pprint(param)

(
    quality_metrics,
    param,
    unit_type,
    unit_type_string,
) = bc.run_bombcell(
    ks_dir, save_path, param
)


# Use the output summary plots (below) to see if the 
# quality metric thresholds seem roughly OK for your 
# data (i.e. there isn't one threshold removing all 
# units or a threshold may below that removes none)
# more details on these output plots in the wiki:
# https://github.com/Julie-Fabre/bombcell/wiki/Summary-output-plots

# quality metric values
#quality_metrics_table = pd.DataFrame(quality_metrics)
#quality_metrics_table.insert(0, 'Bombcell_unit_type', unit_type_string)
#quality_metrics_table

# boolean table, if quality metrics pass threshold given parameters
#boolean_quality_metrics_table = bc.make_qm_table(
#    quality_metrics, param, unit_type_string
#)
#boolean_quality_metrics_table


