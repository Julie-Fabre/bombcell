import os, sys
from pathlib import Path
from pprint import pprint 
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bombcell as bc


def main(argv):    
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    data_dir = Path(args.directory)
    #data_dir = Path('/data1/hyunwoo/tDNMS_project/data/AC01_ephys/AC01_01232025_g0/')
    ks_dir = [p for p in data_dir.rglob('sorter_output') if p.is_dir()]
    raw_file_path = None#"/home/julie/Dropbox/Example datatsets/JF093_2023-03-09_site1/site1/2023-03-09_JF093_g0_t0_bc_decompressed.imec0.ap.bin" # ks_dir
    meta_file_path = None#"/home/julie/Dropbox/Example datatsets/JF093_2023-03-09_site1/site1/2023-03-09_JF093_g0_t0.imec0.ap.meta"

    # set parameters
    ## Get default parameters
    param = bc.get_default_parameters(ks_dir[0], 
                                      raw_file=raw_file_path,
                                      meta_file=meta_file_path,
                                      kilosort_version=4)
    param["computeDistanceMetrics"] = 0
    param["computeDrift"] = 0
    
    # MUA
    param["maxRPVviolations"] = 30 # using kilosort contamPct metric
    param["maxPercSpikesMissing"] = 100 # default: 20
    param["minNumSpikes"] = 1 # default: 300
    param["minPresenceRatio"] = 0 # default: 0.7
    # noise
    param["minSpatialDecaySlopeExp"] = 0.01 # default
    param["maxSpatialDecaySlopeExp"] = 0.1 # default
    param["maxNPeaks"] = 2 # default
    param["maxNTroughs"] = 2 # default: 1
    param["minWvDuration"] = 100 # default
    param["maxWvDuration"] = 1150 # default
    param["maxWvBaselineFraction"] = 2.0 # default: 0.3
    param["maxScndPeakToTroughRatio_noise"] = 1.0 # default: 0.8
    # non-somatic
    param["maxMainPeakToTroughRatio_nonSomatic"] = 0.8 # default
    print("Bombcell parameters:")
    pprint(param)
    
    # run bombcell
    for p in ks_dir:
        save_path = p / 'bombcell'
        print(f"Using kilosort directory: {p}")
        (
            quality_metrics,
            param,
            unit_type,
            unit_type_string,
        ) = bc.run_bombcell(
            p, save_path, param
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


if __name__ == '__main__':
    main(sys.argv[1:])

