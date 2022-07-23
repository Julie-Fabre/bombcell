from pathlib import Path
import matlab
import numpy as np
from mtscomp import decompress

def bc_uncompressBin(datapath,JsonPath,savepath):
	
	# Use decompress from mtscomp
	decompress(datapath,cmeta=JsonPath,out=savepath)
	
	success = 1
	return success

success = bc_uncompressBin(datapath,JsonPath,savepath)

