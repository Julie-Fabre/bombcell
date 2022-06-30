from phylib.io.traces import get_ephys_reader
from pathlib import Path
import glob
import matlab

def Ephys_Reader_FromMatlab(datapath,start_time,end_time):
	#tmpdatapath = f'{datapath}'
	#print(datapath)
	#print(tmpdatapath)
	#datapath = glob.glob(tmpdatapath)
	#print(datapath)
	#print(start_time)
	#print(end_time)
	data = get_ephys_reader(datapath, n_channels=385, dtype='int16')	
	chunk = data[int(start_time):int(end_time)] 
	chunk = matlab.uint16(chunk)
	#print(chunk)
	return chunk

#datapath = '//128.40.198.18/Subjects/EB014/2022-05-05/ephys/2022-05-05_EB014_1_g0/2022-05-05_EB014_1_g0_imec0/2022-05-05_EB014_1_g0_t0.imec0.ap.cbin'
#start_time = 0
#end_time = 5000
#channel = 384
chunk = Ephys_Reader_FromMatlab(datapath,start_time,end_time)
#print(chunk+2)

