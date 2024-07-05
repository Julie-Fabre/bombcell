/* CCGEngine.c /*
/* This is a bare-bones C program whos purpose is to compute
   Multi-unit cross correlograms quickly.  Not intended for
   use on its own.  It is designed to be wrapped by a MATLAB
   function.

   Usage - [CCG, PAIRS] = CCGEngine(TIMES, MARKS, BINSIZE, HALFBINS)
   TIMES ( is the name of a binary file containing N doubles giving the spike times
   MARKS is the name of a binary file containing N unsigned ints giving the spike markers.  Don't use zero!
   BINSIZE is a number giving the size of the ccg bins in TIMES units
   HALFBINS is the number of bins to compute - so there are a total of nBins = 1+2*HALFBINS bins
   
   These should be: double, uint32,double,uint32

   NB The spikes MUST be sorted.

   CCG contains unsigned ints containing the counts in each bin
   It is like a 3d array, indexed by [nBins*nMarks*Mark1 + nBins*Mark2 + Bin]
   
   PAIRS contains the spike numbers of every pair of spikes included in the ccg.

   If you think this program is anal, you're right.  Use the MATLAB wrapper instead.
 */

#include <string.h>
#include "mex.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define CHECK
#define STRLEN 10000
#define PAIRBLOCKSIZE 100000000

unsigned int *Pairs;
unsigned int PairCnt, PairSz;
void AddPair(unsigned int n1, unsigned int n2) {
	unsigned int *pui;

	if (PairSz==0) {
/*		mexPrintf("Allocating pair memory\n"); */
		Pairs = mxMalloc(PAIRBLOCKSIZE*sizeof(unsigned int));
		PairSz = PAIRBLOCKSIZE;
		if (!Pairs) {
			/*mxErrMsgTxt("Could not allocate memory for pairs");*/
			mexErrMsgIdAndTxt("someerror", "Could not allocate memory for pairs");
		}
	}
	/* check if array is full, if so add more memory*/
	if(PairCnt>=PairSz) {
/*		mexPrintf("Reallocating pair memory ... ");
		PairSz += PAIRBLOCKSIZE;
		pui = mxRealloc(Pairs, PairSz);
		mexPrintf("got %x\n", pui);
		if (!pui) {
			mxFree(Pairs);

			mxErrMsgTxt("Could not reallocate memory for pairs");
		}
		Pairs = pui;
*/
		/*mxErrMsgTxt("Too many pairs");*/
		mexErrMsgIdAndTxt("someerror", "Too many pairs");
	}
	Pairs[PairCnt++] = n1;
	Pairs[PairCnt++] = n2;
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {

	unsigned int nSpikes, nMarks, HalfBins, nBins, i, CountArraySize, CountIndex;
	double *Times;
	double BinSize, FurthestEdge;
	unsigned int *Marks, Mark, *Count;
	unsigned int CenterSpike, Mark1, Mark2, Bin;
	int SecondSpike; /* we want to let it go negative so we can stop it there... */
	double Time1, Time2;
	char errstr[STRLEN];

	/* global variables are not initialized on each call to mex fn!! */
	PairCnt = 0; PairSz = 0;

	if (nrhs!=4) {
		/*mxErrMsgTxt("Must have 4 arguments\nBut listen:  You don't want to use this program.\nUse the MATLAB wrapper function CCG instead.\n");*/
		mexErrMsgIdAndTxt("someerror", "Must have 4 arguments\nBut listen:  You don't want to use this program.\nUse the MATLAB wrapper function CCG instead.\n");
	}
	if (mxGetClassID(prhs[0])!=mxDOUBLE_CLASS || mxGetClassID(prhs[1])!=mxUINT32_CLASS
		|| mxGetClassID(prhs[2])!=mxDOUBLE_CLASS || mxGetClassID(prhs[3])!=mxUINT32_CLASS ) {
		/*mxErrMsgTxt("Arguments are wrong type\n");*/
		mexErrMsgIdAndTxt("someerror", "Arguments are wrong type\n");
	}


	/* get arguments */
	Times = mxGetPr(prhs[0]);
	Marks = (unsigned int *) mxGetPr(prhs[1]);
	nSpikes = mxGetNumberOfElements(prhs[0]);
	if (mxGetNumberOfElements(prhs[1])!=nSpikes) {
		/*mxErrMsgTxt("Number of marks ~= number of spikes");*/
		mexErrMsgIdAndTxt("someerror", "Number of marks ~= number of spikes");
	}
	BinSize = mxGetScalar(prhs[2]);
	HalfBins = (unsigned int) mxGetScalar(prhs[3]);

	/* derive other constants */
	nBins = 1+2*HalfBins;
	FurthestEdge = BinSize * (HalfBins + 0.5);


	/* count nMarks */
	nMarks = 0;
	for(i=0; i<nSpikes; i++) {
		Mark = Marks[i];
		if (Mark>nMarks) nMarks = Mark;
		if (Mark==0) {
			/*mxErrMsgTxt("CCGEngine: No zeros allowed in Marks");*/
			mexErrMsgIdAndTxt("someerror", "CCGEngine: No zeros allowed in Marks");
			abort();
		}
	}

	/* allocate output array */
	CountArraySize = nMarks * nMarks * nBins;
	plhs[0] = mxCreateNumericMatrix(CountArraySize, 1, mxUINT32_CLASS, mxREAL);
	Count = (unsigned int *) mxGetPr(plhs[0]);

	if (!Times || !Marks || !Count) {
		/*mxErrMsgTxt("CCGEngine could not allocate memory!\n");*/
		mexErrMsgIdAndTxt("someerror", "CCGEngine could not allocate memory!\n");
	}

	/* Now the main program .... */


	for(CenterSpike=0; CenterSpike<nSpikes; CenterSpike++) {
		Mark1 = Marks[CenterSpike];
		Time1 = Times[CenterSpike];

		/* Go back from CenterSpike */
		for(SecondSpike=CenterSpike-1; SecondSpike>=0; SecondSpike--) {
			Time2 = Times[SecondSpike];

			/* check if we have left the interesting region */
			if(fabs(Time1 - Time2) > FurthestEdge) break;

			/* calculate bin */
			Bin = HalfBins + (int)(floor(0.5+(Time2-Time1)/BinSize));

			Mark2 = Marks[SecondSpike];
			CountIndex = nBins*nMarks*(Mark1-1) + nBins*(Mark2-1) + Bin;
#ifdef CHECK
			if (CountIndex<0 || CountIndex >= CountArraySize) {
				sprintf(errstr, "err a: t1 %f t2 %f m1 %d m2 %d Bin %d, index %d out of bounds",
					Time1, Time2, Mark1, Mark2, Bin, CountIndex);
				/*mxErrMsgTxt(errstr);*/
				mexErrMsgIdAndTxt("someerror", errstr);
			}
#endif

			/* increment count */
			Count[CountIndex]++;
			if (nlhs>=2) AddPair(CenterSpike, SecondSpike);

		}

		/* Now do the same thing going forward... */
		for(SecondSpike=CenterSpike+1; SecondSpike<nSpikes; SecondSpike++) {
			Time2 = Times[SecondSpike];

			/* check if we have left the interesting region */
			if(fabs(Time1 - Time2) >= FurthestEdge) break;

			/* calculate bin */
			Bin = HalfBins + (unsigned int)(floor(0.5+(Time2-Time1)/BinSize));

			Mark2 = Marks[SecondSpike];
			CountIndex = nBins*nMarks*(Mark1-1) + nBins*(Mark2-1) + Bin;

#ifdef CHECK
			if (CountIndex<0 || CountIndex >= CountArraySize) {
				sprintf(errstr, "err b: t1 %f t2 %f m1 %d m2 %d Bin %d, index %d out of bounds",
					Time1, Time2, Mark1, Mark2, Bin, CountIndex);
				/*mxErrMsgTxt(errstr);*/
				mexErrMsgIdAndTxt("someerror", errstr);
			}
#endif

			/* increment count */
			Count[CountIndex]++;
			if (nlhs>=2) AddPair(CenterSpike, SecondSpike);

		}
	}

	if (nlhs>=2) {
		plhs[1] = mxCreateNumericMatrix(PairCnt, 1, mxUINT32_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[1]), (void *)Pairs, PairCnt*sizeof(unsigned int));
		mxFree(Pairs);
	}



	/* sayonara */
}
