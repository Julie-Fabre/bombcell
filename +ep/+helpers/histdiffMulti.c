/*

  HISTDIFF.C	fast histogramming code

  [C, B] = HISTDIFF (stimes, ev, BINS).

  Calculate histogram of differences stimes(i) - ev(j) for all i and
  j.  C is the counts per bin, B is the centers of the bins.

  stimes and ev are each treated as a single array

  BINS is a vector of bin boundaries (NOTE: this is different to the
  mathworks HIST which takes bin centers)  If BINS is a scalar it is 
  taken to be a bin count.

  If BINS is specified as a count, or as a regularly spaced vector, a
  rapid binning algorithm is used that depends on this and is linear
  in ndata.  If BINS is irregularly spaced the algorithm used takes
  time proportional to ndata*nbins.

*/

#include <stdio.h>
#include <math.h>
#include "mex.h"

/* Arguments */

#define	STIMES_IN   prhs[0]
#define	EV_IN   prhs[1]
#define	BINS_IN	   prhs[2]
#define GROUPS_IN    prhs[3]
#define N_GROUPS    prhs[4]
#define	C_OUT	   plhs[0]
#define	B_OUT	   plhs[1]

/* Auxilliary prototypes */
static void findext (double *, int, double *, int, double *, double *);
static int  chckord (double *, int, double *, int);


/******************************************************************************
  WORKHORSE FUNCTIONS
  */


/*
 *  reghist: bins are regularly spaced (takes min boundary, size of
 *  bin and number)
 */

static void reghist(double	 stimes[], 
		    int		 nSp, 
		    double	 ev[], 
		    int		 nEv, 
		    double	 min, 
		    double	 size, 
		    int		 nbins,
		    double	 cnts[],
		    double	 ctrs[])
{
  int i, j;
  double max = min + size * nbins;
  /*mexPrintf("\nusing reghist\n");*/
  for (i = 0; i < nbins; i++) {
    cnts[i] = 0;
    ctrs[i] = min + i*size + size/2;
  }

  for (i = 0; i < nSp; i++) {
    for (j = 0; j < nEv; j++) {
      register double diff = stimes[i] - ev[j];
      if (diff < min || diff >= max)
	continue;
      cnts[(int) ((diff-min)/size)] ++;
    }
  }
}



/*
 *  ordhist: bins are regularly spaced (see reghist) and each data
 *  stream is ordered from smallest to largest.
 */

static void ordhist(double	 stimes[], 
		    int		 nSp, 
		    double	 ev[], 
		    int		 nEv,
        double   dataGr[], 
		    double	 min, 
		    double	 size, 
		    int		 nbins,
		    double	 cnts[],
		    double	 ctrs[])
{
  int i, j, jmin;
  double max = min + size * nbins;
  double diff;
  /*mexPrintf("\nusing ordhist\n");
  mexPrintf("\nnbins = %d\n", nbins);
  mexPrintf("\nnEv = %d\n", nEv);*/

  for (i = 0; i < nbins; i++) {
    cnts[i] = 0;
    ctrs[i] = min + i*size + size/2;
  }

  jmin = 0;

  for (i = 0; i < nSp; i++) {
    for (j = jmin; j < nEv && (stimes[i] - ev[j]) >= max; j++)
      ;
    
    jmin = j;

    for (j = jmin; j < nEv && (diff = stimes[i] - ev[j]) > min; j++)
      /* The way this works it to create one long vector of counts, using something like subs2ind. reshape later*/
      cnts[(int) ((int)((diff-min)/size)+nbins*j+nEv*nbins*dataGr[i])] ++;
  }
}



/*
 *  binhist: general purpose histogrammer: bins must be ordered but
 *  may be irregular data need not be ordered.
 */

static void binhist(double	 stimes[], 
		    int		 nSp, 	
		    double	 ev[], 
		    int		 nEv, 
		    double	 bins[],
		    int		 nbins,
		    double	 cnts[],
		    double	 ctrs[])
{

  /*mexPrintf("\nusing binhist\n");*/
  int i, j, b;

  for (b = 0; b < nbins; b++) {
    cnts[b] = 0;
    ctrs[b] = (bins[b] + bins[b+1])/2;
  }

  for (i = 0; i < nSp; i++)
    for (j = 0; j < nEv; j++) {
      for (b = 0; b < nbins; b++)
      if ((stimes[i]-ev[j]) >= bins[b] && (stimes[i]-ev[j]) < bins[b+1]) 
	cnts[b] ++;
    }
}


/******************************************************************************
  INTERFACE FUNCTION
  */

void mexFunction(int		nlhs,
		 mxArray	*plhs[],
		 int		nrhs,
		 const mxArray	*prhs[])
{
  double	*stimes = NULL, *ev = NULL, *dataGr = NULL;
  double	*bins = NULL, *cnts = NULL, *ctrs = NULL;
  int		nSp, nEv, nbins, nG;
  double	min, max, size = -1;
  double	*t,*y;
  unsigned int	i,m,n;

  /* Check numbers of arguments */
  if (nrhs == 0) {
    mexErrMsgTxt("HISTDIFF: no data to histogram");
  } else if (nrhs > 5) {
    mexErrMsgTxt("HISTDIFF: too many arguments.");
  }
  if (nlhs < 2) {
    mexErrMsgTxt("HISTDIFF: must be called with two output arguments");
  }

  /* Get data */
  m = mxGetM(STIMES_IN);
  n = mxGetN(STIMES_IN);
  nSp = m*n;
  if (!mxIsNumeric(STIMES_IN) || mxIsComplex(STIMES_IN) || 
      mxIsSparse(STIMES_IN)  || !mxIsDouble(STIMES_IN) || 
      m*n == 0) {
    mexErrMsgTxt("HISTDIFF: data must be a full real valued matrix.");
  }
  stimes = mxGetPr(STIMES_IN);

  m = mxGetM(EV_IN);
  n = mxGetN(EV_IN);
  nEv = m*n;
  if (!mxIsNumeric(EV_IN) || mxIsComplex(EV_IN) || 
      mxIsSparse(EV_IN)  || !mxIsDouble(EV_IN) || 
      m*n == 0) {
    mexErrMsgTxt("HISTDIFF: data must be a full real valued matrix.");
  }
  ev = mxGetPr(EV_IN);


  /* Get bin specification */
  m = mxGetM(BINS_IN);
  n = mxGetN(BINS_IN);

  if (!mxIsNumeric(BINS_IN) || mxIsComplex(BINS_IN) ||
      mxIsSparse(BINS_IN) || !mxIsDouble(BINS_IN) ||
      (m != 1 && n != 1)) {
    mexErrMsgTxt("HISTDIFF: bins spec must be a real scalar or vector");
  }
    
  if (m == 1 && n == 1) {	/* number of bins */
    nbins = (int)*(double *)mxGetPr(BINS_IN);
    findext (stimes, nSp, ev, nEv, &min, &max);
    size = (max-min)/nbins;
  } else {			/* vector of bin boundaries */
    nbins = n*m - 1;
    bins  = mxGetPr (BINS_IN);

    /* check to see if spacing is regular -- if so use fast algorithm */
    size = bins[1] - bins[0];
    for (i = 1; i < nbins; i++)
      if (fabs(bins[i+1] - bins[i] - size) > 1e-3*size) {
	size = 0;
	break;
      }
    if (size) {
      min = bins[0];
    }
  }

  dataGr = mxGetPr(GROUPS_IN);
  nG = (int)*(double *)mxGetPr(N_GROUPS);

  /* Create output matrices */
  C_OUT = mxCreateDoubleMatrix(1, nbins*nEv*nG, mxREAL); 
  cnts = mxGetPr(C_OUT);

  B_OUT = mxCreateDoubleMatrix(1, nbins, mxREAL);
  ctrs = mxGetPr(B_OUT);
  

  /* Do the actual computations in a subroutine */

  if (size) {
    if (chckord (stimes, nSp, ev, nEv))
      ordhist(stimes, nSp, ev, nEv, dataGr, min, size, nbins, cnts, ctrs);
    else
      reghist(stimes, nSp, ev, nEv, min, size, nbins, cnts, ctrs);
  } else {
    binhist(stimes, nSp, ev, nEv, bins, nbins, cnts, ctrs);
  }

  return;
}


/******************************************************************************
  AUXILLIARY FUNCTIONS
  */


static void findext(double	 stimes[], 
		    int		 nSp, 
		    double	 ev[], 
		    int		 nEv, 
		    double	*min, 
		    double      *max)
{
  int i,j;

  *min = *max = stimes[0] - ev[0];
  for (i = 1; i < nSp; i++) {
    for (j = 1; j < nEv; j++) {
      double diff = stimes[i] - ev[j];
      if (diff < *min) *min = diff;
      if (diff > *max) *max = diff;
    }
  }
}


static int  chckord (double	stimes[], 
		    int		 nSp, 
		    double	 ev[], 
		    int		 nEv)
{
  int i;

  for (i = 1; i < nSp; i++)
    if (stimes[i] < stimes[i-1])
      return 0;

  for (i = 1; i < nEv; i++)
    if (ev[i] < ev[i-1])
      return 0;

  return 1;
}
