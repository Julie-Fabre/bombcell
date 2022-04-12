/*

  HISTDIFF.C	fast histogramming code

  [C, B] = HISTDIFF (DATA1, DATA2, BINS).

  Calculate histogram of differences DATA1(i) - DATA2(j) for all i and
  j.  C is the counts per bin, B is the centers of the bins.

  DATA1 and DATA2 are each treated as a single array

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

#define	DATA1_IN   prhs[0]
#define	DATA2_IN   prhs[1]
#define	BINS_IN	   prhs[2]
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

static void reghist(double	 data1[], 
		    int		 ndata1, 
		    double	 data2[], 
		    int		 ndata2, 
		    double	 min, 
		    double	 size, 
		    int		 nbins,
		    double	 cnts[],
		    double	 ctrs[])
{
  int i, j;
  double max = min + size * nbins;

  for (i = 0; i < nbins; i++) {
    cnts[i] = 0;
    ctrs[i] = min + i*size + size/2;
  }

  for (i = 0; i < ndata1; i++) {
    for (j = 0; j < ndata2; j++) {
      register double diff = data1[i] - data2[j];
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

static void ordhist(double	 data1[], 
		    int		 ndata1, 
		    double	 data2[], 
		    int		 ndata2, 
		    double	 min, 
		    double	 size, 
		    int		 nbins,
		    double	 cnts[],
		    double	 ctrs[])
{
  int i, j, jmin;
  double max = min + size * nbins;
  double diff;

  for (i = 0; i < nbins; i++) {
    cnts[i] = 0;
    ctrs[i] = min + i*size + size/2;
  }

  jmin = 0;

  for (i = 0; i < ndata1; i++) {
    for (j = jmin; j < ndata2 && (data1[i] - data2[j]) >= max; j++)
      ;
    
    jmin = j;

    for (j = jmin; j < ndata2 && (diff = data1[i] - data2[j]) > min; j++)
      cnts[(int) ((diff-min)/size)] ++;
  }
}



/*
 *  binhist: general purpose histogrammer: bins must be ordered but
 *  may be irregular data need not be ordered.
 */

static void binhist(double	 data1[], 
		    int		 ndata1, 	
		    double	 data2[], 
		    int		 ndata2, 
		    double	 bins[],
		    int		 nbins,
		    double	 cnts[],
		    double	 ctrs[])
{
  int i, j, b;

  for (b = 0; b < nbins; b++) {
    cnts[b] = 0;
    ctrs[b] = (bins[b] + bins[b+1])/2;
  }

  for (i = 0; i < ndata1; i++)
    for (j = 0; j < ndata2; j++) {
      for (b = 0; b < nbins; b++)
      if ((data1[i]-data2[j]) >= bins[b] && (data1[i]-data2[j]) < bins[b+1]) 
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
  double	*data1 = NULL, *data2 = NULL;
  double	*bins = NULL, *cnts = NULL, *ctrs = NULL;
  int		ndata1, ndata2, nbins;
  double	min, max, size = -1;
  double	*t,*y;
  unsigned int	i,m,n;

  /* Check numbers of arguments */
  if (nrhs == 0) {
    mexErrMsgTxt("HISTDIFF: no data to histogram");
  } else if (nrhs > 3) {
    mexErrMsgTxt("HISTDIFF: too many arguments.");
  }
  if (nlhs < 2) {
    mexErrMsgTxt("HISTDIFF: must be called with two output arguments");
  }

  /* Get data */
  m = mxGetM(DATA1_IN);
  n = mxGetN(DATA1_IN);
  ndata1 = m*n;
  if (!mxIsNumeric(DATA1_IN) || mxIsComplex(DATA1_IN) || 
      mxIsSparse(DATA1_IN)  || !mxIsDouble(DATA1_IN) || 
      m*n == 0) {
    mexErrMsgTxt("HISTDIFF: data must be a full real valued matrix.");
  }
  data1 = mxGetPr(DATA1_IN);

  m = mxGetM(DATA2_IN);
  n = mxGetN(DATA2_IN);
  ndata2 = m*n;
  if (!mxIsNumeric(DATA2_IN) || mxIsComplex(DATA2_IN) || 
      mxIsSparse(DATA2_IN)  || !mxIsDouble(DATA2_IN) || 
      m*n == 0) {
    mexErrMsgTxt("HISTDIFF: data must be a full real valued matrix.");
  }
  data2 = mxGetPr(DATA2_IN);


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
    findext (data1, ndata1, data2, ndata2, &min, &max);
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


  /* Create output matrices */
  C_OUT = mxCreateDoubleMatrix(1, nbins, mxREAL);
  cnts = mxGetPr(C_OUT);

  B_OUT = mxCreateDoubleMatrix(1, nbins, mxREAL);
  ctrs = mxGetPr(B_OUT);
  

  /* Do the actual computations in a subroutine */

  if (size) {
    if (chckord (data1, ndata1, data2, ndata2))
      ordhist(data1, ndata1, data2, ndata2, min, size, nbins, cnts, ctrs);
    else
      reghist(data1, ndata1, data2, ndata2, min, size, nbins, cnts, ctrs);
  } else {
    binhist(data1, ndata1, data2, ndata2, bins, nbins, cnts, ctrs);
  }

  return;
}


/******************************************************************************
  AUXILLIARY FUNCTIONS
  */


static void findext(double	 data1[], 
		    int		 ndata1, 
		    double	 data2[], 
		    int		 ndata2, 
		    double	*min, 
		    double      *max)
{
  int i,j;

  *min = *max = data1[0] - data2[0];
  for (i = 1; i < ndata1; i++) {
    for (j = 1; j < ndata2; j++) {
      double diff = data1[i] - data2[j];
      if (diff < *min) *min = diff;
      if (diff > *max) *max = diff;
    }
  }
}


static int  chckord (double	data1[], 
		    int		 ndata1, 
		    double	 data2[], 
		    int		 ndata2)
{
  int i;

  for (i = 1; i < ndata1; i++)
    if (data1[i] < data1[i-1])
      return 0;

  for (i = 1; i < ndata2; i++)
    if (data2[i] < data2[i-1])
      return 0;

  return 1;
}
