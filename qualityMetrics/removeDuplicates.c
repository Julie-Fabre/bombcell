#include <mex.h>
#include <math.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Input data
    double *spikeTimes_samples = mxGetPr(prhs[0]);
    double *spikeTemplates = mxGetPr(prhs[1]);
    double *templateAmplitudes = mxGetPr(prhs[2]);
    double duplicateSpikeWindow_samples = mxGetScalar(prhs[3]);
    mwSize numSpikes = mxGetN(prhs[0]);

    // Output data
    plhs[0] = mxCreateDoubleMatrix(1, numSpikes, mxREAL);
    double *output_spikeTimes_samples = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix(1, numSpikes, mxREAL);
    double *output_spikeTemplates = mxGetPr(plhs[1]);

    plhs[2] = mxCreateLogicalMatrix(1, numSpikes);
    mxLogical *output_removeIdx = mxGetLogicals(plhs[2]);

    // Copy data to output
    memcpy(output_spikeTimes_samples, spikeTimes_samples, numSpikes * sizeof(double));
    memcpy(output_spikeTemplates, spikeTemplates, numSpikes * sizeof(double));

    // Initialize removeIdx to false
    for (mwSize i = 0; i < numSpikes; i++) {
        output_removeIdx[i] = false;
    }

    // Debugging prints
    mexPrintf("numSpikes: %d\n", numSpikes);

    // Remove duplicates
    for (mwSize i = 0; i < numSpikes; i++) {
        mexPrintf("i: %d\n", i);
        for (mwSize j = 0; j < numSpikes; j++) {
            mexPrintf("j: %d\n", j);
            if (i != j && spikeTemplates[i] == spikeTemplates[j]) {
                if (fabs(spikeTimes_samples[i] - spikeTimes_samples[j]) <= duplicateSpikeWindow_samples) {
                    if (templateAmplitudes[i] < templateAmplitudes[j]) {
                        output_spikeTimes_samples[i] = NAN;
                        output_removeIdx[i] = true;
                    } else {
                        output_spikeTimes_samples[j] = NAN;
                        output_removeIdx[j] = true;
                    }
                }
            }
        }
    }
}
