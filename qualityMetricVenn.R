## venn diagram -hacky way by saving a csv in matlab and loading it here 
## -don't have time to implement this in matlab. 
# why does matlab suck so much
# r is so great

# library(venneuler)
# myVenn <- venneuler(c(A=10, B=11, C=13, D=12, 'A&B'=1, 'A&C'=13, 'B&C'=4, 'A&B&C'=1))
# myVenn$labels <- c("A\n22","B\n7","C\n5","D\n58")
# plot(myVenn)
# text(0.59,0.52,"1")
# text(0.535,0.51,"3")
# text(0.60,0.57,"2")
# text(0.64,0.48,"4") 

# library(eulerr)
# 
# vd <- euler(c(A = 0.3, B = 0.3, C = 1.1, D=1.2,
#               "A&B" = 0.1, "A&C" = 0.2, "B&C" = 0.1,
#               "A&B&C" = 0.1))
# plot(vd, legend = TRUE, quantities = TRUE)


library(eulerr)
mets <- read.csv(file = 'C:/Users/Julie/Dropbox/Paper/metricsTableAll.csv')
plot(euler(mets), legend = list(labels = c("# Peaks/Troughs", "# Spikes",
                                  "Waveform Amplitude", "Refractory period violations", 
                                  "Waveform Duration", "Axonal", 
                                  "% Spikes Missing")), quantities = TRUE)