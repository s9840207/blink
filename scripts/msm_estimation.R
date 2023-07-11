library('msm')
library('reticulate')
np <- import('numpy')
dir <- '/home/tzu-yu/Downloads/LYT/Cdc13-WT 5 nM/'
observations <- read.csv(paste(dir, 'observations.csv', sep=''))
Q <- rbind(c(0, 0.25, 0.25),
           c(0.25, 0, 0.25),
           c(0.25, 0.25, 0))
Q.crude <- crudeinits.msm(state ~ time, molecule, data=observations, qmatrix=Q)
print(Q.crude)
result <- msm(state ~ time, subject=molecule, data=observations, qmatrix=Q.crude,
              control=list(trace=1, REPORT=1))
print(result)
q <- qmatrix.msm(result)
#a <- q['estimate',0]
prevalence <- prevalence.msm(result, times=seq(0, max(observations[['time']]), 0.05))
np$savez(paste(dir, 'msm_result.npz', sep=''),
         estimate=q$estimate,
         SE=q$SE,
         lower_bound=q$L,
         upper_bound=q$U,
         expected_prevalence=prevalence$'Expected percentages',
         observed_prevalence=prevalence$'Observed percentages')
q
q$L
#q
#qmatrix.msm(result, ci='boot')
