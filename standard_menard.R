rm(list=ls())
library(mice)

library(dplyr)
set.seed(123456789)


#==== Goodman ====
#bg = b/se(b)

b <- out2$coefficients[-1,1]
se_b <- out2$coefficients[-1,2]
bg <- data.frame(b/se_b)

#==== Agresti ====
#ba <- b(Sx)

b <- out2$coefficients[-1,1]
sx <- apply(mylogit$model[2:17], 2, sd)
ba <- data.frame(b*sx)

#==== SAS ====
#bsas <- [(b)(sx)]/[pi/SQRT(3)]

b <- out2$coefficients[-1,1]
sx <- apply(mylogit$model[2:17], 2, sd)
bsas <- data.frame((b*sx)/(pi/sqrt(3)))

#==== Menard ====
library(car)
#bm <- [b(sx)R]/Slogit(y^)

b <- out2$coefficients[-1,1]
sx <- apply(mylogit$model[2:17], 2, sd)
yob <- data.frame(mylogit$y)
ypred <- data.frame(mylogit$fitted.values)
syobsqr <- apply(yob, 2, var)
sypredsqr <- apply(ypred, 2, var)
sy_logit_predsqr <- apply(logit(ypred), 2, sd)
R <- sqrt(sypredsqr/syobsqr)
bm <- data.frame((b*sx*R)/sy_logit_predsqr)

#==== Long ====
#bl <- [b(sx)]/SQRT(S²logit(y^) + pi²/3)
b <- out2$coefficients[-1,1]
sx <- apply(mylogit$model[2:17], 2, sd)
ypred <- data.frame(mylogit$fitted.values)
sy_logit_predvar <- apply(logit(ypred), 2, var)
bl <- data.frame((b*sx)/sqrt(sy_logit_predvar + (pi^2)/3))
