#with packages

install.packages("BSDA")
library(BSDA)
n=36
sample_mean=105
popu_sd=15
x=rnorm(n,mean=sample_mean,sd=popu_sd)
z.test(x,mu=100,sigma.x = 15)
x=c(4,5,6,7,8,9)
t.test(x,mu=100)
before=c(7,4,5,6,5)
after=c(4,5,7,9,3)
t.test(before,after,paired = TRUE)
grp1=c(3,4,5,6,7)
grp2=c(2,5,4,7,8)
t.test(grp1,grp2,var.equal = TRUE)
grp1=c(3,4,5,6,7)
grp2=c(2,5,4,7,8)
t.test(grp1,grp2,var.equal = FALSE)
