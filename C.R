mu0=70
mu1=75
alpha=0.05
sigma=10
n=36
SE=sigma/sqrt(n)
Z_alpha=qnorm(1-alpha)
critical_mean=mu0+Z_alpha*SE
x=seq(60,90,length=1000)

h0=dnorm(x,mu0,SE)
h1=dnorm(x,mu1,SE)
plot(x,h0,type="l",main="plot",xlab="sample mean",ylab="density",col="blue")
lines(x,h1,col="red")
abline(v=critical_mean,col="red",lty=3,lwd=2)
x_p=seq(critical_mean,90,length=1000)
z_p=dnorm(x_p,mu1,SE)
polygon(c(critical_mean,x_p),c(0,z_p),col="pink")
legend("topright",
       legend = c("H0", "H1", "Power"),
       col = c("blue", "red", "pink"),
       lwd = 2,
       cex = 0.8,
       bty = "n")