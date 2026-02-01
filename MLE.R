#1.Bernoulli Distribution
set.seed(129)
x <- rbinom(50, 1, 0.3)

loglikelihood <- function(p){
  sum(dbinom(x, 1, p, log=TRUE))
}

p_value <- seq(0.01, 0.99, length=1000)
logLik <- sapply(p_value, loglikelihood)

plot(p_value, logLik, type='l', lty=2, lwd=4, col='SkyBlue',
     main='Bernoulli: Log-Likelihood vs p',
     xlab=expression(p), ylab='Log-Likelihood')
abline(v=p_value[which.max(logLik)], lty=3, lwd=4, col='Pink')

p_value[which.max(logLik)]  # MLE


#2.Poisson Distribution
set.seed(129)
x <- rpois(50, lambda=3)
x

loglikelihood <- function(lambda){
  sum(dpois(x, lambda, log=TRUE))
}

lambda_value <- seq(0.1, 10, length=1000)
logLik <- sapply(lambda_value, loglikelihood)
logLik

plot(lambda_value, logLik, type='l', lty=2, lwd=3, col='Purple',
     main='Poisson: Log-Likelihood vs Lambda',
     xlab=expression(lambda), ylab='Log-Likelihood')
abline(v=lambda_value[which.max(logLik)], lty=3, lwd=4, col='Orange')

lambda_value[which.max(logLik)]


#3.Geometric Distribution 
set.seed(129)
x <- rgeom(50, prob=0.4) + 1

loglikelihood <- function(p){
  sum(dgeom(x-1, prob=p, log=TRUE))
}

p_value <- seq(0.01, 0.99, length=1000)
logLik <- sapply(p_value, loglikelihood)

plot(p_value, logLik, type='l', lty=2, lwd=3, col='Green',
     main='Geometric: Log-Likelihood vs p',
     xlab=expression(p), ylab='Log-Likelihood')
abline(v=p_value[which.max(logLik)], lty=3, lwd=4, col='Red')

p_value[which.max(logLik)]


#4.Hypergeometric Distribution 
set.seed(129)
x <- rhyper(50, m=20, n=30, k=10)
loglikelihood <- function(K){
  sum(dhyper(x, m=K, n=50-K, k=10, log=TRUE))
}
K_value <- 5:40
logLik <- sapply(K_value, loglikelihood)
plot(K_value, logLik, type='l', lty=2, lwd=3, col='Blue',
     main='Hypergeometric: Log-Likelihood vs K',
     xlab='K', ylab='Log-Likelihood')
abline(v=K_value[which.max(logLik)], lty=3, lwd=4, col='Orange')
K_value[which.max(logLik)]


#5.Negative Binomial Distribution
set.seed(129)
x <- rnbinom(50, size=5, prob=0.4)
x
loglikelihood <- function(p){
  sum(dnbinom(x, size=5, prob=p, log=TRUE))
}

p_value <- seq(0.01,0.99,length=1000)
logLik <- sapply(p_value, loglikelihood)
logLik
plot(p_value, logLik, type='l', lty=2, lwd=3, col='DarkRed',
     main='NegBin: Log-Likelihood vs p',
     xlab=expression(p), ylab='Log-Likelihood')
abline(v=p_value[which.max(logLik)], lty=3, lwd=4, col='Pink')

p_value[which.max(logLik)] 


#6.Exponential Distribution
set.seed(129)
x <- rexp(50, rate=0.5)
x
loglikelihood <- function(lambda){
  sum(dexp(x, rate=lambda, log=TRUE))
}

lambda_value <- seq(0.1,1.5,length=1000)
logLik <- sapply(lambda_value, loglikelihood)
logLik
plot(lambda_value, logLik, type='l', lty=2, lwd=3, col='Green',
     main='Exponential: Log-Likelihood vs λ',
     xlab=expression(lambda), ylab='Log-Likelihood')
abline(v=lambda_value[which.max(logLik)], lty=3, lwd=4, col='Purple')

lambda_value[which.max(logLik)]


#7.Gamma Distribution 
set.seed(129)
x <- rgamma(50, shape=2, rate=1)

loglikelihood <- function(shape){
  sum(dgamma(x, shape=shape, rate=1, log=TRUE))
}

shape_value <- seq(0.5,5,length=1000)
logLik <- sapply(shape_value, loglikelihood)

plot(shape_value, logLik, type='l', lty=2, lwd=3, col='Orange',
     main='Gamma: Log-Likelihood vs Shape',
     xlab='Shape', ylab='Log-Likelihood')
abline(v=shape_value[which.max(logLik)], lty=3, lwd=4, col='Blue')

shape_value[which.max(logLik)]


#8.Beta Distribution
set.seed(129)
x <- rbeta(50, 2, 5)

loglikelihood <- function(alpha){
  sum(dbeta(x, alpha, 5, log=TRUE))
}

alpha_value <- seq(0.5,6,length=1000)
logLik <- sapply(alpha_value, loglikelihood)

plot(alpha_value, logLik, type='l', lty=2, lwd=3, col='Violet',
     main='Beta: Log-Likelihood vs α',
     xlab=expression(alpha), ylab='Log-Likelihood')
abline(v=alpha_value[which.max(logLik)], lty=3, lwd=4, col='Red')

alpha_value[which.max(logLik)]


#9.Normal Distribution
set.seed(129)
x <- rnorm(50, mean=5, sd=2)

loglikelihood <- function(mu){
  sum(dnorm(x, mean=mu, sd=2, log=TRUE))
}

mu_value <- seq(2,8,length=1000)
logLik <- sapply(mu_value, loglikelihood)

plot(mu_value, logLik, type='l', lty=2, lwd=3, col='Cyan',
     main='Normal: Log-Likelihood vs μ',
     xlab=expression(mu), ylab='Log-Likelihood')
abline(v=mu_value[which.max(logLik)], lty=3, lwd=4, col='DarkBlue')

mu_value[which.max(logLik)]


#10.Weibull Distribution
set.seed(129)
x <- rweibull(50, shape=2, scale=1)

loglikelihood <- function(shape){
  sum(dweibull(x, shape=shape, scale=1, log=TRUE))
}

shape_value <- seq(0.5,5,length=1000)
logLik <- sapply(shape_value, loglikelihood)

plot(shape_value, logLik, type='l', lty=2, lwd=3, col='DarkOrange',
     main='Weibull: Log-Likelihood vs Shape',
     xlab='Shape', ylab='Log-Likelihood')
abline(v=shape_value[which.max(logLik)], lty=3, lwd=4, col='Green')

shape_value[which.max(logLik)]


#11.Log-normal Distribution 
set.seed(129)
x <- rlnorm(50, meanlog=0, sdlog=1)

loglikelihood <- function(mu){
  sum(dlnorm(x, meanlog=mu, sdlog=1, log=TRUE))
}

mu_value <- seq(-1,1,length=1000)
logLik <- sapply(mu_value, loglikelihood)

plot(mu_value, logLik, type='l', lty=2, lwd=3, col='Pink',
     main='Log-Normal: Log-Likelihood vs meanlog',
     xlab='meanlog', ylab='Log-Likelihood')
abline(v=mu_value[which.max(logLik)], lty=3, lwd=4, col='Purple')

mu_value[which.max(logLik)]


#12.Truncated Normal Distribution
install.packages("truncnorm")l
ibrary(truncnorm)
set.seed(129)
x <- rtruncnorm(50, a=0, b=Inf, mean=5, sd=1)

loglikelihood <- function(mu){
  sum(dtruncnorm(x, a=0, b=Inf, mean=mu, sd=1, log=TRUE))
}

mu_value <- seq(3,7,length=1000)
logLik <- sapply(mu_value, loglikelihood)

plot(mu_value, logLik, type='l', lty=2, lwd=3, col='Blue',
     main='Truncated Normal: Log-Likelihood vs μ',
     xlab=expression(mu), ylab='Log-Likelihood')
abline(v=mu_value[which.max(logLik)], lty=3, lwd=4, col='Red')

mu_value[which.max(logLik)]
