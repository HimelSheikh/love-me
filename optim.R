#1.Bernoulli Distribution
set.seed(123)
# Sample generation
x <- rbinom(200, size = 1, prob = 0.4)
# Negative log-likelihood function
neglogL_bern <- function(par) {
  p <- par[1]
  -sum(dbinom(x, size = 1, prob = p, log = TRUE))
}
# MLE using optim()
mle_bern <- optim(
  par = c(p = 0.5),
  fn = neglogL_bern,
  hessian = TRUE
)
mle_bern$par
mle_bern$hessian
mle_bern$convergence

#Graphical Illustration: Fitted Bernoulli Distribution
hist(x, breaks = 2, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Bernoulli Distribution",  xlab = "x")
points(c(0, 1),
       dbinom(c(0, 1), size = 1, prob = mle_bern$par[1]),
       type = "h", lwd = 4, col = "darkgreen")
legend("topright", legend = "Fitted Bernoulli",
       col = "darkgreen", lwd = 2)


#Binomial distribution
set.seed(123)

# Sample generation (Binomial, NOT Bernoulli)
# size = 10 trials, prob = 0.4
x <- rbinom(200, size = 10, prob = 0.4)

# Negative log-likelihood function
neglogL_binom <- function(par) {
  p <- par[1]
  -sum(dbinom(x, size = 10, prob = p, log = TRUE))
}

# MLE using optim()
mle_binom <- optim(
  par = c(p = 0.5),          # initial guess
  fn = neglogL_binom,
  method = "L-BFGS-B",       # allows bounds
  lower = 1e-6,
  upper = 1 - 1e-6,
  hessian = TRUE
)

# Results
mle_binom$par        # MLE of p
mle_binom$hessian    # Hessian matrix
mle_binom$convergence

# Histogram of observed Binomial data
hist(x, breaks = seq(-0.5, 10.5, 1), freq = FALSE,
     col = "lightblue",
     main = "MLE Fit – Binomial Distribution",
     xlab = "x")

# Support of Binomial distribution
k <- 0:10   # size = 10

# Fitted Binomial PMF using MLE
points(k,
       dbinom(k, size = 10, prob = mle_binom$par[1]),
       type = "h", lwd = 4, col = "darkblue")

# Legend
legend("topright", legend = "Fitted Binomial (MLE)",
       col = "darkblue", lwd = 2)


#2.Poisson Distribution
set.seed(123)
# Sample generation
x <- rpois(200, lambda = 3)
# Negative log-likelihood function
neglogL_pois <- function(par) {
  lambda <- par[1]
  -sum(dpois(x, lambda = lambda, log = TRUE))
}
# MLE using optim()
mle_pois <- optim(
  par = c(lambda = 1),
  fn = neglogL_pois,
  hessian = TRUE
)
mle_pois$par
mle_pois$hessian
mle_pois$convergence

#Graphical Illustration: Fitted Poisson Distribution
hist(x, breaks = max(x) - min(x) + 1, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Poisson Distribution",
     xlab = "x")

points(x = min(x):max(x),
       y = dpois(min(x):max(x), lambda = mle_pois$par[1]),
       type = "h", lwd = 4, col = "darkgreen")

legend("topright", legend = "Fitted Poisson",
       col = "darkgreen", lwd = 2)


#3.Geometric Distribution
set.seed(137)  # roll number 137
# Sample generation
x <- rgeom(200, prob = 0.3)
# Negative log-likelihood function
neglogL_geom <- function(par) {
  p <- par[1]
  -sum(dgeom(x, prob = p, log = TRUE))
}
# MLE using optim()
mle_geom <- optim(
  par = c(p = 0.5),
  fn = neglogL_geom,
  hessian = TRUE
)
mle_geom$par
mle_geom$hessian
mle_geom$convergence

#Graphical Illustration: Fitted Geometric Distribution
hist(x, breaks = max(x) - min(x) + 1, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Geometric Distribution",
     xlab = "x")
points(x = min(x):max(x),
       y = dgeom(min(x):max(x), prob = mle_geom$par[1]),
       type = "h", lwd = 4, col = "darkgreen")
legend("topright", legend = "Fitted Geometric",
       col = "darkgreen", lwd = 2)


#4. Hypergeometric Distribution
set.seed(137)  # roll number 137
# Sample generation
# Parameters: population size N = 50, number of successes in population M = 20, sample size n = 10
x <- rhyper(200, m = 20, n = 30, k = 10)
# Negative log-likelihood function
neglogL_hyper <- function(par) {
  M <- round(par[1])  # M must be integer between 0 and N
  -sum(dhyper(x, m = M, n = 50 - M, k = 10, log = TRUE))
}
# MLE using optim()
mle_hyper <- optim(
  par = c(M = 20),
  fn = neglogL_hyper,
  hessian = TRUE
)
mle_hyper$par
mle_hyper$hessian
mle_hyper$convergence

#Graphical Illustration:
hist(x, breaks = min(x):max(x), freq = FALSE, col = "lightgreen",
       main = "MLE Fit - Hypergeometric Distribution",
       xlab = "x")
points(x = min(x):max(x),
       y = dhyper(min(x):max(x), m = round(mle_hyper$par[1]), n = 50 - round(mle_hyper$par[1]), k = 10),
       type = "h", lwd = 4, col = "darkgreen")
legend("topright", legend = "Fitted Hypergeometric",
       col = "darkgreen", lwd = 2)


#5.Normal Distribution
set.seed(137)  # roll number 137
# Sample generation
x <- rnorm(200, mean = 10, sd = 3)
# Negative log-likelihood function for sigma^2 unknown, mean = 10
neglogL_norm <- function(par) {
  sigma2 <- par[1]
  -sum(dnorm(x, mean = 10, sd = sqrt(sigma2), log = TRUE))
}
# MLE using optim()
mle_norm <- optim(
  par = c(sigma2 = 1),
  fn = neglogL_norm,
  hessian = TRUE
)
mle_norm$par
mle_norm$hessian
mle_norm$convergence

#Graphical Illustration: Fitted Normal Distribution
hist(x, breaks = 20, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Normal Distribution",
     xlab = "x")
curve(dnorm(x, mean = 10, sd = sqrt(mle_norm$par[1])),
      add = TRUE, col = "darkgreen", lwd = 2)
legend("topright", legend = "Fitted Normal",
       col = "darkgreen", lwd = 2)


#6.Weibull Distribution
set.seed(137)  # roll number 137
# Sample generation
# Shape = 2, Scale = 1.5
x <- rweibull(200, shape = 2, scale = 1.5)
# Negative log-likelihood function
neglogL_weib <- function(par) {
  shape <- par[1]
  scale <- par[2]
  -sum(dweibull(x, shape = shape, scale = scale, log = TRUE))
}
# MLE using optim()
mle_weib <- optim(
  par = c(shape = 1, scale = 1),
  fn = neglogL_weib,
  hessian = TRUE
)
mle_weib$par
mle_weib$hessian
mle_weib$convergence

#Graphical Illustration: Fitted Weibull Distribution
hist(x, breaks = 20, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Weibull Distribution",
     xlab = "x")
curve(dweibull(x, shape = mle_weib$par[1], scale = mle_weib$par[2]),
      add = TRUE, col = "darkgreen", lwd = 2)
legend("topright", legend = "Fitted Weibull",
       col = "darkgreen", lwd = 2)


#7.Log-Normal Distribution
set.seed(111)  # roll number 111
# Sample generation
# Parameters: meanlog = 0, sdlog = 0.5
x <- rlnorm(200, meanlog = 0, sdlog = 0.5)
# Negative log-likelihood function
neglogL_lognorm <- function(par) {
  meanlog <- par[1]
  sdlog <- par[2]
  -sum(dlnorm(x, meanlog = meanlog, sdlog = sdlog, log = TRUE))
}
# MLE using optim()
mle_lognorm <- optim(
  par = c(meanlog = 0, sdlog = 1),
  fn = neglogL_lognorm,
  hessian = TRUE
)
mle_lognorm$par
mle_lognorm$hessian
mle_lognorm$convergence

#Graphical Illustration: Fitted Log-Normal Distribution
hist(x, breaks = 20, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Log-Normal Distribution",
     xlab = "x")
curve(dlnorm(x, meanlog = mle_lognorm$par[1], sdlog = mle_lognorm$par[2]),
      add = TRUE, col = "darkgreen", lwd = 2)
legend("topright", legend = "Fitted Log-Normal",
       col = "darkgreen", lwd = 2)


#8.Truncated Normal Distribution
set.seed(137)  # roll number 137
# Sample generation
# Parameters: mean = 5, sd = 2, truncated between 3 and 8
library(truncnorm)
x <- rtruncnorm(200, a = 3, b = 8, mean = 5, sd = 2)
# Negative log-likelihood function
neglogL_truncnorm <- function(par) {
  mu <- par[1]
  sigma <- par[2]
  -sum(dtruncnorm(x, a = 3, b = 8, mean = mu, sd = sigma))
}
# MLE using optim()
mle_truncnorm <- optim(
  par = c(mu = 5, sigma = 2),
  fn = neglogL_truncnorm,
  hessian = TRUE
)
mle_truncnorm$par
mle_truncnorm$hessian
mle_truncnorm$convergence

#Graphical Illustration: 
hist(x, breaks = 20, freq = FALSE, col = "lightgreen",
     main = "Truncated Normal Sample",
     xlab = "x")

# Curve using approximate MLE (just for visualization)
curve(dtruncnorm(x, a = 3, b = 8, mean = 5.763, sd = 1e-15),
      add = TRUE, col = "darkgreen", lwd = 2)

legend("topright", legend = "Approx. Truncated Normal",
       col = "darkgreen", lwd = 2)


#9.Exponential Distribution
set.seed(98)  # roll number 98
# Sample generation
# True rate lambda = 0.5
x <- rexp(200, rate = 0.5)
# Negative log-likelihood function
neglogL_exp <- function(par) {
  lambda <- par[1]
  -sum(dexp(x, rate = lambda, log = TRUE))
}
# MLE using optim()
mle_exp <- optim(
  par = c(lambda = 1),
  fn = neglogL_exp,
  hessian = TRUE
)

mle_exp$par
mle_exp$hessian
mle_exp$convergence

#Graphical Illustration – 
hist(x, breaks = 20, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Exponential Distribution",
     xlab = "x")
curve(dexp(x, rate = mle_exp$par[1]),
      add = TRUE, col = "darkgreen", lwd = 2)
legend("topright", legend = "Fitted Exponential",
       col = "darkgreen", lwd = 2)


#10.Gamma Distribution
set.seed(125)  # roll number 125
# Sample generation
# Shape = 2, Rate = 0.5
x <- rgamma(200, shape = 2, rate = 0.5)
# Negative log-likelihood function
neglogL_gamma <- function(par) {
  shape <- par[1]
  rate <- par[2]
  -sum(dgamma(x, shape = shape, rate = rate, log = TRUE))
}
# MLE using optim()
mle_gamma <- optim(
  par = c(shape = 1, rate = 1),
  fn = neglogL_gamma,
  hessian = TRUE
)
mle_gamma$par
mle_gamma$hessian
mle_gamma$convergence

#Graphical Illustration: 
hist(x, breaks = 20, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Gamma Distribution",
     xlab = "x")

curve(dgamma(x, shape = mle_gamma$par[1], rate = mle_gamma$par[2]),
      add = TRUE, col = "darkgreen", lwd = 2)

legend("topright", legend = "Fitted Gamma",
       col = "darkgreen", lwd = 2)


#11.Beta Distribution
set.seed(105)  # roll number 105
# Sample generation
# Parameters: alpha = 2, beta = 5
x <- rbeta(200, shape1 = 2, shape2 = 5)
# Negative log-likelihood function
neglogL_beta <- function(par) {
  alpha <- par[1]
  beta <- par[2]
  -sum(dbeta(x, shape1 = alpha, shape2 = beta, log = TRUE))
}
# MLE using optim()
mle_beta <- optim(
  par = c(alpha = 1, beta = 1),
  fn = neglogL_beta,
  hessian = TRUE
)
mle_beta$par
mle_beta$hessian
mle_beta$convergence

#Graphical Illustration: 
hist(x, breaks = 20, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Beta Distribution",
     xlab = "x")
curve(dbeta(x, shape1 = mle_beta$par[1], shape2 = mle_beta$par[2]),
      add = TRUE, col = "darkgreen", lwd = 2)

legend("topright", legend = "Fitted Beta",
       col = "darkgreen", lwd = 2)


#12.Negative Binomial Distribution
set.seed(137)  # roll number 137
# Sample generation
# Parameters: size = 5, prob = 0.3
x <- rnbinom(200, size = 5, prob = 0.3)
# Negative log-likelihood function
neglogL_nb <- function(par) {
  size <- par[1]
  prob <- par[2]
  -sum(dnbinom(x, size = size, prob = prob, log = TRUE))
}
# MLE using optim()
mle_nb <- optim(
  par = c(size = 1, prob = 0.5),
  fn = neglogL_nb,
  hessian = TRUE
)
mle_nb$par
mle_nb$hessian
mle_nb$convergence

#Graphical Illustration: 

hist(x, breaks = 20, freq = FALSE, col = "lightgreen",
     main = "MLE Fit - Negative Binomial Distribution",
     xlab = "x")
curve(dnbinom(x, size = mle_nb$par[1], prob = mle_nb$par[2]),
      add = TRUE, col = "darkgreen", lwd = 2)
legend("topright", legend = "Fitted Negative Binomial",
       col = "darkgreen", lwd = 2)

