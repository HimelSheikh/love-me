N <- 10000
n <- 200
c <- 2

p <- seq(0, 0.05, by = 0.001)

lambda <- n * p

Pa <- ppois(c, lambda)

AOQ <- p * Pa * ((N - n) / N)
AOQL <- max(AOQ)
p_AOQL <- p[which.max(AOQ)]

ATI <- n * Pa + N * (1 - Pa)

plot(p * 100, Pa,
     type = "l",
     lwd = 2,
     xlab = "Incoming Fraction Defective (%)",
     ylab = "Probability of Acceptance",
     main = "OC Curve - Poisson Sampling")

plot(p * 100, AOQ * 100,
     type = "l",
     lwd = 2,
     xlab = "Incoming Fraction Defective (%)",
     ylab = "AOQ (%)",
     main = "AOQ Curve - Poisson Sampling")

abline(h = AOQL * 100, lty = 2)

plot(p * 100, ATI,
     type = "l",
     lwd = 2,
     xlab = "Incoming Fraction Defective (%)",
     ylab = "Average Total Inspection",
     main = "ATI Curve - Poisson Sampling")
