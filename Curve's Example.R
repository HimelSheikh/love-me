# Example 1.16
# Lot size = 2200
# Sample size = 225
# Acceptance number = 14

N <- 2200
n <- 225
c <- 14

# Incoming fraction defective
p <- seq(0, 0.20, by = 0.001)

# Poisson parameter
lambda <- n * p

# Probability of acceptance
Pa <- ppois(c, lambda)

# 1. OC Curve

plot(p * 100, Pa,
     type = "l",
     lwd = 2,
     xlab = "Incoming Fraction Defective (%)",
     ylab = "Probability of Acceptance",
     main = "OC Curve - Poisson Sampling")

aog<-p*Pa
l<-max(aog)
# 2. AOQ Curve

AOQ <- p * Pa * ((N - n) / N)

# AOQL
AOQL <- max(AOQ)

# Incoming fraction defective at AOQL
p_AOQL <- p[which.max(AOQ)]

plot(p * 100, AOQ * 100,
     type = "l",
     lwd = 2,
     xlab = "Incoming Fraction Defective (%)",
     ylab = "AOQ (%)",
     main = "AOQ Curve - Poisson Sampling")

# Show AOQL
abline(h = AOQL * 100, lty = 2)

# Show location of AOQL
abline(v = p_AOQL * 100, lty = 2)


# 3. ATI Curve

ATI <- n * Pa + N * (1 - Pa)

plot(p * 100, ATI,
     type = "l",
     lwd = 2,
     ylim=c(0,2600),
     xlab = "Incoming Fraction Defective (%)",
     ylab = "Average Total Inspection",
     main = "ATI Curve - Poisson Sampling")



# ============================================

# Example 1.17
# N = 5000, n = 100
# c = 1, 2, 3

N <- 5000
n <- 100

# Range of incoming fraction defective
p <- seq(0, 0.20, by = 0.001)

# Poisson parameter
lambda <- n * p


# (a) OC CURVES

# Probability of acceptance for c = 1, 2, 3
Pa1 <- ppois(1, lambda)
Pa2 <- ppois(2, lambda)
Pa3 <- ppois(3, lambda)

# Plot OC curves
plot(p * 100, Pa1,
     type = "l",
     lwd = 2,
     xlab = "Incoming Fraction Defective (%)",
     ylab = "Probability of Acceptance",
     main = "OC Curves - Single Sampling Plan",
     ylim = c(0, 1))

lines(p * 100, Pa2,
      lwd = 2,
      lty = 2)

lines(p * 100, Pa3,
      lwd = 2,
      lty = 3)

legend("topright",
       legend = c("c = 1", "c = 2", "c = 3"),
       lty = c(1, 2, 3),
       lwd = 2)

# LOT TOLERANCE FRACTION DEFECTIVE
# Assuming Pc = 0.10

Pc <- 0.10

# Find p where Pa is approximately 0.10
LTPD_c1 <- p[which.min(abs(Pa1 - Pc))]
LTPD_c2 <- p[which.min(abs(Pa2 - Pc))]
LTPD_c3 <- p[which.min(abs(Pa3 - Pc))]

# (b) AOQ CURVES

AOQ1 <- p * Pa1 * ((N - n) / N)
AOQ2 <- p * Pa2 * ((N - n) / N)
AOQ3 <- p * Pa3 * ((N - n) / N)

# AOQL
AOQL1 <- max(AOQ1)
AOQL2 <- max(AOQ2)
AOQL3 <- max(AOQ3)

# Plot AOQ curves
plot(p * 100, AOQ1 * 100,
     type = "l",
     lwd = 2,
     ylim=c(0,2.2),
     xlab = "Incoming Fraction Defective (%)",
     ylab = "AOQ (%)",
     main = "AOQ Curves - Single Sampling Plan")

lines(p * 100, AOQ2 * 100,
      lwd = 2,
      lty = 2)

lines(p * 100, AOQ3 * 100,
      lwd = 2,
      lty = 3)

abline(h = AOQL1 * 100, lty = 2)
abline(h = AOQL2 * 100, lty = 2)
abline(h = AOQL3 * 100, lty = 2)

legend("topright",
       legend = c("c = 1", "c = 2", "c = 3"),
       lty = c(1, 2, 3),
       lwd = 2)

