#CT2
#Q1
# Data
before <- c(85, 78, 90, 82, 88, 80, 76, 85, 87, 90)
after <- c(83, 76, 87, 80, 84, 78, 75, 82, 86, 89)

# Paired t-test
t.test(before, after,
       paired = TRUE,
       alternative = "greater")

#Q2
mu0 <- 80
mu1 <- 85
sigma <- 12
n <- 49
alpha <- 0.01

SE <- sigma/sqrt(n)

z_alpha <- qnorm(1-alpha)

xbar_critical <- mu0 + z_alpha*SE

SE
z_alpha
xbar_critical

power <- 1 - pnorm(xbar_critical,
                   mean = mu1,
                   sd = SE)

power

x <- seq(70, 95, length = 1000)

y0 <- dnorm(x, mean = mu0, sd = SE)
y1 <- dnorm(x, mean = mu1, sd = SE)

plot(x, y0,
     type = "l",
     col = "blue",
     xlab = expression(bar(X)),
     ylab = "Density",
     main = "Sampling Distributions Under H0 and H1")

lines(x, y1,
      col = "red")

abline(v = xbar_critical,
       lwd = 2,
       lty = 2)

# Shade power region
xp <- seq(xbar_critical, max(x), length = 500)
yp <- dnorm(xp, mean = mu1, sd = SE)

polygon(c(xbar_critical, xp, max(x)),
        c(0, yp, 0),
        col = "lightgreen")

legend("topright",
       legend = c("H0: mu=80",
                  "H1: mu=85",
                  "Critical Value",
                  "Power Region"),
       col = c("blue","red","black","lightgreen"),
       lwd = c(2,2,2,10),
       lty = c(1,1,2,1))

#CT3
#Q1
benzene <- c(0.21, 1.44, 2.54, 2.97, 0.00,
             3.91, 2.24, 2.41, 4.50, 0.15,
             0.30, 0.36, 4.50, 5.03, 0.00,
             2.89, 4.71, 0.85, 2.60, 1.26)

t.test(benzene,
       mu = 1,
       alternative = "greater",
       conf.level = 0.95)
#Q2
mu0 <- 1200
xbar <- 1150
s <- 100
n <- 30

t <- (xbar - mu0)/(s/sqrt(n))
p_value <- 2*(1 - pt(abs(t), df=n-1))

t
p_value