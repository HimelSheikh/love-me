set.seed(123)
x <- rnorm(100, mean = 50, sd = 10)
x
hist(x,
     main = "Random Numbers from Normal Distribution",
     xlab = "Values")
x1<- runif(20, min = 0, max = 1)
x1
x2 <- rbinom(20, size = 10, prob = 0.5)
x2
x3 <- rpois(20, lambda = 4)
x3
x4 <- rexp(20, rate = 0.5)
x4
x5 <- rt(20, df = 10)
x5
x6 <- rchisq(20, df = 5)
x6
F <- rf(20, df1 = 5, df2 = 20)
F
B <- rbeta(100, shape1 = 2, shape2 = 5)
B
#100 = number of random observations
#shape1 =alpha
#shape2 =beta
#Plot the generated Beta data
hist(B,
     main = "Random Numbers from Beta Distribution",
     xlab = "Values",
     breaks = 15)

W <- rweibull(100, shape = 2, scale = 5)
W
set.seed(123)

c <- rcauchy(
  100,
  location = 10,
  scale = 2
)
c
hist(
  c,
  breaks = 30,
  main = "Random Numbers from Cauchy Distribution",
  xlab = "Values"
)
#compare cauchy and normal
set.seed(123)
normal <- rnorm(1000, mean = 0, sd = 1)
cauchy <- rcauchy(1000, location = 0, scale = 1)
dev.off()
par(mfrow=c(2,2))
hist(
  cauchy,
  breaks = 50,
  main = "Cauchy Distribution",
  xlab = "Value"
)
hist(
  normal,
  breaks = 50,
  main = "Normal Distribution",
  xlab = "Value"
)
##
x6 <- seq(-4, 4, length = 1000)
y <- dnorm(x6, mean = 0, sd = 1)

plot(x6, y,
     type = "l",
     lwd = 2,
     main = "Normal Distribution",
     xlab = "x",
     ylab = "Density",col="red")

x7 <- seq(0, 1, length = 1000)
y <- dbeta(x7, shape1 = 2, shape2 = 5)
plot(x7, y,
     type = "l",
     lwd = 2,
     main = "Beta Distribution",
     xlab = "x",
     ylab = "Density",col="red")
###########diff parameter
x8 <- seq(0, 1, length = 1000)
y1 <- dbeta(x8, shape1 = 1, shape2 = 5)
y2 <- dbeta(x8, shape1 = 2, shape2 = 5)
y3 <- dbeta(x8, shape1 = 5, shape2 = 5)

plot(x8, y1,
     type = "l",
     lwd = 2,
     xlab = "x",
     ylab = "Density",col="red",
     main = "Beta Distribution: Changing Shape1")

lines(x8, y2, lwd = 2,col="blue")
lines(x8, y3, lwd = 2, col="green")

legend("topright",
       legend = c("α = 1", "α = 2", "α = 5"),
       lty = 1,
       lwd = 2,col="red")
