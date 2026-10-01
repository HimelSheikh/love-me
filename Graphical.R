#####
library(lpSolve)
library(ggplot2)

# LPP
f <- c(50,40)

A <- matrix(c(2,3,
              3,2), 2, 2, byrow=TRUE)

b <- c(100,90)

sol <- lp("max", f, A, c("<=","<="), b)

sol$solution
sol$objval


# Graph
x <- seq(0,50,0.1)

y1 <- (100-2*x)/3
y2 <- (90-3*x)/2

df <- data.frame(x,y1,y2)

ggplot(df,aes(x=x)) +
  geom_ribbon(aes(ymin=0,ymax=pmin(y1,y2)),
              fill="pink") +
  geom_line(aes(y=y1),color="blue") +
  geom_line(aes(y=y2),color="red") +
  geom_point(aes(x=sol$solution[1],
                 y=sol$solution[2]),
             color="green",size=3) +
  labs(title="Feasible Region and Optimal Solution",
       x="Units of X1",
       y="Units of X2") +
  theme_bw() +
  xlim(0,70) +
  ylim(0,50)




########
library(lpSolve)
library(ggplot2)

# LPP
f <- c(6,4)

A <- matrix(c(2,1,
              1,2), 2, 2, byrow=TRUE)

b <- c(8,7)

sol <- lp("min", f, A, c(">=",">="), b)

sol$solution
sol$objval


# Graph
x <- seq(0,10,0.1)

y1 <- 8-2*x
y2 <- (7-x)/2

df <- data.frame(x,y1,y2)

ggplot(df,aes(x)) +
  geom_ribbon(aes(ymin=pmax(y1,y2),ymax=10),
              fill="pink") +
  geom_line(aes(y=y1),color="blue") +
  geom_line(aes(y=y2),color="red") +
  geom_point(aes(x=sol$solution[1],
                 y=sol$solution[2]),
             color="green",size=3) +
  labs(title="Feasible Region and Optimal Solution",
       x="Units of X1",
       y="Units of X2") +
  theme_bw() +
  xlim(0,10) +
  ylim(0,10)













