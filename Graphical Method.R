#For maximization objective function
library(lpSolve)
objective <- c(50, 40)
constraints <- matrix(c(2, 3, 3, 2), nrow = 2, byrow = TRUE)
rhs <- c(100, 90)
direction <- c("<=", "<=")

solution <- lp("max", objective, constraints, direction, rhs, compute.sens = TRUE)
solution
optimal_values <- solution$solution
max_profit <- solution$objval


library(ggplot2)
constraint1 <- function(x) (100 - 2 * x) / 3
constraint2 <- function(x) (90 - 3 * x) / 2

# Define the range of x values
x_values <- seq(0, 50, length.out = 400)
# Create a data frame for ggplot
df <- data.frame(x = x_values,
                 y1 = pmax(0, constraint1(x_values)),
                 y2 = pmax(0, constraint2(x_values)))

# Plot the constraints and feasible region
ggplot(df, aes(x = x)) +
  geom_line(aes(y = y1), color = "blue", size = 1) +
  geom_line(aes(y = y2), color = "red", size = 1) +
  geom_ribbon(aes(ymin = 0, ymax = pmin(y1, y2)), fill = "pink", alpha = 0.5) +
  geom_point(aes(x = optimal_values[1], y = optimal_values[2]), color = "green", size = 3) +
  labs(title = "Feasible Region and Optimal Solution",
       x = "Units of P1",
       y = "Units of P2") +
  annotate("text", x = optimal_values[1], y = optimal_values[2], label = "Optimal Solution", hjust = -0.2, vjust = -1.5) +
  xlim(0, 50) +
  ylim(0, 50) +
  theme_minimal()



########
########

#For minimization objective function
########
########
library(lpSolve)

# Objective function coefficients
objective <- c(6, 4)

# Constraint matrix
constraints <- matrix(c(2, 1,
                        1, 2),
                      nrow = 2, byrow = TRUE)

# Right-hand side
rhs <- c(8, 7)
direction <- c(">=", ">=")
solution <- lp("min", objective, constraints, direction, rhs, compute.sens = TRUE)

solution
optimal_values <- solution$solution
min_cost <- solution$objval

library(ggplot2)

# Constraint boundary functions
constraint1 <- function(x) (8 - 2*x)
constraint2 <- function(x) (7 - 1*x) / 2

# Generate x values
x_values <- seq(0, 10, length.out = 400)

# Data frame for plotting
df <- data.frame(
  x = x_values,
  y1 = constraint1(x_values),
  y2 = constraint2(x_values)
)

# Keep only non-negative values

# Create separate data frame for optimal solution
opt_df <- data.frame(
  x = optimal_values[1],
  y = optimal_values[2]
)

ggplot(df, aes(x = x)) +
  geom_line(aes(y = y1), color = "blue", size = 1.2) +
  geom_line(aes(y = y2), color = "red", size = 1.2) +
  geom_ribbon(aes(ymin =pmax(y1, y2) , ymax = Inf),
              fill = "lightgreen", alpha = 0.4) +
  geom_point(data = opt_df, aes(x = x, y = y),
             color = "black", size = 4) +
  labs(
    title = "Feasible Region and Optimal Solution (Minimization)",
    x = "Units of Material 1",
    y = "Units of Material 2"
  ) +
  xlim(0, 10) +
  ylim(0, 10) +
  theme_minimal()
