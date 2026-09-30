# Example 1
library(lpSolve)
library(ggplot2)

# Objective function coefficients (Profit: P1 = 40, P2 = 30)
objective <- c(40, 30)

# Constraint matrix (Machine hours, Labour hours)[cite: 1]
constraints <- matrix(c(2, 1, 
                        1, 2), nrow = 2, byrow = TRUE)

# Right-hand side (Available resources)[cite: 1]
rhs <- c(100, 80)
direction <- c("<=", "<=")

# Solve the Linear Programming problem
solution <- lp("max", objective, constraints, direction, rhs, compute.sens = TRUE)
solution
optimal_values <- solution$solution
max_profit <- solution$objval

# Constraint boundary functions
constraint1 <- function(x) (100 - 2 * x)
constraint2 <- function(x) (80 - x) / 2

# Define the range of x values
x_values <- seq(0, 60, length.out = 400)

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
  labs(title = "Feasible Region and Optimal Solution (Product-Mix)",
       x = "Units of P1",
       y = "Units of P2") +
  annotate("text", x = optimal_values[1], y = optimal_values[2], label = "Optimal Solution", hjust = -0.2, vjust = -1.5) +
  xlim(0, 60) +
  ylim(0, 60) +
  theme_minimal()





# Example 2

library(lpSolve)
library(ggplot2)

# Objective function coefficients (Cost: Food A = 4, Food B = 3)[cite: 1]
objective <- c(4, 3)

# Constraint matrix (Protein, Vitamin requirements)[cite: 1]
constraints <- matrix(c(3, 2,
                        1, 3), nrow = 2, byrow = TRUE)

# Right-hand side (Minimum requirements)[cite: 1]
rhs <- c(12, 9)
direction <- c(">=", ">=")

# Solve the Linear Programming problem
solution <- lp("min", objective, constraints, direction, rhs, compute.sens = TRUE)
solution
optimal_values <- solution$solution
min_cost <- solution$objval

# Constraint boundary functions
constraint1 <- function(x) (12 - 3 * x) / 2
constraint2 <- function(x) (9 - x) / 3

# Generate x values
x_values <- seq(0, 10, length.out = 400)

# Data frame for plotting
df <- data.frame(
  x = x_values,
  y1 = constraint1(x_values),
  y2 = constraint2(x_values)
)

# Plot the constraints and feasible region
ggplot(df, aes(x = x)) +
  geom_line(aes(y = y1), color = "blue", size = 1.2) +
  geom_line(aes(y = y2), color = "red", size = 1.2) +
  geom_ribbon(aes(ymin = pmax(0, pmax(y1, y2)), ymax = Inf),
              fill = "lightgreen", alpha = 0.4) +
  geom_point(aes(x = optimal_values[1], y = optimal_values[2]),
             color = "black", size = 4) +
  annotate("text", x = optimal_values[1], y = optimal_values[2], label = "Optimal Solution", hjust = -0.2, vjust = -1.5) +
  labs(
    title = "Feasible Region and Optimal Solution (Diet Problem)",
    x = "Units of Food A",
    y = "Units of Food B"
  ) +
  xlim(0, 10) +
  ylim(0, 10) +
  theme_minimal()

