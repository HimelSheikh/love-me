# Example 1.1: X-bar and R Chart

means  <- c(43,49,37,44,45,37,51,46,43,47)
ranges <- c(5,6,5,7,7,4,8,6,4,6)
samples <- 1:10
n <- 5

Xbarbar <- mean(means)
Rbar    <- mean(ranges)

A2 <- 0.577   # standard table value for n=5
D3 <- 0
D4 <- 2.115

# X-bar chart limits
CLx  <- Xbarbar
UCLx <- Xbarbar + A2*Rbar
LCLx <- Xbarbar - A2*Rbar

# R chart limits
CLr  <- Rbar
UCLr <- D4*Rbar
LCLr <- D3*Rbar

plot(samples,
     means,
     type="b",
     pch=19,
     col="purple",
     main="X-bar Chart", xlab="Sample No.", ylab="Sample Mean",
     ylim=c(35,60))

abline(h=CLx,  col="darkgreen", lty=2, lwd=2)
abline(h=UCLx, col="red", lty=2, lwd=2)
abline(h=LCLx, col="red", lty=2, lwd=2)

legend("topright",
       legend = c(
         "Sample Mean",
         "Center Line",
         "UCL / LCL"),
       
       col = c("purple","darkgreen","red"),
       lty = c(1,2,2),
       pch = c(19,NA,NA),
       bty = "n"
)


plot(
  samples,
  ranges,
  type="b",
  pch=19,
  col="orange",
  main="R Chart", xlab="Sample No.", ylab="Sample Range",
  ylim=c(0,20))

abline(h=CLr,  col="darkgreen", lty=2, lwd=2)
abline(h=UCLr, col="red", lty=2, lwd=2)
abline(h=LCLr, col="red", lty=2, lwd=2)

legend("topright",
       legend=c(
         "Sample Range",
         "Center Line",
         "UCL/LCL"),
       col=c("orange","darkgreen","red"),
       lty=c(1,2,2),
       pch=c(19,NA,NA),
       bty="n"
)


# ================================
# Example 1.2:

# 1. Enter the sample data (12 samples of size n = 5)
data <- matrix(c(
  42, 65, 75, 78, 87,
  42, 45, 68, 72, 90,
  19, 24, 80, 81, 81,
  36, 54, 69, 77, 84,
  42, 51, 57, 59, 78,
  51, 74, 75, 78, 132,
  60, 60, 72, 95, 138,
  18, 20, 27, 42, 60,
  15, 30, 39, 62, 84,
  69, 109, 113, 118, 153,
  64, 90, 93, 109, 112,
  61, 78, 94, 109, 136
), 
byrow = TRUE, ncol = 5)

# 2. Calculate sample means and sample ranges
sample.means  <- rowMeans(data)
sample.ranges <- apply(data, 1, function(x) max(x) - min(x))

# 3. Overall grand mean (X-double-bar) and average range (R-bar)
Xbarbar <- mean(sample.means)
Rbar    <- mean(sample.ranges)

# 4. Control chart constants for subgroup size n = 5
A2 <- 0.577
D3 <- 0
D4 <- 2.114

# X-BAR CHART (MEAN CHART)

CL_x  <- Xbarbar
UCL_x <- Xbarbar + A2 * Rbar
LCL_x <- Xbarbar - A2 * Rbar


# Plot X-bar Chart
plot(
  sample.means,
  type = "b",
  pch  = 19,
  col  = "purple",
  ylim = c(30,140),
  main = "X-bar Control Chart",
  xlab = "Sample Number",
  ylab = "Sample Mean"
)

abline(h = CL_x,  col = "darkgreen", lty = 2, lwd = 2)
abline(h = UCL_x, col = "red",       lty = 2, lwd = 2)
abline(h = LCL_x, col = "red",       lty = 2, lwd = 2)

legend(
  "topright",
  legend = c("Sample Mean", "Center Line (CL)", "Control Limits (UCL/LCL)"),
  col    = c("purple", "darkgreen", "red"),
  lty    = c(1, 2, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)

# R CHART (RANGE CHART)

CL_r  <- Rbar
UCL_r <- D4 * Rbar
LCL_r <- D3 * Rbar


# Plot R Chart
plot(
  sample.ranges,
  type = "b",
  pch  = 19,
  col  = "purple",
  ylim = c(0,180),
  main = "R Control Chart",
  xlab = "Sample Number",
  ylab = "Sample Range"
)

abline(h = CL_r,  col = "darkgreen", lty = 2, lwd = 2)
abline(h = UCL_r, col = "red",       lty = 2, lwd = 2)
abline(h = LCL_r, col = "red",       lty = 2, lwd = 2)

legend(
  "topright",
  legend = c("Sample Range", "Center Line (CL)", "Control Limits (UCL/LCL)"),
  col    = c("purple", "darkgreen", "red"),
  lty    = c(1, 2, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# ================================
# Example 1.3: (Out of Syllabus)

# 1. Input the dataset for 20 subgroups (n = 5)

x_bar <- c(
  34.0, 31.8, 30.6, 33.0, 35.0, 32.2, 33.6, 32.0, 33.8, 37.8,
  35.8, 38.4, 34.0, 35.0, 33.8, 31.6, 33.0, 28.2, 31.8, 35.6)

r_values <- c(
  4, 4, 2, 3, 5, 2, 5, 13, 19, 6,
  4, 4, 14, 4, 7, 5, 5, 3, 9, 6)

samples <- 1:20

# Control Chart Constants for subgroup size n = 5
A2 <- 0.577
D3 <- 0
D4 <- 2.114

# INITIAL CONTROL LIMITS (Stage 1)

x_double_bar <- mean(x_bar)
r_bar        <- mean(r_values)

# X-bar Chart Limits
CL_X  <- x_double_bar
UCL_X <- x_double_bar + A2 * r_bar
LCL_X <- x_double_bar - A2 * r_bar

# R-Chart Limits
CL_R  <- r_bar
UCL_R <- D4 * r_bar
LCL_R <- D3 * r_bar

# REVISED CONTROL LIMITS (Stage 2 - Eliminating Out-of-Control Points)

# Standard SPC Procedure: First eliminate out-of-control points from the R-chart,
# recompute limits, and then eliminate any remaining out-of-control points from the X-bar chart.

# Step A: Eliminate R-chart out-of-control points (subgroup 9 where R = 19 > UCL_R)
valid_indices_r <- which(!(r_values > UCL_R | r_values < LCL_R))

x_bar_rev1 <- x_bar[valid_indices_r]
r_rev1     <- r_values[valid_indices_r]

r_bar_rev <- mean(r_rev1)
x_double_bar_rev1 <- mean(x_bar_rev1)

# Step B: Recompute revised limits using updated R-bar
UCL_R_rev <- D4 * r_bar_rev
LCL_R_rev <- D3 * r_bar_rev

UCL_X_rev1 <- x_double_bar_rev1 + A2 * r_bar_rev
LCL_X_rev1 <- x_double_bar_rev1 - A2 * r_bar_rev

# Check if any X-bar values fall outside these new revised X-bar limits
samples_rev1 <- samples[valid_indices_r]
out_X_rev1    <- samples_rev1[x_bar_rev1 > UCL_X_rev1 | x_bar_rev1 < LCL_X_rev1]

# Step C: Exclude out-of-control points on X-bar chart (subgroups 12 and 18)
valid_indices_final <- samples_rev1[!(x_bar_rev1 > UCL_X_rev1 | x_bar_rev1 < LCL_X_rev1)]

x_bar_final <- x_bar[valid_indices_final]
r_final     <- r_values[valid_indices_final]

# Final parameters for future control limits
r_bar_final        <- mean(r_final)
x_double_bar_final <- mean(x_bar_final)

# Final Control Limits for Future Use

CL_X_final  <- x_double_bar_final
UCL_X_final <- x_double_bar_final + A2 * r_bar_final
LCL_X_final <- x_double_bar_final - A2 * r_bar_final

CL_R_final  <- r_bar_final
UCL_R_final <- D4 * r_bar_final
LCL_R_final <- D3 * r_bar_final

# 1. Final X-bar Chart
plot(
  samples, x_bar, type = "b", pch = 19, col = "blue",
  main = "Revised X-bar Control Chart",
  xlab = "Sample Number", ylab = "Sample Mean",
  ylim = c(25,50)
)

abline(h = CL_X_final, col = "darkgreen", lwd = 2)
abline(h = UCL_X_final, col = "red", lty = 2, lwd = 2)
abline(h = LCL_X_final, col = "red", lty = 2, lwd = 2)

legend(
  "topright",
  legend = c("Sample Mean", "Center Line (CL)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 2, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# 2. Final R Chart

plot(
  samples, r_values, type = "b", pch = 19, col = "blue",
  main = "Revised R Control Chart",
  xlab = "Subgroup Number", ylab = "Sample Range",
  ylim = c(0,30)
)

abline(h = CL_R_final, col = "darkgreen", lwd = 2)
abline(h = UCL_R_final, col = "red", lty = 2, lwd = 2)
abline(h = LCL_R_final, col = "red", lty = 2, lwd = 2)

legend(
  "topright",
  legend = c("Sample Range", "Center Line (CL)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 2, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# ================================
# Example 1.4:(Out Of Syllabus)

# 1. Input the dataset for 20 samples (n = 5)

x_bar <- c(
  0.8372, 0.8324, 0.8318, 0.8344, 0.8346, 0.8332, 0.8340, 0.8344, 0.8308, 0.8350,
  0.8380, 0.8322, 0.8356, 0.8322, 0.8404, 0.8372, 0.8282, 0.8346, 0.8360, 0.8374
)

r_values <- c(
  0.010, 0.009, 0.008, 0.004, 0.005, 0.011, 0.009, 0.003, 0.002, 0.006,
  0.006, 0.002, 0.013, 0.005, 0.008, 0.011, 0.006, 0.006, 0.004, 0.006
)

samples <- 1:20

# Control Chart Constants for subgroup size n = 5
A2 <- 0.577
D3 <- 0
D4 <- 2.114

# INITIAL CONTROL LIMITS (Stage 1)

x_double_bar <- mean(x_bar)
r_bar        <- mean(r_values)

# X-bar Chart Limits
CL_X  <- x_double_bar
UCL_X <- x_double_bar + A2 * r_bar
LCL_X <- x_double_bar - A2 * r_bar

# R-Chart Limits
CL_R  <- r_bar
UCL_R <- D4 * r_bar
LCL_R <- D3 * r_bar

# Check for out-of-control points in Stage 1
out_R <- samples[r_values > UCL_R | r_values < LCL_R]
out_X <- samples[x_bar > UCL_X | x_bar < LCL_X]

# REVISED CONTROL LIMITS FOR FUTURE USE (Stage 2)

# Step A: Remove out-of-control points from R-chart (if any)
valid_r_indices <- which(!(r_values > UCL_R | r_values < LCL_R))
r_bar_rev       <- mean(r_values[valid_r_indices])

# Step B: Remove out-of-control points from X-bar chart (e.g., Groups 15 & 17)
valid_x_indices <- which(!(x_bar > UCL_X | x_bar < LCL_X))
x_double_bar_rev <- mean(x_bar[valid_x_indices])

# Final Control Limits for Future Production

CL_X_final  <- x_double_bar_rev
UCL_X_final <- x_double_bar_rev + A2 * r_bar_rev
LCL_X_final <- x_double_bar_rev - A2 * r_bar_rev

CL_R_final  <- r_bar_rev
UCL_R_final <- D4 * r_bar_rev
LCL_R_final <- D3 * r_bar_rev



# PLOTTING REVISED CONTROL CHARTS

# 1. X-bar Chart

plot(
  samples, x_bar, type = "b", pch = 19, col = "blue",
  main = "X-bar Control Chart for Length of Bomb Bases",
  xlab = "Group Number", ylab = "Sample Mean",
  ylim = c(0.828,0.846)
)

abline(h = CL_X_final, col = "darkgreen", lwd = 2)
abline(h = UCL_X_final, col = "red", lty = 2, lwd = 2)
abline(h = LCL_X_final, col = "red", lty = 2, lwd = 2)

legend(
  "topright",
  legend = c("Sample Mean", "Center Line (CL)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 2, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)

# 2. R Chart
plot(
  samples, r_values, type = "b", pch = 19, col = "blue",
  main = "R Control Chart for Length of Bomb Bases",
  xlab = "Group Number", ylab = "Sample Range",
  ylim = c(0,0.02)
)

abline(h = CL_R_final, col = "darkgreen", lwd = 2)
abline(h = UCL_R_final, col = "red", lty = 2, lwd = 2)
abline(h = LCL_R_final, col = "red", lty = 2, lwd = 2)

legend(
  "topright",
  legend = c("Sample Range", "Center Line (CL)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 2, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# ================================
# Example 1.10:

# 1. Enter the sample data (22 lots, lot size n = 2000)
defectives <- c(
  425, 430, 216, 341, 225, 322, 280, 306, 337, 305, 356,
  402, 216, 264, 126, 409, 193, 326, 280, 389, 451, 420
)

n <- 2000
lots <- 1:length(defectives)

# 2. Calculate fraction defective for each lot (p_i)
p_i <- defectives / n

# 3. Calculate overall average fraction defective (p-bar)
p_bar <- sum(defectives) / (length(defectives) * n)

# 4. Calculate Standard Error
SE <- sqrt(p_bar * (1 - p_bar) / n)

# 5. Calculate Control Limits
CL  <- p_bar
UCL <- p_bar + 3 * SE
LCL <- p_bar - 3 * SE

# Floor constraint: proportion cannot be less than zero
if (LCL < 0) LCL <- 0

# 6. Plot the p-Chart
plot(
  lots, p_i,
  type = "b", pch = 19, col = "blue",
  ylim = c(0.05,0.35),
  main = "p-Chart for Rubber Belts Fraction Defective",
  xlab = "Lot Number",
  ylab = "Fraction Defective (p)"
)

# Add Control Lines
abline(h = CL,  col = "darkgreen", lwd = 2)
abline(h = UCL, col = "red",       lty = 2, lwd = 2)
abline(h = LCL, col = "red",       lty = 2, lwd = 2)

# Add Legend
legend(
  "topright",
  legend = c("Fraction Defective (p_i)", "Center Line (p-bar)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 1, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# ================================
# Example 1.11:

# 1. Enter the sample data (30 subgroups, sample size n = 1000)

defectives <- c(
  22, 40, 36, 32, 42, 40, 30, 44, 42, 38,
  70, 80, 44, 22, 32, 42, 20, 46, 28, 36,
  66, 50, 46, 32, 42, 46, 30, 38, 40, 24
)

dates <- 1:30

n <- 1000

# 2. Calculate average proportion defective (p-bar)
p_bar <- sum(defectives) / (length(defectives) * n)

# 3. Calculate Center Line (CL) and Standard Error (SE) for d-chart (np-chart)
CL <- n * p_bar
SE <- sqrt(n * p_bar * (1 - p_bar))

# 4. Calculate 3-sigma Control Limits
UCL <- CL + 3 * SE
LCL <- CL - 3 * SE

# Lower limit floor constraint (count cannot be negative)
if (LCL < 0) LCL <- 0


# 5. Plot the d-Chart (Number of Defectives Chart)
plot(
  dates, defectives,
  type = "b", pch = 19, col = "blue",
  ylim = c(15,100),
  main = "d-Chart for Number of Defectives",
  xlab = "Date (Sept.)",
  ylab = "Number of Defectives (d)"
)

# Add Control Lines
abline(h = CL,  col = "darkgreen", lwd = 2)
abline(h = UCL, col = "red",       lty = 2, lwd = 2)
abline(h = LCL, col = "red",       lty = 2, lwd = 2)

# Legend
legend(
  "topright",
  legend = c("Number of Defectives (d)", "Center Line (CL)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 1, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# ================================
# Example 1.12:

# 1. Enter the sample data (20 samples, sample size n = 10)

defectives <- c(
  0, 1, 0, 3, 9, 2, 0, 7, 0, 1,
  1, 0, 0, 3, 1, 0, 0, 2, 1, 0
)

sample_no <- 1:20 

n <- 10

# Calculate average proportion defective (p-bar)
p_bar <- sum(defectives) / (length(defectives) * n)

# Center Line (CL) and Standard Error (SE) for present d-chart
CL <- n * p_bar
SE <- sqrt(n * p_bar * (1 - p_bar))

# 3-sigma Control Limits
UCL <- CL + 3 * SE
LCL <- CL - 3 * SE

# Lower limit floor constraint (count cannot be negative)
if (LCL < 0) LCL <- 0

# PLOTTING THE d-CHART (PRESENT DATA)

plot(
  sample_no, defectives,
  type = "b", pch = 19, col = "blue",
  ylim = c(-2,15),
  main = "d-Chart (Number of Defectives) - Present Control",
  xlab = "Sample Number",
  ylab = "Number of Defectives (d)"
)

# Add Present Control Lines
abline(h = CL,  col = "darkgreen", lwd = 2)
abline(h = UCL, col = "red",       lty = 2, lwd = 2)
abline(h = LCL, col = "red",       lty = 2, lwd = 2)

# Legend
legend(
  "topright",
  legend = c("Number of Defectives (d)", "Center Line (CL)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 1, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# ================================
# Example 1.13:

# 1. Enter the sample data (10 independent samples of varying sizes)

sample_size <- c(
  2000, 1500, 1400, 1350, 1250,
  1760, 1875, 1955, 3125, 1575)

defectives <- c(
  425, 430, 216, 341, 225,
  322, 280, 306, 337, 305)

samples <- 1:10

# 2. Calculate fraction defective for each sample (p_i)
p_i <- defectives / sample_size

# 3. Calculate overall average fraction defective (p-bar)
p_bar <- sum(defectives) / sum(sample_size)

# 4. Calculate variable Standard Error and 3-sigma Control Limits for each sample
SE  <- sqrt(p_bar * (1 - p_bar) / sample_size)

UCL <- p_bar + 3 * SE
LCL <- p_bar - 3 * SE

# Lower limit floor constraint (proportion cannot be negative)
LCL[LCL < 0] <- 0

# 5. Plot the p-Chart with variable control limits
plot(
  samples, p_i,
  type = "b", pch = 19, col = "blue",
  ylim = c(0.1,0.4),
  main = "p-Chart for Fraction Defective (Variable Sample Sizes)",
  xlab = "Sample Number",
  ylab = "Fraction Defective (p_i)"
)

# Add fixed Center Line (p-bar)
abline(h = p_bar, col = "darkgreen", lwd = 2)

# Add variable Upper and Lower Control Limits (stepped limits)
lines(samples, UCL, col = "red", lty = 2, lwd = 2, type = "s")
lines(samples, LCL, col = "red", lty = 2, lwd = 2, type = "s")

# Legend
legend(
  "topright",
  legend = c("Fraction Defective (p_i)", "Center Line (p-bar)", "Variable UCL/LCL"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 1, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)



####################For U Chart

# 1. Enter the sample data (10 independent samples of varying sizes)

sample_size <- c(
  2000, 1500, 1400, 1350, 1250,
  1760, 1875, 1955, 3125, 1575
)

defects <- c(
  425, 430, 216, 341, 225,
  322, 280, 306, 337, 305
)

samples <- 1:10

# 2. Calculate defects per unit for each sample (u_i)
u_i <- defects / sample_size

# 3. Calculate overall average defects per unit (u-bar)
u_bar <- sum(defects) / sum(sample_size)

# 4. Calculate variable Standard Error and 3-sigma Control Limits for each sample
SE  <- sqrt(u_bar / sample_size)
UCL <- u_bar + 3 * SE
LCL <- u_bar - 3 * SE

# Lower limit floor constraint (rate cannot be negative)
LCL[LCL < 0] <- 0

# 5. Plot the u-Chart with variable control limits
plot(
  samples, u_i,
  type = "b", pch = 19, col = "blue",
  ylim = c(0.10,0.30),
  main = "u-Chart for Defects per Unit (Variable Sample Sizes)",
  xlab = "Sample Number",
  ylab = "Defects per Unit (u_i)"
)

# Add fixed Center Line (u-bar)
abline(h = u_bar, col = "darkgreen", lwd = 2)

# Add variable Upper and Lower Control Limits (stepped lines)
lines(samples, UCL, col = "red", lty = 2, lwd = 2, type = "s")
lines(samples, LCL, col = "red", lty = 2, lwd = 2, type = "s")

# Legend
legend(
  "topright",
  legend = c("Defects per Unit (u_i)", "Center Line (u-bar)", "Variable UCL/LCL"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 1, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# ================================
# Example 1.14:

# 1. Enter the sample data (24 hourly observations across 3 days)

defects <- c(
  2, 4, 7, 3, 1, 4, 8, 9,
  5, 3, 7, 11, 6, 4, 9, 9,
  6, 4, 3, 9, 7, 4, 7, 12)

hours <- 1:24

# 2. Calculate average number of defects per inspection unit (c-bar)
c_bar <- mean(defects)

# 3. Calculate Center Line (CL) and 3-sigma Control Limits for c-chart
CL  <- c_bar
UCL <- c_bar + 3 * sqrt(c_bar)
LCL <- c_bar - 3 * sqrt(c_bar)

# Lower limit floor constraint (count cannot be negative)
if (LCL < 0) LCL <- 0

# 4. Plot the c-Chart
plot(
  hours, defects,
  type = "b", pch = 19, col = "blue",
  ylim = c(-2,20),
  main = "c-Chart for Number of Defects in Seam Welding",
  xlab = "Observation Number (Hour)",
  ylab = "Number of Defects (c)"
)

# Add Control Lines
abline(h = CL,  col = "darkgreen", lwd = 2)
abline(h = UCL, col = "red",       lty = 2, lwd = 2)
abline(h = LCL, col = "red",       lty = 2, lwd = 2)

# Legend
legend(
  "topright",
  legend = c("Number of Defects (c)", "Center Line (c-bar)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 1, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)


# ================================
# Example 1.15:

# 1. Enter the sample data (20 items, fixed sample size = 1 unit each)

defects <- c(
  2, 0, 4, 1, 0, 8, 0, 1, 2, 0,
  6, 0, 2, 1, 0, 3, 2, 1, 0, 2
)

items <- 1:20

# 2. Calculate average number of defects per unit (c-bar)
c_bar <- mean(defects)

# 3. Calculate Center Line (CL) and 3-sigma Control Limits for c-chart
CL  <- c_bar
UCL <- c_bar + 3 * sqrt(c_bar)
LCL <- c_bar - 3 * sqrt(c_bar)

# Lower limit floor constraint (count cannot be negative)
if (LCL < 0) LCL <- 0

# 4. Plot the c-Chart
plot(
  items, defects,
  type = "b", pch = 19, col = "blue",
  ylim = c(-2,10),
  main = "c-Chart for Number of Defects per Item",
  xlab = "Item Number",
  ylab = "Number of Defects (c)"
)

# Add Control Lines
abline(h = CL,  col = "darkgreen", lwd = 2)
abline(h = UCL, col = "red",       lty = 2, lwd = 2)
abline(h = LCL, col = "red",       lty = 2, lwd = 2)

# Legend
legend(
  "topright",
  legend = c("Number of Defects (c)", "Center Line (c-bar)", "Control Limits (UCL/LCL)"),
  col    = c("blue", "darkgreen", "red"),
  lty    = c(1, 1, 2),
  pch    = c(19, NA, NA),
  bty    = "n"
)
