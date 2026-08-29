# X-bar (Mean) Control Chart using R-bar method

# 1. Enter the sample data

data = matrix(c(
  27,23,36,14,
  28,30,30,17,
  21,22,41,44,
  42,35,28,40,
  51,23,21,34,
  33,34,39,30,
  30,22,18,12,
  35,48,20,47,
  20,34,15,42,
  22,50,45,41,
  34,32,48,22,
  44,24,32,33,
  30,28,28,25,
  30,38,38,52,
  36,32,39,45,
  44,48,33,34,
  39,44,29,40,
  38,44,32,27,
  39,34,32,28,
  38,34,36,30
),
byrow = T, ncol = 4)

data


# 2. Calculate sample means

sample.means = rowMeans(data)

sample.means

# 3. Calculate sample ranges

sample.ranges = apply(data, 1, function(x)
  max(x) - min(x))

sample.ranges

# 4. Calculate X-double-bar
 
Xbarbar = mean(sample.means)

Xbarbar

# 5. Calculate R-bar

Rbar = mean(sample.ranges)

Rbar

# 6. Control chart constant

A2 = 0.729


# 7. Calculate Center Line

CL = Xbarbar

# 8. Calculate Upper Control Limit

UCL = Xbarbar + A2 * Rbar

# 9. Calculate Lower Control Limit

LCL = Xbarbar - A2 * Rbar

# 10. Display control limits

CL
UCL
LCL

# 11. Plot the X-bar Control Chart

plot(
  sample.means,
  type = "b",
  pch = 19,
  col = "purple",
  main = "X-bar Chart",
  xlab = "Sample No.",
  ylab = "Sample Mean",
  ylim = c(0,50)
)


# Center Line

abline(h = CL,col = "darkgreen",lty = 2)

# Upper Control Limit

abline(h = UCL,col = "red",lty = 2)

# Lower Control Limit

abline(h = LCL,col = "red",lty = 2)

# 12. Add Legend

legend("bottomright",
       legend = c(
         "Sample Mean",
         "Center Line",
         "UCL / LCL"),
       
       col = c("purple","darkgreen","red"),
       lty = c(1,2,2),
       pch = c(19,NA,NA),
       bty = "n"
)


# ============================================================

# R (Range) CONTROL CHART

# 1. Enter the sample data

data = matrix(c(
  42,65,75,78,87,
  42,45,68,72,90,
  19,24,80,81,81,
  36,54,69,77,84,
  42,51,57,59,78,
  51,74,75,78,132,
  60,60,72,95,138,
  18,20,27,42,60,
  15,30,39,62,84,
  69,109,113,118,153,
  64,90,93,109,112,
  61,78,94,109,136
), byrow=TRUE, ncol=5)

data

# 2. Calculate sample ranges

sample.ranges = apply(data, 1, function(x)
  max(x) - min(x))

sample.ranges

# 3. Calculate average range

Rbar = mean(sample.ranges)

Rbar

# 4. Control chart constants

D3 = 0
D4 = 2.11

# 5. Calculate Center Line

CL = Rbar

# 6. Calculate Upper Control Limit

UCL = D4 * Rbar

# 7. Calculate Lower Control Limit

LCL = D3 * Rbar

# 8. Display control limits

Rbar
UCL
LCL

# 9. Draw the R Control Chart

plot(
  sample.ranges,
  type="b",
  pch=19,
  col="purple",
  ylim=c(0,200),
  main="R Chart",
  xlab="Sample No.",
  ylab="Sample Range"
)

# Center Line

abline(h=CL,col="darkgreen",lty=2)

# Upper Control Limit-

abline(h=UCL,col="red",lty=2)

# Lower Control Limit

abline(h=LCL,col="red",lty=2)

# 10. Add Legend

legend("topright",
       legend=c(
         "Sample Range",
         "Center Line",
         "UCL/LCL"),
       col=c("purple","darkgreen","red"),
       lty=c(1,2,2),
       pch=c(19,NA,NA),
       bty="n"
)


# ============================================================

# p-CHART FOR SPARK PLUG INSPECTION

# Step 1: Enter the data

defectives <- c(
  5,3,15,10,3,16,4,10,5,8,
  5,4,6,8,7,4,6,8,6,10
)

# Sample size of each lot
n <- 100

# Number of lots
lots <- 1:length(defectives)

# Step 2: Calculate proportion defective for each lot

p_i <- defectives / n

p_i

# Step 3: Calculate overall average proportion defective

p_bar <- sum(defectives) / (length(defectives) * n)

p_bar

# Step 4: Calculate Standard Error

SE <- sqrt(p_bar * (1 - p_bar) / n)

SE

# Step 5: Calculate Control Limits

CL<- p_bar

UCL <- p_bar + 3 * SE

LCL <- p_bar - 3 * SE

# A proportion cannot be less than zero
if (LCL < 0) LCL <- 0

# Step 6: Display the values

CL
UCL
LCL

# Step 7: Plot the p-Chart

plot(
  lots,
  p_i,
  type="b",
  pch=19,
  col="blue",
  ylim=c(-0.05,0.3),
  main="p-Chart for Spark Plug Inspection",
  xlab="Lot Number",
  ylab="Fraction Defective"
)

# Center Line


abline(h=CL,col="darkgreen",lwd=2)

# Upper Control Limit

abline(h=UCL,col="red",lty=2,lwd=2)

# Lower Control Limit

abline(h=LCL,col="red",lty=2,lwd=2)

# Step 8: Add Legend

legend("topright",
       legend=c("p_i (each lot)","Center Line","UCL/LCL"),
       col=c("blue","darkgreen","red"),
       lty=c(1,1,2),
       pch=c(19,NA,NA),
       bty="n"
)


# ============================================================

# d-CHART FOR DEFECTIVES

# Step 1: Enter the number of defectives

defectives <- c(
  22,40,36,32,42,40,30,44,42,38,
  70,80,44,22,32,42,20,46,28,36,
  66,50,46,32,42,46,30,38,40,24
)

# Sample size
n <- 1000

# Sample numbers
samples <- 1:length(defectives)

# Step 2: Calculate p-bar

p_bar <- sum(defectives) / (length(defectives) * n)

p_bar

# Step 3: Calculate Center Line

CL <- n * p_bar

# Step 4: Calculate Standard Error

SE <- sqrt(n * p_bar * (1 - p_bar))

SE

# Step 5: Calculate Upper Control Limit

UCL <- CL + 3 * SE

# Step 6: Calculate Lower Control Limit

LCL <- CL - 3 * SE

if (LCL < 0) LCL <- 0

# Step 7: Display the results

p_bar
CL
UCL
LCL

# Step 8: Plot the d-Chart

plot(
  samples,
  defectives,
  type="b",
  pch=19,
  col="blue",
  ylim=c(0,90),
  main="d-Chart for Number of Defectives",
  xlab="Sample Number",
  ylab="Number of Defectives"
)

# Center Line

abline(h=CL,col="darkgreen",lwd=2)

# Upper Control Limit

abline(h=UCL,col="red",lty=2,lwd=2)

# Lower Control Limit

abline(h=LCL,col="red",lty=2,lwd=2)

# Step 8: Add Legend

legend("topright",
       legend=c("Number of Defectives","Center Line","UCL/LCL"),
       col=c("blue","darkgreen","red"),
       lty=c(1,1,2),
       pch=c(19,NA,NA),
       bty="n"
)


# ============================================================

# c-CHART FOR NUMBER OF DEFECTS

# Step 1: Enter the number of defects

defects <- c(
  2,4,7,3,1,4,8,9,
  5,3,7,11,6,4,9,9,
  6,4,3,9,7,4,7,12
)

# Sample numbers
samples <- 1:length(defects)

# Step 2: Calculate average number of defects

c_bar <- mean(defects)

c_bar



# Step 3: Calculate Center Line

CL <- c_bar

# Step 4: Calculate Upper Control Limit

UCL <- c_bar + 3 * sqrt(c_bar)

# Step 5: Calculate Lower Control Limit

LCL <- c_bar - 3 * sqrt(c_bar)

if(LCL < 0) LCL <- 0

# Step 6: Display the results

c_bar
CL
UCL
LCL

# Step 7: Plot the c-Chart

plot(
  samples,
  defects,
  type="b",
  pch=19,
  col="blue",
  ylim=c(0,25),
  main="c-Chart for Number of Defects",
  xlab="Sample Number",
  ylab="Number of Defects"
)

# Center Line

abline(h=CL,col="darkgreen",lwd=2)

# Upper Control Limit

abline(h=UCL,col="red",lty=2,lwd=2)

# Lower Control Limit

abline(h=LCL,col="red",lty=2,lwd=2)

# Step 8: Add Legend

legend("topright",
       legend=c("Number of Defects","Center Line","UCL/LCL"),
       col=c("blue","darkgreen","red"),
       lty=c(1,1,2),
       pch=c(19,NA,NA),
       bty="n"
)


# ============================================================

# u-CHART FOR NUMBER OF DEFECTS PER UNIT

# Step 1: Enter the data

# Number of units inspected in each sample

sample.size <- c(
  100,120,80,150,100,
  90,130,110,100,140
)

# Number of defects found in each sample

defects <- c(
  4,6,3,9,5,
  4,7,6,5,8
)

# Sample numbers

samples <- 1:length(defects)

# Step 2: Calculate defects per unit

u_i <- defects / sample.size

u_i

# Step 3: Calculate average defects per unit

u_bar <- mean(u_i)

u_bar

# Step 4: Calculate Center Line

CL <- u_bar

# Step 5: Calculate Standard Error

SE <- sqrt(u_bar / sample.size)

SE

# Step 6: Calculate Upper Control Limit

UCL <- CL + 3 * SE

# Step 7: Calculate Lower Control Limit

LCL <- CL - 3 * SE

LCL[LCL < 0] <- 0

# Step 8: Display the results

u_i
u_bar
UCL
LCL

# Step 9: Plot the u-Chart

plot(
  samples,
  u_i,
  type="b",
  pch=19,
  col="blue",
  ylim=c(0,0.18),
  main="u-Chart for Number of Defects per Unit",
  xlab="Sample Number",
  ylab="Defects per Unit"
)

# Step 10: Add Center Line

abline(h=u_bar,col="darkgreen",lwd=2)

# Step 11: Add Variable Control Limits

lines(samples,UCL,col="red",lty=2,lwd=2)

lines(samples,LCL,col="red",lty=2,lwd=2)

# Step 12: Add Legend

legend("topright",
       legend=c("u_i (Defects per Unit)","Center Line","UCL/LCL"),
       col=c("blue","darkgreen","red"),
       lty=c(1,1,2),
       pch=c(19,NA,NA),
       bty="n"
)


