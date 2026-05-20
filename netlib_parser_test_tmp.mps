NAME          TINY
OBJSENSE
  MIN
ROWS
 N  COST
 L  R1
 G  R2
 E  R3
COLUMNS
    X1      COST      1      R1        1
    X1      R2        1
    X2      COST      2      R1        1
    X2      R3        1
    X3      COST     -1      R2        1
RHS
    RHS1    R1        5      R2        1
    RHS1    R3        2
BOUNDS
 LO BND     X1        1
 UP BND     X1        4
 FR BND     X2
 MI BND     X3
 UP BND     X3       10
RANGES
 RNG1       R3        3
ENDATA
