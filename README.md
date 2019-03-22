# cmpt318-cybersecurity

# Anomaly Detection (Hidden Markov Models)

## Problem Description
- Detecting unknown attacks using Anomaly-based IDS such as point anomalies, and contextual anomalies in individual household electric power consumption based on the data set being provided and being able to find patterns on data instances which don’t conform with the normal behavior.
- The electric grid uses Supervisory Control and Data Acquisition (SCADA).
- If anything happens to the SCADA system, it can cause catastrophic effects to the grid.

## Data set Description
1. **Global active power** is the real power consumption by appliances not included in the submeters. It is household global minute-averaged active power and its unit is kilowatt. 

2. **Global reactive power** is the  household global minute-averaged reactive power with the unit of a kilowatt. It refers that the power moves forward or backward that is not used or leaked. Compared to active power, it is the imaginary power consumption and active power is the real power consumption.

3. **Voltage** is the minute average voltage that is used in the household. The higher the voltage means there is a high flow of electrical current.

4. **Sub Metering 1** is the power that is consumed by appliances in the kitchen, such as microwave, oven dishwasher.

5. **Sub Metering 2**

6. **Sub Metering 3**

## Methods
a. Point anomaly
  - Min-Max:
    - Determined the Min and Max value in our train set and applied the knowledge to our test set.
    - If the values in the test data set are smaller than the min or greater than the max then we classify these values as point anomalies.
  - Moving Average:
    - Selected a window size of 15 minutes, and slided the window by 1 minute each time.
    - Made sure to NOT count the next day’s observation in the previous day’s calculation.
    
b. Context Anomalies
   - Observation window (Using Ntimes in the depmix model in R)
   - Cross-Validation Split the Train Data (70%) Train and (30%) Validation.
   - Computed the Log Likelihood for the Train and Validation
   - Recorded the BIC of each state.
   - Normalized the Log Likelihood Value.
   - Compared the Log Likelihood Values for the Train and Validation.

## Challenges
- Initial parameters of the Hidden Markov Model
- Data verification
- Finding anomalies

# Lessons learned
- Working with HMM models, learnt how to train data set more efficiently for better robustness.
- Designing an HMM model requires you to recognize the patterns in the data to determine what window to use
