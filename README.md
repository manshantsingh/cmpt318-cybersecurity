# cmpt318-cybersecurity

Anamoly Detection (Hidden Markov Models)

Problem Description
-The electric grid uses Supervisory Control and Data Acquisition (SCADA).
-If anything happens to the SCADA system, it can cause catastrophic effects to the grid.

-Data set Decription
Global active power: is the real power consumption by appliances not included in the sub meters. It is household global minute-averaged active power and its unit is kilowatt. 

Global reactive power :is the  household global minute-averaged reactive power with the unit of kilowatt. It refers that the power moves forward or backward that is not used or leaked. Compared to active power, it is the imaginary power consumption and the active power is the real power consumption.

Voltage
Sub Metering 1: refers that the power are consumed by appliances in the kitchen, such as microwave, oven dishwasher.

-Missing labels

Methods
-Point anomaly
	min max
	Moving Average:
		- Selected a window size of 15 minutes, and slided the window by 1 minute each time
		- Made sure to NOT count the next day’s observation in the previous day’s calculation
-Context Anomalies
	- Observation window (Using Ntimes in the depmix model in R)
	- Cross Validation Split the Train Data (70%) Train and (30%) Validation.
	- Computed the Log Likelihood for the Train and Validation


Challanges
-Intial parameters of the hidden markov model
-Data verification
-Finding anomalies

Lessons learned
-Working with HMM models, learnt how to train data set more efficiently for better robustness.
-Designing an HMM model requires you to recognize the patterns in the data to determine what window to use
