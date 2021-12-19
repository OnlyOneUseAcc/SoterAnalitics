# Soter Analytics 

## Data description
`*.pickle` contains accelerometer data recorded for 15 min.
Every line consists of 4 values: 
* time
* ax
* ay
* az

We know that during this time-series (est. 07:35:00 - 07:50:00 GMT+8) we had two periods
when the person was driving a car:
* `07:35:30 - 07:36:20`
* `07:48:00 - 07:49:20`

## Task description
Task: you need to come up with an algorithm/solution how to classify a time period (let's say
every 15 seconds) what type of activity (driving/not driving) it was.

## Algorithm requirements
The algorithm should work as it has line-by-line input, so if the data will come to you in real
time. (data frequency is 16hz) You can remember previous values but you can't look forward
to the data. Classification decisions need to be made not later than 1 minute (we do them
every 15 seconds).

## Algorithm description 
Algorithm receive sensor data with 16hz frequency and, with condition that it has already received
15 seconds of data, makes prediction on one time series

## Experiment 
Demonstration how algorithm can work with line by line input

start experiment - `python experiment.py`

model for experiment can be found [here](https://drive.google.com/file/d/1-SYOkcW7XcmsHZMVSmP44y5pH5GloPZ3/view?usp=sharing)

It saves results of experiment in `.json` file with 
`
{(start, end) : class}
`
format, where 
* `start` and `end` timestamps describing time series location
* `class` - flag describing is this time series marked as driving or not 


