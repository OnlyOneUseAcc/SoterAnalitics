# Soter Analytics 

## Data description
Please find *.pickle file attached. You can use the same-named library to read this file into
the python model. This file contains accelerometer data recorded for 15 min.
Every line consists of 4 values: time, ax, ay, az
We know that during this time-series (est. 07:35:00 - 07:50:00 GMT+8) we had two periods
when the person was driving a car:
07:35:30 - 07:36:20
07:48:00 - 07:49:20

## Task description
Task: you need to come up with an algorithm/solution how to classify a time period (let's say
every 15 seconds) what type of activity (driving/not driving) it was.

## Algorithm requirements
The algorithm should work as it has line-by-line input, so if the data will come to you in real
time. (data frequency is 16hz) You can remember previous values but you can't look forward
to the data. Classification decisions need to be made not later than 1 minute (we do them
every 15 seconds).