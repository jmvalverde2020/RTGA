# RTGA
Real-Time Gait Analysis Algorithm - Python Library

## Description of the library
This library allows the automatic real time calculation of the following gait parameters during treadmill walking:
- Cadence (steps/min)
- Stride time (s)
- Step time* (s)
- Stance GC% (%)
- Swing GC% (%)
- Loading Response GC%* (%)

\* Parameters are only available with two-leg configurations.

Parameters may not be accurate in non-straight walks.

## Workflow
The RTGA main workflow is composed of three stages:

### Calibration - only needed if the orientation of the gyroscope is unkown
At least 2-seconds data of straight walking must be recorded for the calibration process.

Using the stored raw data, each orientation is tested to find the combination with the most and more prominent peaks.

### Window Build
Real-time stride detection is based on a Finit State Machine (FSM) which finds prominent peak candidates corresponding to the Terminal Swing (TSw) gait phase.

To avoid false positives and event loss, the stride is considered valid if there are no more candidates until Mid Stance (MSt).

The window will be completed when the setted number of strides are detected. Once the window is completed, the gait parameters can be obtained.

### Parameter calculation
The automatic calculation of parameters process the window containing the desired number of strides to find the initial and terminal contact events and calculate the time intervals between said events of the ipsilateral and contralateral legs to obtain temporal gait parameters*.

\* Some gait parameters are only available for 2 legs configurations, as the calculation of said metrics require time intervals between events of both legs.


### Code example (include + for 2 legs configuration)
```
leg1 = rtga.LegInfo("name", Fs)
+ leg2 = rtga.LegInfo("name", Fs)

# -- Calibration process --
calibrated = False
+ leg1_calibrated = False
+ leg2_calibrated = False
while not calibrated:
  leg1.saveRaw([x, y, z], time)
+ leg2.saveRaw([x, y, z], time)

  if len(leg1.buffer) > REFERENCE_TIME:
    leg1_calibrated = leg1.fixOrientation()
  
+  if len(leg2.buffer) > REFERENCE_TIME:
+    leg2_calibrated = leg2.fixOrientation()

+  calibrated = leg1_calibrated and leg2_calibrated
# -- End of calibration process --

leg1.setGoal(3)
+ leg2.setGoal(3)
while run:
  
  # -- Window Build -- #
  leg1.buildWindow([x, y, z], time)
+  leg2.buildWindow([x, y, z], time)

+  rtga.checkFirst(leg1, leg2)

  if leg1.goalReached() +and leg2.goalReached()+:
    # -- End of window build -- #

    # -- Parameter calculation -- #
    params1, +params2+ = rtga.getParameters(leg1 +,leg2+)
    leg1.clear()
+    leg2.clear()
    # -- End of parameter calculation -- #

```
