from scipy import signal # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import time

OFFSET = 0.5

# ------ Leg class ------ #
class LegInfo():
  def __init__(self, name, sampling_frequency: int, orientation: list = [], reference: int = 5, peak_thp: float = 0.2, valley_thp: float = 0.1, time_thp: float = 0.7):
    """
    Class constructor

    Parameters
    --------------
    name : str 
      Name of the sensorized leg (left/right)
    sampling_frequency : int
      Sampling frequency of the gyroscope
    orientation : list of int, optional
      List with signs for axis
    reference : int, optional
      Number of reference cycles used for adaptive thresholds
    peak_thp : float, optional
      Percentage of peak magnitude for adaptive threshold
    valley_thp : float, optional
      Percentage of valley magnitude for adaptive threshold
    time_thp : float, optional
      Percentage of time for adaptive threshold
    """

    self.reference = reference

    
    self.p_th = peak_thp
    self.v_th = valley_thp
    self.t_th = time_thp

    self.fs = sampling_frequency
    self.name = name
    self.run = False
    self.first = True
    
    self.initVars_(orientation)
    self.initArrays_()

  def initVars_(self, orientation: list):
    """
    Initialize variables
    """

    # -- Threshold variables -- #
    self.TTh = 2
    self.ITh = 0.5
    self.time_th = self.t_th

    # -- Time variables -- #
    self.offset = 0
    self.last_tsw = 0
    self.last_time = 0
    self.stride_time = 0
    self.start = getTime()

    # -- Magnitude variables -- #
    self.last = 0
    self.orientation = orientation

    # -- Counter variables -- #
    self.cycle = 0
    self.state = 0
    self.goal = 0
    
    # -- Connection variables -- #
    self.connected = False
    self.id = 0
    self.device = 0

  def initArrays_(self):
    """
    Initialize arrays
    """

    self.window = []
    self.buffer = []
    self.t_window = []
    self.t_buffer = []
    self.Tsw = []
    self.Tc = []
    self.Ic = []
    self.refP = []
    self.refV = []
    self.refT = []

  def clear(self):
    """
    Clear arrays, flush buffer and save time of last IC
    """
    if len(self.Ic) > 0:
      self.last_time = self.t_window[self.Ic[-1]]

    self.window.clear()
    self.t_window.clear()
    self.Tsw.clear()
    self.Tc.clear()
    self.Ic.clear()

    self.flushBuffer()

    self.cycle = 0
    self.start = getTime()

  def setGoal(self, goal:int):
    """
    Set desired number of cycles for feedback

    Parameters
    ----------
    goal : int
      Number of strides included in the feedback window
    """
    self.goal = goal
  
  def goalReached(self) -> bool:
    """
    Check if the goal has been reached
    """
    if self.cycle >= self.goal:
      return True
    
    return False

  def updateThresholds(self):
    """
    Update adaptive thresholds based of last reference cycles
    """
    if len(self.refP) < self.reference or len(self.refV) < self.reference:
      return

    self.refT.append(self.stride_time)
    while len(self.refP) > self.reference:
      self.refP.pop(0)

    while len(self.refV) > self.reference:
      self.refV.pop(0)
    
    while len(self.refT) > self.reference:
      self.refT.pop(0)

    self.TTh = abs(np.mean(self.refP) * self.p_th)
    self.ITh = abs(np.mean(self.refV) * self.v_th)
    self.time_th = np.mean(self.refT) * self.t_th

  def checkOrder(self, peaks:list, valleys:list) -> tuple[list, list]:
    """
    Check events order and eliminate least probable

    Parameters
    ----------
    peaks : list
      List of positions of terminal swing candidates
    valleys : list
      List of positions of terminal contact candidates
    
    Returns
    -------
    Tc : list
      List of definitive terminal contact positions
    TSw : list
      List of definitive terminal swing positions
    """

    tc = []
    tsw = []

    for peak in peaks:
      if len(tsw) > 0 and (peak - tsw[-1]) < self.time_th*self.fs:
        if self.window[peak] > self.window[tsw[-1]]:
          if len(tsw) > 0:
            tsw.pop()
            if len(tc) > 0:
              tc.pop()
          else:
            continue
      elif (self.t_window[peak] - self.last_tsw) < self.time_th:
        continue

      tsw.append(int(peak))
      dist = peak - valleys

      if all(dist < 0):
        continue

      idx = np.argmin(dist[dist > self.time_th*self.fs*0.1])

      tc.append(int(valleys[idx]))

    self.last_tsw = self.t_window[tsw[-1]]
    return tc, tsw
  
  def findEvents(self):
    """
    Run event-detection algorithm in the current data window
    """
    data = np.array(self.window)

    peaks, _ = signal.find_peaks(data, height=self.TTh, distance=self.time_th*self.fs)
    valley, _ = signal.find_peaks(-data, height=self.ITh)

    if len(valley) < 1:
      self.ITh = 0.5
      return

    self.Tc, self.Tsw = self.checkOrder(peaks, valley)
    for peak in self.Tsw:
      for j in range(len(data[peak:])):
        if data[peak+j] <= 0:
          idx = peak + j - 1
          self.Ic.append(int(idx))
          break
          
    self.refP.extend(data[self.Tsw])
    self.refV.extend(data[self.Tc])

    self.updateThresholds()
  
  def saveRaw(self, gyro:float):
    """
    Save raw gyroscope data for calibration
    """
    self.buffer.append(gyro)

  def flushBuffer(self):
    """
    Flush buffer into data and time window
    """
    self.window.extend(self.buffer)
    self.buffer.clear()

    self.t_window.extend(self.t_buffer)
    self.t_buffer.clear()

  def countStep(self):
    """
    Count the detected step
    """
    self.state = 0

    if not self.run:
      self.run = True
      self.start = getTime()
      self.last = 0
      return True
    
    self.cycle += 1
    now = getTime()
    self.stride_time = now - self.start
    self.start = now
    self.last = 0
  
  def savePeak(self):
    """
    Save current peak
    """
    self.offset = getTime()

    if not self.run:
      self.last_time = getTime()

  def buildWindow(self, gyro:list, t_data:float) -> float:
    """
    Stride-detection algorithm for window filling

    Parameters
    ----------
    gyro : list of float
      List of angular velocity in each axis (x, y, z)
    t_data : float
      Time of the reading
    
    Returns
    -------
    float
      Resulting sum of angular velocities in the movement octant
    """
    data = sum(np.array(gyro)*self.orientation)

    if self.goalReached():
      self.buffer.append(data)
      self.t_buffer.append(t_data)

    if self.run and not self.goalReached():
      self.window.append(data)
      self.t_window.append(t_data)

    if self.state == 0:
      if data > self.TTh and data > self.last:
        self.last = data

      if self.last != 0 and (self.last - data) > 0.6:
        self.state = 1
  
    if self.state == 1:
      if data > self.last:
        self.state = 0

      if data < 0:
        self.savePeak()
        self.state = 2

    if self.state == 2:
      if data > self.last:
        self.state = 0
        return data

      if (getTime() - self.offset) > self.time_th*OFFSET:
        self.countStep()
      return data
    return data

  def fixOrientation(self) -> bool:
    """
    Find best axis sign based on the raw stored data
    """
    if len(self.orientation) > 0:
      return True

    if len(self.buffer) < 2:
      return False

    w_means = []
    means = []
    count = 0
    combinations = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]])

    rot_data = np.array(self.buffer).T
 
    estimated_steps = len(rot_data) / (self.fs*2)
    combinations = np.unique(combinations, axis=0)

    # Check best axis orientation
    for rot in combinations:
      data = rot @ rot_data
      peaks, _ = signal.find_peaks(data, distance=self.fs)

      if len(peaks) == 0:
        peak_mean = 0
      
      else:
        peak_mean = np.mean(abs(data[peaks]))

      weighted_mean = peak_mean * (estimated_steps / (estimated_steps + abs(len(peaks) - estimated_steps)))
      w_means.append(weighted_mean)
      means.append(peak_mean)
      count += 1

    best = np.argmax(w_means)

    self.orientation = combinations[best]
    self.TTh = max(means) * 0.75
    self.mean_time = (len(peaks) / 40) * 0.8 
    self.buffer.clear()

    return True
  
  def getEventTimes(self) -> tuple[list, list]:
    """
    Get time of each inital and terminal contact

    Returns
    -------
    ic_times : list of floats
      Times of initial contact
    tc_times : list of floats
      Times of terminal contact
    """

    ic_times = []
    tc_times = []
    for i in range(len(self.Ic)):
      ic_times.append(self.t_window[self.Ic[i]] if len(self.Ic) > i else -1)
      tc_times.append(self.t_window[self.Tc[i]] if len(self.Tc) > i else -1)

    return ic_times, tc_times

  def getLastIc(self) -> float:
    """
    Return time of last initial contact
    """
    return self.t_window[self.Ic[-1]]
  
# ------ End Leg class ------ #


def getLoadingResponse(ipsi:LegInfo, contra:LegInfo, stride:list) -> list:
  """
  Calculate loading response based on ipsilateral leg

  Parameters
  ----------
  ipsi : LegInfo
    Leg from which obtain loading response gait cycle percentage
  contra : LegInfo
    Contralateral leg
  stride : list of float
    List of considered stride times
  
  Returns
  -------
  list of float
    List of gait cycle percentage at loading response in every stride
  """
  loading_response = []

  if len(ipsi.Ic) < 1 or len(contra.Tc) < 1 or len(stride) < 1:
    print(f"Not enough events for LR")
    return loading_response

  rg = len(ipsi.Ic)
  if contra.first:
    ic = ipsi.last_time
    tc = contra.t_window[contra.Tc[0]]

    loading_response.append((tc - ic) / stride[0] * 100)

    rg = len(contra.Tc) - 1

  if len(stride) < rg or len(contra.Tc) < rg - 1 or len(contra.Ic) < rg:
    return loading_response
  
  for i in range(rg):
    ic = ipsi.t_window[ipsi.Ic[i]]
    st = i
    if ipsi.first:
      tc = contra.t_window[contra.Tc[i]]
    else:
      tc = contra.t_window[contra.Tc[i+1]]
      st = i+1

    loading_response.append((tc - ic) / stride[st] * 100)

  return loading_response

def getSwingStance(leg:LegInfo, stride:list) -> tuple[list, list]:
  """
  Calculate swing and stance percentages

  Parameters
  ----------
  leg : LegInfo
    Leg from which obtain swing and stance gait cycle percentage
  stride : list of float
    List of considered stride times
  
  Returns
  -------
  swing : list of float
    List of gait cycle percentage at swing in every stride
  stance : list of float
    List of gait cycle percentage at stance in every stride
  """

  if len(leg.Ic) < 1 or len(leg.Tc) < 2:
    print(f"Not enough events for {leg.name} leg swing/stance calculation")
    return [], []

  # Stance
  stance = []
  shortest = min(len(leg.Ic), len(leg.Tc)-1)

  if len(stride) < shortest:
    return np.array(swing), np.array(stance)

  stance.append((leg.t_window[leg.Tc[0]] - leg.last_time)/stride[0] * 100)
  for i in range(shortest):
    tc = leg.Tc[i+1]
    ic = leg.Ic[i]

    stance.append((leg.t_window[tc] - leg.t_window[ic])/stride[i+1] * 100)

  # Swing
  swing = []
  shortest = min(len(leg.Ic), len(leg.Tc))

  if len(stride) < shortest:
    return np.array(swing), np.array(stance)
  
  for i in range(shortest):
    tc = leg.Tc[i]
    ic = leg.Ic[i]
    swing.append((leg.t_window[ic] - leg.t_window[tc])/stride[i] * 100)

  return np.array(swing), np.array(stance)

def getStepTime(leg1:LegInfo, leg2:LegInfo) -> tuple[list, list]:
  """
  Calculate step time between both legs

  Parameters
  ----------
  leg1, leg2 : LegInfo

  Returns
  -------
  time1 : list of float
    List of step times from leg1
  time2 : list of float
    List of step times from leg2
  """

  time1 = []
  time2 = []

  if len(leg1.Ic) < 2:
    print(f"Not enough {leg1.name} steps to calculate step time")
    return np.array(time1), np.array(time2)
  elif len(leg2.Ic) < 2:
    print(f"Not enough {leg2.name} steps to calculate step time")
    return np.array(time1), np.array(time2)

  if leg1.first:
    time1.append(leg2.last_time - leg1.last_time)
    time2.append(leg1.t_window[leg1.Ic[0]] - leg2.last_time)

    if time1[0] < 0:
      print(f"{leg1.name} | Error in step 1 {leg2.last_time} - {leg1.last_time}")
    if time2[0] < 0:
      print(f"{leg1.name} | Error in step 2 {leg1.t_window[leg1.Ic[0]]} - {leg2.last_time}")

  else:
    time1.append(leg2.t_window[leg2.Ic[0]] - leg1.last_time)
    time2.append(leg1.last_time - leg2.last_time)

    if time1[0] < 0:
      print(f"{leg2.name} | Error in step 1 {leg2.t_window[leg2.Ic[0]]} - {leg1.last_time}")
    if time2[0] < 0:
      print(f"{leg2.name} | Error in step 2 {leg1.last_time} - {leg2.last_time}")

  shortest = min(len(leg1.Ic)-1, len(leg2.Ic)-1)
  for i in range(shortest):
    ic_1 = leg1.t_window[leg1.Ic[i]]
    ic_2 = leg2.t_window[leg2.Ic[i]]
    
    if leg1.first:
      next_ic = leg1.t_window[leg1.Ic[i+1]]

      time1.append(ic_2 - ic_1)
      time2.append(next_ic - ic_2)

    else:
      next_ic = leg2.t_window[leg2.Ic[i+1]]

      time1.append(ic_1 - ic_2)
      time2.append(next_ic - ic_1)

  return np.array(time1), np.array(time2)

def getStrideTime(leg:LegInfo) -> list:
  """
  Calculate stride time

  Parameters
  ----------
  leg : LegInfo
    Leg from which obtain stride times
  
  Returns
  -------
  list of float
    List of stride times from each cycle
  """
  stride_time = []
  if len(leg.Ic) < 2:
    return stride_time
  
  first_stride = leg.t_window[leg.Ic[0]] - leg.last_time
  if first_stride == 0:
    print(f"Error {leg.name} leg - Impossible stride time")
    return stride_time

  stride_time.append(first_stride)
  for i in range(len(leg.Ic)-1):
    t = leg.t_window[leg.Ic[i+1]] - leg.t_window[leg.Ic[i]]

    if t == 0:
      print(f"Error {leg.name} leg - Impossible stride time")
      continue
    stride_time.append(t)

  return stride_time

def checkFirst(leg1:LegInfo, leg2:LegInfo):
  """
  Check leading leg

  Parameters
  ----------
  leg1, leg2 : LegInfo
  """

  if leg2.cycle == 0 and leg1.cycle == 1:
    leg1.first = True
    leg2.first = False
  
  elif leg2.cycle == 1 and leg1.cycle == 0:
    leg2.first = True
    leg1.first = False

def getTime() -> float:
  """
  Return current time
  """
  return time.time() 

def getParameters(leg1:LegInfo, leg2:LegInfo) -> tuple[dict, dict]:
  """
  Return parameter dictionary for each leg

  Parameters
  ----------
  leg1, leg2 : LegInfo

  Returns
  -------
  params1 : dict {str:float}
    Parameter dictionary of leg1
  params2 : dict {str:float}
    Parameter dictionary of leg2
  """
  Rsteps = 0
  Rcadence = []
  Rstride = []
  Rstep_time = []
  Rstance = []
  Rswing = []
  Rloading_response = []

  Lsteps = 0
  Lstride = []
  Lcadence = []
  Lstep_time = []
  Lstance = []
  Lswing = []
  Lloading_response = []

  # Check leg 1 gait events
  if leg1.connected:
    leg1.findEvents()

    if len(leg1.Tsw) != len(leg1.Ic):
      print(f"Right Initial Contact mismatch ({len(leg1.Ic)}/{len(leg1.Tsw)}) detected")
    
    if len(leg1.Tsw) != len(leg1.Tc):
      print(f"Right Terminal Contact mismatch ({len(leg1.Tc)}/{len(leg1.Tsw)}) detected")

    Rsteps = int(len(leg1.Tsw))
    Rstride = getStrideTime(leg1)
    Rswing, Rstance = getSwingStance(leg1, Rstride)

  # Check leg 2 gait events
  if leg2.connected:
    leg2.findEvents()

    if len(leg2.Tsw) != len(leg2.Ic):
      print(f"Left Initial Contact mismatch ({len(leg2.Ic)}/{len(leg2.Tsw)}) detected")
  
    if len(leg2.Tsw) != len(leg2.Tc):
      print(f"Left Terminal Contact mismatch ({len(leg2.Tc)}/{len(leg2.Tsw)}) detected")

    Lsteps = int(len(leg2.Tsw))
    Lstride = getStrideTime(leg2)
    Lswing, Lstance = getSwingStance(leg2, Lstride)

  
  # Check both legs gait events
  if leg2.connected and leg1.connected:
    # Calculate Right leg Loading Response
    Rloading_response = getLoadingResponse(leg1, leg2, Rstride)   

    # Calculate Left leg Loading Response
    Lloading_response = getLoadingResponse(leg2, leg1, Lstride)
    
    # Calculate Step time for each leg
    Rstep_time, Lstep_time = getStepTime(leg1, leg2)

  # Check resulting parameters 
  Rcadence = 1/Rstep_time * 60 if len(Rstep_time) > 0 else [-1]
  Rstride = Rstride if len(Rstride) > 0 else [-1]
  Rstep_time = Rstep_time if len(Rstep_time) > 0 else [-1]
  Rstance = Rstance if len(Rstance) > 0 else [-1]
  Rswing = Rswing if len(Rswing) > 0 else [-1]
  Rloading_response = Rloading_response if len(Rloading_response) > 0 else [-1]

  Lcadence = 1/Lstep_time * 60 if len(Lstep_time) > 0 else [-1]
  Lstride = Lstride if len(Lstride) > 0 else [-1]
  Lstep_time = Lstep_time if len(Lstep_time) > 0 else [-1]
  Lstance = Lstance if len(Lstance) > 0 else [-1]
  Lswing = Lswing if len(Lswing) > 0 else [-1]
  Lloading_response = Lloading_response if len(Lloading_response) > 0 else [-1]

  params1 = {"Cadence":Rcadence, "Steps":Rsteps, "Stride":Rstride, "StepTime":Rstep_time, "Stance":Rstance, "Swing":Rswing, "LR":Rloading_response}
  params2 = {"Cadence":Lcadence, "Steps":Lsteps, "Stride":Lstride, "StepTime":Lstep_time, "Stance":Lstance, "Swing":Lswing, "LR":Lloading_response}

  return params1, params2