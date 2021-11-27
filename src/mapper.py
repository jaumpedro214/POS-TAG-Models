import numpy as np

class TagNumericalMapper:
  """
  Class to assign each distinct value of a numpy array a numerical tag
  """
  def __init__(self):
    pass
  
  def fit(self, X):
    """
    Implement the first computations to transform the array
    
    X : numpy.array
    """
    self.X_ = X
    self.__define_unique_states()

    return self

  def __define_unique_states(self):
    self._states = np.sort( np.unique(self.X_) )
    self._num_states = len(self._states)
    self._state_to_id = { state:id for id,state in enumerate(self._states) }

    self._func_id_to_state = np.vectorize( lambda id: self._states[id] )
    self._func_state_to_id = np.vectorize( lambda state: self._state_to_id[state] )

  def get_states(self):
    """
    Return the array's unique states
    
    """
    return self._states

  def get_num_states(self):
    """
    Return the number of array's unique states
    
    """
    return self._num_states
    
  def transform(self, X):
    """
    Transform the array (values -> id)
    
    X : numpy.array
    """
    return self._func_state_to_id(X)

  def inverse_transform(self, X):
    """
    Inverse Transform the array (id -> values)
    
    X : numpy.array
    """
    return self._func_id_to_state(X)