import numpy as np
class ViterbiSolver:
  """
  Virtebi algorithm implementation

  tran: Transmission probabilities matrix S x S
  emiss: Emission probabilities matrix S x E
  start: Initial states probabilites S x 1
  eps: Smoothing value to avoid log(0)
  """
  def __init__(self, tran, emiss, start, eps=1e-12):
    self.tran = np.log( tran+eps)
    self.emiss = np.log( emiss+eps )
    self.start = np.log( start+eps )

  def solve(self, seq):
    """
    Solve the probability optimization problem to a given sequence

    seq: the emission's sequence 
    """
    seq_len = len(seq)
    num_states = self.tran.shape[0]

    trellis_proba = np.zeros( (num_states, seq_len) )
    trellis_point = np.zeros( (num_states, seq_len), dtype=np.short )

    ## Initial probabilities
    trellis_proba[:, 0] = self.start + self.emiss[:, seq[0] ]
    
    for i in range( 1, seq_len ):
        trellis_previous_proba = trellis_proba[:, i-1]
        trellis_previous_proba = trellis_previous_proba + self.tran.T
        trellis_previous_proba = trellis_previous_proba.T + self.emiss[:, seq[i]]

        trellis_proba[:, i] = np.max( trellis_previous_proba, axis=0 )
        trellis_point[:, i] = np.argmax( trellis_previous_proba, axis=0 )

    states_path = np.zeros( (seq_len), dtype=np.short )
    states_path[seq_len-1] = np.argmax( trellis_proba[:, seq_len-1] )
    
    for i in range(seq_len-1, 0, -1):
      states_path[i-1] = trellis_point[ states_path[i], i ]

    return np.array(states_path)