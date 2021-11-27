from mapper import TagNumericalMapper
from viterbi import ViterbiSolver
import numpy as np
from sklearn.model_selection import train_test_split

class HiddeMarkovModel:
  """
  Hidden Markov Model

  """
  def __init__(self):
    pass

  def fit(self, X_, y_):
    self.X_ = X_.copy()
    self.y_ = y_.copy()

    self.__check_numerical_imput()
    self.__calculate_start_probability()
    self.__calculate_transition_probability()
    self.__calculate_emission_probability()
    self.__create_virtebi_solver()

  def __check_numerical_imput(self):
    """
    TODO: implement the method itself
    Check if the inputs are non-numerical
    
    """
    self.__transform_numerical_imput()

  def __transform_numerical_imput(self):
    """
    Transform any non-numerical imputs into numerical using a TagNumericalMapper
    
    """
    self.X_transform = TagNumericalMapper()
    self.X_transform.fit( self.X_ )
    
    self.y_transform = TagNumericalMapper()
    self.y_transform.fit( self.y_ )

    self._num_hidden_states = self.X_transform.get_num_states()
    self._num_emission_states = self.y_transform.get_num_states()

    self.X_map_ = self.X_transform.transform( self.X_ )
    self.y_map_ = self.y_transform.transform( self.y_ )

  def get_hidden_states(self):
    """
    Return unique hidden states
    
    """
    return self.X_transform.get_states()

  def get_emission_states(self):
    """
    Return unique emission states
    
    """
    return self.y_transform.get_states()

  def __calculate_start_probability(self):
    """
    Calculate each hidden states' start probabilities 
    
    """
    states, counts = np.unique( self.X_map_[:,0], return_counts=True )
    probabilites = counts/np.sum(counts)

    self.start_state_probability_ = np.zeros( self._num_hidden_states )
    self.start_state_probability_[ states ] = probabilites

  def __calculate_transition_probability(self):
    """
    Calculate the transition probabilities matrix
    
    """
    print("__calculate_transition_probability")
    # Creating a vector with all the pairs in the Hidden State data
    X_pairs = np.concatenate( [np.array([self.X_map_[:,i], self.X_map_[:,i+1]]) 
                               for i in range(self.X_map_.shape[1]-1)], axis=1)
    X_pairs = X_pairs.T
    
    self.transition_matrix_ = np.zeros( (self._num_hidden_states, self._num_hidden_states), 
                                        dtype=np.double)
    for i in range(X_pairs.shape[0]):
      self.transition_matrix_[ X_pairs[i,0], X_pairs[i,1] ] += 1

    # Turning counts into probabilities
    transition_sum = np.expand_dims( np.sum(self.transition_matrix_,axis=1), axis=1)
    self.transition_matrix_ = self.transition_matrix_/transition_sum
    self.transition_matrix_ = np.nan_to_num(self.transition_matrix_, copy=False)

  def __calculate_emission_probability(self):
    """
    Calculate the emission probabilities matrix
    
    """
    print("__calculate_emission_probability")
    # Creating a vector with all the pairs (Hidden State, Emission State)
    X_y_pairs = np.vstack( [self.X_map_.flatten(), self.y_map_.flatten()] ).T
    
    self.emission_matrix_ = np.zeros( (self._num_hidden_states, self._num_emission_states),
                                      dtype=np.double)

    for i in range(X_y_pairs.shape[0]):
      self.emission_matrix_[ X_y_pairs[i,0], X_y_pairs[i,1] ] += 1

    # Turning counts into probabilities
    emission_sum = np.expand_dims( np.sum(self.emission_matrix_,axis=1), axis=1)
    self.emission_matrix_ = self.emission_matrix_/emission_sum
    self.emission_matrix_ = np.nan_to_num(self.emission_matrix_, copy=False)

  def __create_virtebi_solver(self):
    """
    Instantiate a viterbi solver
    
    """
    self.solver = ViterbiSolver( self.transition_matrix_, 
                                 self.emission_matrix_, 
                                 self.start_state_probability_ )
    
  def predict_single(self, y_seq):
    """
    Solve the problem to a sequence
    
    """
    y_seq = self.y_transform.transform( y_seq )
    X = self.solver.solve(y_seq)

    # Decoding X to the original names
    X_r = self.X_transform.inverse_transform( X )
    return X_r

  def predict(self, y):
    X_r = []
    for y_seq in y:
      X_r.append( self.predict_single(y_seq) )

    return X_r

class HmmPosTag:
  """
  Implement a POS-TAG model with a Hidden Markov Model
  
  p_mask: probability of masking a emission token
  sample_size: Number of aditional examples with masked tokens to be added
  mask_token: text that represents a masked token
  random_state: random state to reproducibility
  """
  def __init__( self, p_mask=0.15, sample_size=0.25, mask_token="<MASKED>", random_state=None ):
    self.p_mask = p_mask
    self.sample_size = sample_size
    self.random_state = random_state
    np.random.seed(random_state)

    self.mask_token = mask_token

  def fit(self, X, y):
    self.X_ = X.copy()
    self.y_ = y.copy()

    if( self.p_mask > 0 ):
      self.__mask_train_tokens()

    self.hmm_model = HiddeMarkovModel()
    self.hmm_model.fit( self.X_, self.y_ )

    self.__create_vocabulary()
  
  def __create_vocabulary(self):
    """
    Create model's vocabulary (unique emission states set)
    """
    self.vocabulary_ = set( self.hmm_model.get_emission_states() )

  def __mask_train_tokens(self):
    """
    Samples a set of sample_size entries to mask its emission tokens using p_mask. 
    The set is appended to the training set
    """
    _, self.X_sample_, _, self.y_sample_ = train_test_split(self.X_, self.y_, 
                                                            test_size=self.sample_size, 
                                                            random_state=self.random_state 
                                                            )

    self.y_sample_[ np.random.random( self.y_sample_.shape ) <= self.p_mask ] = self.mask_token

    self.X_ = np.vstack( [self.X_, self.X_sample_] )
    self.y_ = np.vstack( [self.y_, self.y_sample_] )

    
  def predict(self, y):
   y = self.handle_missing_tokens( y )

   return self.hmm_model.predict( y )

  def handle_missing_tokens(self, y):
    """
    Mask out-of-vocabulary tokens with mask_token
    
    """
    f = np.vectorize( lambda s: s not in self.vocabulary_ )
    y[ f(y) ] = self.mask_token
    return y