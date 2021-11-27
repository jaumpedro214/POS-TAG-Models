import sys
import numpy as np
sys.path.insert(1, '../src/')
from hmm import HmmPosTag

X = [ ['rain', 'rain', 'sun'],
      ['sun',  'sun', 'snow'],
      ['sun', 'rain', 'rain'],
      ['snow', 'sun', 'snow'],
     ]

y = [ ['home','home','fish'],
      ['fish','home','fire'],
      ['fish','home','fire'],
      ['fire','fire','home'],
     ]

hmm_tag = HmmPosTag()
hmm_tag.fit( np.array(X), np.array(y) )

print( hmm_tag.predict( np.array([['fish','fish','home', 'fish', 'fish'],]) ) )