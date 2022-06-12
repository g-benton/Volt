__version__ = 'alpha'
from .kernels import BMKernel, VolatilityKernel
from .models import BMGP, VoltronGP, MultitaskBMGP
from .train_utils import LearnGPCV
from .option_utils import *
try:
    from .robinhood_utils import GetStockData
except:
    print("Warning no robinhood utils.")
    
from .rollout_utils import Rollouts, GeneratePrediction
    
