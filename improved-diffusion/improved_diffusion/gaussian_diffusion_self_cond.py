# Self-conditioning selection for the licensed-base AP-MolDiff diffusion adapter.
# All probability equations are inherited from the Apache-2.0 Diffusion-LM base.
from .gaussian_diffusion import *
from .gaussian_diffusion import GaussianDiffusion as _ConditionalDiffusion
from .gaussian_diffusion import _extract_into_tensor

class GaussianDiffusion(_ConditionalDiffusion):
    _enable_sc = True
