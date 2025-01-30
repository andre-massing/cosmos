# %%
from ngsolve import *
from cosmos.solvers.base import Base

class BaseSolver(Base):

    def __init__(self, fes_order=1, **kwargs):
        
        self.fes_order = fes_order
        self.params = kwargs
        self.sol = []
        self.problem = None

    def initialize(self):
        
        raise NotImplementedError
    
    def solve_step(self):
        
        raise NotImplementedError
    
    def update(self):
        
        raise NotImplementedError
    
    def get_solution(self):
        
        raise NotImplementedError
    
    def set_solution(self, **kwargs):
        
        raise NotImplementedError

    def draw_solution(self, **kwargs):
        
        raise NotImplementedError
    
    def save_solution(self, **kwargs):
        
        raise NotImplementedError
    
    def compute_error(self, u_ex, norm):
        
        raise NotImplementedError

    def print_info(self):
        
        raise NotImplementedError

