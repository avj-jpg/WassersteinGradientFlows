__all__ = [
    "PME",
    "PME1D",
    "PME2D"
]

from PDE import PDE
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh


class PME(PDE):

    alpha = NotImplemented

    def V1(self, rho): 
        return rho

    def V2(self, rho, C):
        v   = -(self.alpha - 1) / (self.alpha * self.beta) * rho ** (1 - self.alpha)
        rm  = self.rho_exact()** self.alpha
        _v  = self.rho_exact().Diff(x) - self.beta * rm.Diff(y).Diff(y) 
        if self.dim == 2:
            _v += - self.beta * rm.Diff(z).Diff(z)
        v *= _v
        return v
    
    def V3(self, rho):
        return 1 / rho / self.d2E(rho) ** 2
    
    def E(self,rho):
        return rho ** self.alpha / (self.alpha - 1)
    
    def dE(self, rho):
        return rho ** (self.alpha - 1) * self.alpha / ( self.alpha - 1)
    
    def d2E(self, rho):
        return rho ** (self.alpha - 2) * self.alpha

class PME1D(PME):

    dim           = 1
    dirichlet_BCs = "top|bottom"
    initialB      = "left"
    terminalB     = "right"
    
    def __init__(
        self,
        order:    int = 1,
        C:        int = 1,
        alpha:  float = 2,
        beta:   float = 0.01,
        printNum: int = 100,
        maxIter:  int = 1001,
        xmax:     int = 1,
        ymax:     int = 1,
        nx:       int = 16,
        ny:       int = 16,
    ):
        
        self.alpha = alpha
        self.nx, self.ny = nx, ny
        self.xmax, self.ymax = xmax, ymax

        super().__init__(order, C, beta, printNum, maxIter)
        # C not utilized
        

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured2DMesh(nx = self.nx, ny = self.ny, mapping = lambda x, y : (self.xmax * x, self.ymax * y))
    
    def rho_exact(self):
        return (2 - x) + 0.1 * exp(-0.1 * x) * cos(2 * pi * y)
    
    def initial_guess(self):
        return (2 - x) + 0.1 * exp(-0.1 * 0) * cos(2 * pi * y) 
    
class PME2D(PME):
    dim           = 2
    dirichlet_BCs = "top|bottom|left|right"
    initialB      = "back"
    terminalB     = "front"

    def __init__(
        self,
        order:    int = 1,
        C:        int = 1,
        alpha:  float = 2,
        beta:   float = 0.01,
        printNum: int = 100,
        maxIter:  int = 1001,
        xmax:     int = 1,
        ymax:     int = 1,
        zmax:     int = 1,
        nx:       int = 8,
        ny:       int = 8,
        nz:       int = 8
    ):
        
        self.alpha = alpha
        self.nx, self.ny, self.nz = nx, ny, nz
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax

        super().__init__(order, C, beta, printNum, maxIter)
        # C not utilized


    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured3DMesh(
            nx = self.nx, ny = self.ny, nz = self.nz, 
            mapping=lambda x, y, z : (self.xmax * x, self.ymax * y, self.zmax * z)
            )
    
    def rho_exact(self):
        return (2 - x) + 0.1 * exp(-0.1 * x) * cos(2 * pi * y) * cos(2 * pi * z) 
    
    def initial_guess(self):
        return (2 - x) + 0.1 * exp(-0.1 * 0) * cos(2 * pi * y) * cos(2 * pi * z) 
    
    
    
    
    