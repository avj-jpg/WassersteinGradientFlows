__all__ = [
    "HEAT",
    "HEAT1D",
    "HEAT2D"
]

from PDE import PDE
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh


class HEAT(PDE):
    def V1(self, rho): 
        return rho

    def V2(self, rho, C):
        return IfPos((rho - 1) ** 2 - 1e-12, C * (rho - 1) / log(rho), C * rho)
    
    def V3(self, rho):
        return 1 / rho / self.d2E(rho) ** 2
    
    def E(self,rho):
        return rho * (log(rho) - 1)
    
    def dE(self, rho):
        return log(rho)
    
    def d2E(self, rho):
        return 1 / rho
    
    

class HEAT1D(HEAT):

    dim           = 1
    dirichlet_BCs = "top|bottom"
    initialB      = "left"
    terminalB     = "right"
    nonLinearSolver = "newton"
    
    def __init__(
        self,
        order:    int = 1,
        C:        int = 1,
        beta:   float = 0.01,
        printNum: int = 100,
        maxIter:  int = 1001,
        xmax:     int = 1,
        ymax:     int = 1,
        nx:       int = 16,
        ny:       int = 16,
    ):
        
        self.nx, self.ny = nx, ny
        self.xmax, self.ymax = xmax, ymax

        super().__init__(order, C, beta, printNum, maxIter)

        

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured2DMesh(nx = self.nx, ny = self.ny, mapping = lambda x, y : (self.xmax * x, self.ymax * y))
    
    def rho_exact(self):
        return 1 + 0.5 * cos(2 * pi * y) * exp(-x * (4 * pi ** 2 + self.C) * self.beta) 
    
    def initial_guess(self):
        return 1 + 0.5 * cos(2 * pi * y) * exp(-0 * (4 * pi ** 2 + self.C) * self.beta) 
    
    def solveRho(self, m, s, n, rhoBar):
        # Not utilized.
        return 0
    
class HEAT2D(HEAT):
    dim           = 2
    dirichlet_BCs = "top|bottom|left|right"
    initialB      = "back"
    terminalB     = "front"
    nonLinearSolver = "newton"

    def __init__(
        self,
        order:    int = 1,
        C:        int = 1,
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

        self.nx, self.ny, self.nz = nx, ny, nz
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax

        super().__init__(order, C, beta, printNum, maxIter)



    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured3DMesh(
            nx = self.nx, ny = self.ny, nz = self.nz, 
            mapping=lambda x, y, z : (self.xmax * x, self.ymax * y, self.zmax * z)
            )
    
    def rho_exact(self):
        return 1 + 0.5 * cos(2 * pi * y) * cos(2 * pi * z) * exp(-x * (8 * pi ** 2 + self.C) * self.beta) 
    
    def initial_guess(self):
        return 1 + 0.5 * cos(2 * pi * y) * cos(2 * pi * z) * exp(-0 * (8 * pi ** 2 + self.C) * self.beta) 
    
    
    
    
    