__all__ = [
    "FokkerPlanck",
    "FokkerPlanck1D"
]

from PDE import PDE
import numpy as np
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
import brent_minima as brent
import matplotlib.pyplot as plt

class FokkerPlanck(PDE):

    def V1(self, rho): return rho

    def V2(self, rho, C): return 0

    def V3(self, rho): 
        return 1 / self.V1(rho) / self.d2E(rho) ** 2

    def E(self,rho):
        return self.K * rho * (log(rho) - 1) + self.v * y * rho
    
    def dE(self, rho):
        if type(rho) in [comp.ProxyFunction, comp.GridFunction, fem.CoefficientFunction]:
            return self.K * log(rho) +  self.v * y
        elif type(rho) == np.ndarray:
            return self.K * log(rho) + self.v * self.ypoints_GF.vec.FV().NumPy()
        else: 
            print(type(rho))
            raise NotImplementedError

    def d2E(self, rho):
        return self.K / rho
    

class FokkerPlanck1D(FokkerPlanck):

    dim           = 1
    dirichlet_BCs = ""
    initialB      = "left"
    terminalB     = "right"

    def __init__(
        self,
        order:    int = 1,
        C:        int = 0,
        beta:   float = 0.01,
        printNum: int = 100,
        maxIter:  int = 1001,
        xmax:     int = 1,
        ymax:     int = 1,
        nx:       int = 16,
        ny:       int = 16,
        v:      float = 20,
        K:      float = 1e-2,
        nonLinearSolver: str = "brent",
        brentMin: float = 1e-15,
        brentMax: float = 2
    ):

        self.K, self.v = K, v
        self.nx, self.ny = nx, ny
        self.xmax, self.ymax = xmax, ymax

        if nonLinearSolver == "brent":
            self.brentMin, self.brentMax = brentMin, brentMax

        super().__init__(order, C, beta, printNum, maxIter, nonLinearSolver)

        self.ypoints_GF = GridFunction(self.M)
        self.ypoints_GF.Set(y, definedon = self.mesh.Boundaries(self.terminalB))

    def create_mesh(self):
        return MakeStructured2DMesh(nx = self.nx, ny = self.ny, mapping = lambda x, y : (self.xmax * x, self.ymax * y))

    def rho_exact(self):
        t0 = .01
        y0 = 0.7
        return 1/sqrt(4*np.pi*self.K*(self.beta*x+t0))*exp(-(y-y0+self.v*(self.beta*x+t0))**2/4/self.K/(self.beta*x+t0)) 
    
    def initial_guess(self):
        t0 = .1
        y0 = 0.7
        return 1/sqrt(4*np.pi*self.K*(0+t0))*exp(-(y-y0+self.v*(0+t0))**2/4/self.K/(0+t0)) 
    
    def solveRho(self, m, s, n, rhoBar):
        m_flat = m.vec.FV().NumPy()[:] ** 2
        s_flat = 0 * s.vec.FV().NumPy()[:] 
        n_flat = n.vec.FV().NumPy()[:] ** 2
        rho_bar_flat = rhoBar.vec.FV().NumPy()[:]

        def V1(r): return r
        def V3(r): return 1 / V1(r) / d2E(r) ** 2
        def d2E(r): return self.K / r
        

        def F(rho, rhobar, mbar2, nbar2, sbar2, idx):
            return (  0.5 * (rho - rhobar) ** 2 
                    + 0.5 * mbar2 / (1 + V1(rho)) 
                    + 0.5 * nbar2 / (1 + V3(rho)))
        
        rho = brent.solve_rho(F, m_flat, s_flat, n_flat, rho_bar_flat, 1e-15, self.brentMax) 
  
        return rho
    
    

