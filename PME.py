__all__ = [
    "PME",
    "PME1D",
    "PME2D"
]

from PDE import PDE
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
import numpy as np
import brent_minima as brent


"""
Compile brent_minima as:

c++ -O3 -Wall -shared -std=c++14 -fPIC \
    $(python3 -m pybind11 --includes) \
    brent_minima.cpp -o brent_minima$(python3-config --extension-suffix)

"""

class PME(PDE):

    #alpha = NotImplemented

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

    dim                   = 1
    dirichlet_BCs         = "top|bottom"
    initialB              = "left"
    terminalB             = "right"
    nonLinearSolver       = "newton"
    
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
    
    def solveRho(self, m, s, n, rhoBar):
        # Not utilized.
        return 0
    
    
class PME2D(PME):
    dim             = 2
    dirichlet_BCs   = "top|bottom|left|right"
    initialB        = "back"
    terminalB       = "front"
    nonLinearSolver = "newton"

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
    
    def solveRho(self, m, s, n, rhoBar):
        # Not utilized.
        return 0

class PME1DBB(PDE):

    dim           = 1
    dirichlet_BCs = "top|bottom"
    initialB      = "left"
    terminalB     = "right"
    nonLinearSolver = "brent"
    
    
    def __init__(
        self,
        order:    int = 1,
        C:        int = 0,
        alpha:  float = 2,
        beta:   float = 0.01,
        printNum: int = 100,
        maxIter:  int = 1001,
        xmax:     int = 1,
        ymax:     int = 1,
        nx:       int = 16,
        ny:       int = 16,
        sc: float = 0.1
    ):
        
        
        self.sc = sc
        self.alpha = alpha
        self.nx, self.ny = nx, ny
        self.xmax, self.ymax = xmax, ymax
        self.k = 1/(self.alpha-1+2/self.dim)
        self.s0 = self.sc*self.k*(self.alpha-1)/2/self.dim/self.alpha

        if C != 0: raise Exception("C must be 0 for Barenblatt data.")
        if self.alpha < 2: raise Exception("alpha must >= 2.")

        super().__init__(order, C, beta, printNum, maxIter)
        

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured2DMesh(nx = self.nx, ny = self.ny, mapping = lambda x, y : (self.xmax * x, self.ymax * y))
    
    def rho_exact(self):
        r2 = (y-0.5)**2 if self.dim == 1 else (y-0.5)**2 + (z-0.5)**2

        val = 1-r2/self.sc/(self.beta*x+1) ** (2*self.k/self.dim) 
        return (self.beta*x+1)**(-self.k)*IfPos(val, val**(1/(self.alpha-1)), 0)
    
    def initial_guess(self):
        r2 = (y-0.5)**2 if self.dim == 1 else (y-0.5)**2 + (z-0.5)**2
        val = 1-r2/self.sc/(self.beta*0+1) ** (2*self.k/self.dim) 
        return (self.beta*0+1)**(-self.k)*IfPos(val, val**(1/(self.alpha-1)), 0)
    
    def V1(self, rho): 
        return self.s0*rho

    def V2(self, rho, C):
        return 0
    
    def V3(self, rho):
        return 1 / self.V1(rho) / self.d2E(rho) ** 2
    
    def E(self,rho):
        return rho ** self.alpha / (self.alpha - 1)
    
    def dE(self, rho):
        return rho ** (self.alpha - 1) * self.alpha / ( self.alpha - 1)
    
    def d2E(self, rho):
        return rho ** (self.alpha - 2) * self.alpha
    
    def solveRho(self, m, s, n, rhoBar):
        m_flat = m.vec.FV().NumPy()[:]
        s_flat = 0*s.vec.FV().NumPy()[:]
        n_flat = n.vec.FV().NumPy()[:]
        rho_bar_flat = rhoBar.vec.FV().NumPy()[:]

        # def V1(r): 
        #     return r
        # def V2(x): 
        #     if x < 0: return 0
        #     if (x-1) ** 2 - 1e-12 >= 0: return self.C  * (x - 1) / np.log(x)
        #     else: return self.C * x
        # def dE(x): 
        #     if x <= 0: return 1e12
        #     return np.log(x)
        # def d2E(x): return 1 / x
        # def V3(x): return 1 / x / d2E(x) ** 2

        def V1(r): 
            return self.s0*r
        def dE(r): return r ** (self.alpha - 1) * self.alpha / (self.alpha - 1)

        def d2E(r): return r ** (self.alpha - 2) * self.alpha

        def V2(r): return 0

        def V3(r): 
            #if r < 1e-12: return 0
            return 1 / V1(r) / d2E(r) ** 2


        rho_flat = np.zeros_like(m_flat)

        def F(rho, rhobar, mbar2, nbar2, sbar2):
            return 0.5*(rho-rhobar)**2 + 0.5*mbar2/(1+V1(rho)) + 0.5*nbar2/(1+V3(rho)) + 0.5*sbar2/(1+V2(rho)) +0.5*self.beta**2*V2(rho)*dE(rho)*dE(rho)
        

        for idx in range(m_flat.shape[0]):
            mbar2, nbar2, sbar2 = m_flat[idx]**2, n_flat[idx]**2, s_flat[idx]**2
            obj_fixed_idx = lambda rho : F(rho, rho_bar_flat[idx], mbar2, nbar2, sbar2)
            rho_flat[idx] = brent.brent_find_minima(obj_fixed_idx, 1e-12, 2.0)
        
        return rho_flat
    
    
    
    

    
    
    
    