__all__ = [
    "HEAT",
    "HEAT1D",
    "HEAT2D"
]

from PDE import PDE
import numpy as np
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
import brent_minima as brent


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
    
    def solveRho(self, m, s, n, rhoBar):
        s_flat = s.vec.FV().NumPy()[:] ** 2
        rho_bar_flat = rhoBar.vec.FV().NumPy()[:]

        if self.dim == 1:
            m_flat = m.vec.FV().NumPy()[:] ** 2
            n_flat = n.vec.FV().NumPy()[:] ** 2
        else:
            m_flat = m[0].vec.FV().NumPy()[:] ** 2 + m[1].vec.FV().NumPy()[:] ** 2
            n_flat = n[0].vec.FV().NumPy()[:] ** 2 + n[1].vec.FV().NumPy()[:] ** 2

        def V1(r): return r
        def V2(r): return self.C * (r - 1) / np.log(r)
        def dE(r): return np.log(r)
        def d2E(r): return 1 / r
        def V3(r): return 1 / r / d2E(r)**2

        def F(rho, rhobar, mbar2, nbar2, sbar2, *args):
            return (
                0.5 * (rho - rhobar)**2 +
                0.5 * mbar2 / (1 + V1(rho)) +
                0.5 * nbar2 / (1 + V3(rho)) +
                0.5 * sbar2 / (1 + V2(rho)) +
                0.5 * self.beta**2 * V2(rho) * dE(rho)**2
            )

        rho = brent.solve_rho(F, m_flat, s_flat, n_flat, rho_bar_flat, self.brentMin, self.brentMax)
        return rho
    
    def __str__(self):
        """String representation."""
        return "\n".join(
            [
                f"Heat Equation",
                f"  C: {self.C}",
                f"  beta: {self.beta}",
                f"  Time domain: [0, {self.xmax}]",
                f"  Spatial domain: [0, {self.ymax}]" + (f" x [0, {self.zmax}]" if self.dim == 2 else ""),
                f"  Discretization size: {self.nx} x {self.ny}" + (f" x {self.nz}" if self.dim == 2 else ""),
                f"  Order: {self.order}",
                f"  Degrees of freedom (primal space):  {self.W.ndof}",
                f"  Degrees of freedom (dual space):  {self.V.ndof}",
                f"  Non-linear solver: {self.nonLinearSolver}" + (
                f", with interval: [{self.brentMin}, {self.brentMax}]" if self.nonLinearSolver == 'brent' else ""),
                f"  PDHG iterations: {self.maxIter - 1}",
                "\n"
            ]
        )  
    

class HEAT1D(HEAT):

    dim           = 1
    dirichlet_BCs = "top|bottom"
    initialB      = "left"
    terminalB     = "right"
    
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
        nonLinearSolver: str = "brent",
        brentMin: float = 0.4,
        brentMax: float = 1.6
    ):
        
        self.nx, self.ny = nx, ny
        self.xmax, self.ymax = xmax, ymax
        if nonLinearSolver == "brent":
            self.brentMin, self.brentMax = brentMin, brentMax

        super().__init__(order, C, beta, printNum, maxIter, nonLinearSolver)

        

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured2DMesh(nx = self.nx, ny = self.ny, mapping = lambda x, y : (self.xmax * x, self.ymax * y))
    
    def rho_exact(self):
        return 1 + 0.5 * cos(2 * pi * y) * exp(-x * (4 * pi ** 2 + self.C) * self.beta) 
    
    def initial_guess(self):
        return 1 + 0.5 * cos(2 * pi * y) * exp(-0 * (4 * pi ** 2 + self.C) * self.beta) 
    
    
    
class HEAT2D(HEAT):
    dim           = 2
    dirichlet_BCs = "top|bottom|left|right"
    initialB      = "back"
    terminalB     = "front"

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
        nz:       int = 8,
        nonLinearSolver = "brent",
        brentMin: float = 0.4,
        brentMax: float = 1.6
    ):

        self.nx, self.ny, self.nz = nx, ny, nz
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax
        if nonLinearSolver == "brent":
            self.brentMin, self.brentMax = brentMin, brentMax

        super().__init__(order, C, beta, printNum, maxIter, nonLinearSolver)



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
    
    
    
    
    