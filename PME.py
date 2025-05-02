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
import matplotlib.pyplot as plt


"""
Compile brent_minima as:

c++ -O3 -Wall -shared -std=c++14 -fPIC \
    $(python3 -m pybind11 --includes) \
    brent_minima.cpp -o brent_minima$(python3-config --extension-suffix)

"""

class PME(PDE):

    def V1(self, rho): 
        return rho

    def V2(self, rho, C):
        v = 1 / self.beta / self.dE(rho)
        _v  = -self.rho_exact().Diff(x) + self.beta * (self.rho_exact() ** self.alpha).Diff(y).Diff(y) 
        if self.dim == 2:
            _v += + self.beta * (self.rho_exact() ** self.alpha).Diff(z).Diff(z) 
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
        def V2(r, idx): 
            return self.rhoMfdRes_flat[idx] / self.beta / self.dE(r)
        def V3(r):  return 1 / V1(r) / d2E(r) ** 2

        def dE(r): return r ** (self.alpha - 1) * self.alpha / (self.alpha - 1)
        def d2E(r): return r ** (self.alpha - 2) * self.alpha
        

        def F(rho, rhobar, mbar2, nbar2, sbar2, idx):
            return (
                0.5 * (rho - rhobar) ** 2   +
                0.5 * mbar2 / (1 + V1(rho)) +
                0.5 * nbar2 / (1 + V3(rho)) +
                0.5 * sbar2 / (1 + V2(rho, idx)) +
                0.5 * self.beta ** 2 * V2(rho, idx) * dE(rho) * dE(rho)
            )

        rho = brent.solve_rho(F, m_flat, s_flat, n_flat, rho_bar_flat, self.brentMin, self.brentMax)
        return rho
    
    def __str__(self):
        """String representation."""
        return "\n".join(
            [
                f"Porous Medium Equation",
                f"  C: {self.C}",
                f"  alpha: {self.alpha}",
                f"  beta: {self.beta}",
                f"  Time domain: [0, {self.xmax}]",
                f"  Spatial domain: [0, {self.ymax}]" + (f" x [0, {self.zmax}]" if self.dim == 2 else ""),
                f"  Discretization size: {self.nx} x {self.ny}" + (f" x {self.nz}" if self.dim == 2 else ""),
                f"  Order: {self.order}",
                f"  Degrees of freedom (primal space):  {self.W.ndof}",
                f"  Degrees of freedom (dual space):  {self.V.ndof}",
                f"  Non-linear solver: {self.nonLinearSolver}" + (
                f", with interval: [{self.brentMin}, {self.brentMax}]" if self.nonLinearSolver == 'brent' else ""),
                f"  PDHG iterations: {self.maxIter - 1}"
            ]
        )  


class PME1D(PME):

    dim                   = 1
    dirichlet_BCs         = "top|bottom"
    initialB              = "left"
    terminalB             = "right"
    
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
        nonLinearSolver: str = "brent",
        brentMin: float = 0.8,
        brentMax: float = 2.2
    ):
        
        self.alpha = alpha
        self.nx, self.ny = nx, ny
        self.xmax, self.ymax = xmax, ymax
        if nonLinearSolver == "brent":
            self.brentMin, self.brentMax = brentMin, brentMax

        # C must be > 0
        super().__init__(order, C, beta, printNum, maxIter, nonLinearSolver)

        # Residual of the exact manufactured solution
        self.rhoMfdRes = GridFunction(self.W)
        rm  = -self.rho_exact().Diff(x) + self.beta * (self.rho_exact() ** self.alpha).Diff(y).Diff(y) 
        self.rhoMfdRes.Interpolate(rm)
        self.rhoMfdRes_flat = self.rhoMfdRes.vec.FV().NumPy()[:]

    

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured2DMesh(nx = self.nx, ny = self.ny, mapping = lambda x, y : (self.xmax * x, self.ymax * y))
    
    def rho_exact(self):
        return (2 - x) + 0.1 * exp(-0.1 * x) * cos(2 * pi * y)
    
    def initial_guess(self):
        return (2 - 0) + 0.1 * exp(-0.1 * 0) * cos(2 * pi * y) 
    
 
class PME2D(PME):
    dim             = 2
    dirichlet_BCs   = "top|bottom|left|right"
    initialB        = "back"
    terminalB       = "front"

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
        nz:       int = 8,
        nonLinearSolver: str = "brent",
        brentMin: float = 0.8,
        brentMax: float = 2.2
    ):
        
        self.alpha = alpha
        self.nx, self.ny, self.nz = nx, ny, nz
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax
        if nonLinearSolver == "brent":
            self.brentMin, self.brentMax = brentMin, brentMax

        # C not utilized but must be > 0
        super().__init__(order, C, beta, printNum, maxIter, nonLinearSolver)
        
        # Residual of the exact manufactured solution
        self.rhoMfdRes = GridFunction(self.W)
        rm  = -self.rho_exact().Diff(x) 
        rm += self.beta * (self.rho_exact() ** self.alpha).Diff(y).Diff(y)
        rm += self.beta * (self.rho_exact() ** self.alpha).Diff(z).Diff(z)
        
        self.rhoMfdRes.Interpolate(rm)
        self.rhoMfdRes_flat = self.rhoMfdRes.vec.FV().NumPy()[:]


    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured3DMesh(
            nx = self.nx, ny = self.ny, nz = self.nz, 
            mapping=lambda x, y, z : (self.xmax * x, self.ymax * y, self.zmax * z)
            )
    
    def rho_exact(self):
        return (2 - x) + 0.1 * exp(-0.1 * x) * cos(2 * pi * y) * cos(2 * pi * z) 
    
    def initial_guess(self):
        return (2 - 0) + 0.1 * exp(-0.1 * 0) * cos(2 * pi * y) * cos(2 * pi * z) 

# Barenblatt --------------------------------------------------------------------------------------------

class PME1DBB(PDE):

    dim           = 1
    dirichlet_BCs = "top|bottom"
    initialB      = "left"
    terminalB     = "right"
    
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
        sc: float = 0.1,
        nonLinearSolver: str = "brent",
        brentMin: float = 1e-15,
        brentMax: float = 2
    ):
        
        
        self.sc = sc
        self.alpha = alpha
        self.nx, self.ny = nx, ny
        self.xmax, self.ymax = xmax, ymax
        if nonLinearSolver == "brent":
            self.brentMin, self.brentMax = brentMin, brentMax

        self.k = 1 / (self.alpha - 1 + 2 / self.dim)
        self.s0 = self.sc * self.k * (self.alpha - 1) / 2 / self.dim / self.alpha

        if C != 0: raise Exception("C must be 0 for Barenblatt data.")
        if self.alpha < 2: raise Exception("alpha must >= 2.")

        super().__init__(order, C, beta, printNum, maxIter, nonLinearSolver)
        

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured2DMesh(nx = self.nx, ny = self.ny, mapping = lambda x, y : (self.xmax * x, self.ymax * y))
    
    def rho_exact(self):
        r2 = (y - 0.5) ** 2 if self.dim == 1 else (y - 0.5) ** 2 + (z - 0.5) ** 2

        val = 1 - r2 / self.sc / (self.beta * x + 1) ** (2 * self.k / self.dim) 
        return (self.beta * x + 1) ** (-self.k) * IfPos(val, val ** (1 / (self.alpha - 1)), 0)
    
    def initial_guess(self):
        r2 = (y - 0.5) ** 2 if self.dim == 1 else (y - 0.5) ** 2 + (z - 0.5) ** 2
        val = 1 - r2 / self.sc / (self.beta * 0 + 1) ** (2 * self.k / self.dim) 
        return (self.beta * 0 + 1) ** (-self.k) * IfPos(val, val ** (1 / (self.alpha - 1)), 0)
    
    def V1(self, rho): 
        return self.s0 * rho

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
        m_flat = m.vec.FV().NumPy()[:] ** 2
        s_flat = 0 * s.vec.FV().NumPy()[:] 
        n_flat = n.vec.FV().NumPy()[:] ** 2
        rho_bar_flat = rhoBar.vec.FV().NumPy()[:]

        def V1(r): return self.s0*r
        def V2(r): return 0
        def V3(r): return 1 / V1(r) / d2E(r) ** 2

        def dE(r): return r ** (self.alpha - 1) * self.alpha / (self.alpha - 1)
        def d2E(r): return r ** (self.alpha - 2) * self.alpha

        def F(rho, rhobar, mbar2, nbar2, sbar2, *args):
            return (  0.5 * (rho - rhobar) ** 2 
                    + 0.5 * mbar2 / (1 + V1(rho)) 
                    + 0.5 * nbar2 / (1 + V3(rho)) 
                    + 0.5 * sbar2 / (1 + V2(rho)) 
            )
                    #+ 0.5 * self.beta ** 2 * V2(rho) * dE(rho) * dE(rho) )
        
        rho = brent.solve_rho(F, m_flat, s_flat, n_flat, rho_bar_flat, 1e-15, 2.0) 
  
        return rho
    

    def animateWithErr(self, fig, ax, save="True", color='r'):
        if self.dim != 1: raise NotImplementedError
        if len(ax) !=2: raise Exception("Length of ax must be two.")

        t_ = self.getTimeIntPoints()
        x_ = self.getSpaceIntPoints()

        ax1, ax2 = ax[0], ax[1]
        line1, = ax1.plot([], [], '--k', label=r'Initial condition')
        line2, = ax1.plot([], [], ':', label=r'Exact solution', color=color,linewidth=3)
        line4, = ax1.plot([], [], '-', label=r'Numerical solution', color=color)

        ax1.set_xlim(min(x_), max(x_))
        ax1.set_ylim(0,1.05)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$\rho(x)$')
        fig.subplots_adjust(bottom=0.2)  
        ax[0].legend(
            loc='upper center', 
            bbox_to_anchor=(1.2, -0.15), 
            ncol=3, 
            frameon=False
        )


        error_line, = ax2.semilogy([], [], '-k', label=r'Absolute error')
        ax2.set_xlim(min(x_), max(x_))
        ax2.set_ylim(1e-8, 1)  
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'Absolute Error')

        initial_y_rhohex = self.evaluateQuadratureFun(t_[0], x_, self.rhohex)

        def anim(i):
            current_t = t_[i]
            current_y_rhohex = np.array(self.evaluateQuadratureFun(current_t, x_, self.rhohex))
            current_y_rhoh = np.array(self.evaluateQuadratureFun(current_t, x_, self.rhoh))
            
            line1.set_data(x_, initial_y_rhohex)
            line2.set_data(x_, current_y_rhohex)
            line4.set_data(x_, current_y_rhoh)

            error_line.set_data(x_, abs(current_y_rhohex - current_y_rhoh))

            ax1.set_title(f'Time $t = {current_t:.3f}$')
            #ax2.set_title(f'Error at Time $t = {current_t:.3f}$')

            return line1, line2, line4, error_line

        # Create and keep animation alive
        ani = animation.FuncAnimation(fig, anim, frames=len(t_), interval=25, blit=True, repeat=False)
        plt.show()
        if save:
            filename = "Err_alpha_" + str(self.alpha) + "_order_" + str(self.order) + "_nx_" + str(self.nx) + "_ny_" + str(self.ny)
            ani.save(filename + ".gif", writer='pillow', fps=20)
            

class PME2DBB(PDE):

    dim             = 2
    dirichlet_BCs   = "top|bottom|left|right"
    initialB        = "back"
    terminalB       = "front"
    
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
        zmax:     int = 1,
        nx:       int = 8,
        ny:       int = 8,
        nz:       int = 8,
        sc: float = 0.1,
        nonLinearSolver: str = "brent",
        brentMin: float = 1e-15,
        brentMax: float = 2
    ):
        
        
        self.sc = sc
        self.alpha = alpha
        self.nx, self.ny, self.nz = nx, ny, nz
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax
        if nonLinearSolver == "brent":
            self.brentMin, self.brentMax = brentMin, brentMax

        self.k = 1 / (self.alpha - 1 + 2 / self.dim)
        self.s0 = self.sc * self.k * (self.alpha - 1) / 2 / self.dim / self.alpha

        if C != 0: raise Exception("C must be 0 for Barenblatt data.")
        if self.alpha < 2: raise Exception("alpha must >= 2.")

        super().__init__(order, C, beta, printNum, maxIter, nonLinearSolver)
        

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        return MakeStructured3DMesh(
            nx = self.nx, ny = self.ny, nz = self.nz, 
            mapping=lambda x, y, z : (self.xmax * x, self.ymax * y, self.zmax * z)
            )
    
    def rho_exact(self):
        r2 = (y - 0.5) ** 2 if self.dim == 1 else (y - 0.5) ** 2 + (z - 0.5) ** 2

        val = 1 - r2 / self.sc / (self.beta * x + 1) ** (2 * self.k / self.dim) 
        return (self.beta * x + 1) ** (-self.k) * IfPos(val, val ** (1 / (self.alpha - 1)), 0)
    
    def initial_guess(self):
        r2 = (y - 0.5) ** 2 if self.dim == 1 else (y - 0.5) ** 2 + (z - 0.5) ** 2
        val = 1 - r2 / self.sc / (self.beta * 0 + 1) ** (2 * self.k / self.dim) 
        return (self.beta * 0 + 1) ** (-self.k) * IfPos(val, val ** (1 / (self.alpha - 1)), 0)
    
    def V1(self, rho): 
        return self.s0 * rho

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
        m_flat = m[0].vec.FV().NumPy()[:] ** 2 + m[1].vec.FV().NumPy()[:] ** 2
        s_flat = 0 * s.vec.FV().NumPy()[:] 
        n_flat = n[0].vec.FV().NumPy()[:] ** 2 + n[0].vec.FV().NumPy()[:] ** 2
        rho_bar_flat = rhoBar.vec.FV().NumPy()[:]

        def V1(r): return self.s0*r
        def V2(r): return 0
        def V3(r): return 1 / V1(r) / d2E(r) ** 2

        def dE(r): return r ** (self.alpha - 1) * self.alpha / (self.alpha - 1)
        def d2E(r): return r ** (self.alpha - 2) * self.alpha

        def F(rho, rhobar, mbar2, nbar2, sbar2, *args):
            return (  0.5 * (rho - rhobar) ** 2 
                    + 0.5 * mbar2 / (1 + V1(rho)) 
                    + 0.5 * nbar2 / (1 + V3(rho)) 
                    + 0.5 * sbar2 / (1 + V2(rho)) 
                    + 0.5 * self.beta ** 2 * V2(rho) * dE(rho) * dE(rho) )
        
        rho = brent.solve_rho(F, m_flat, s_flat, n_flat, rho_bar_flat, 1e-15, 2.0) 
  
        return rho
    def evaluateQuadratureFun(self, t, xvals, yvals, qfu):
        # t: scalar
        # xvals, yvals: meshgrid
        # qfu: spacetime quadrature function
        
        evals = np.zeros_like(xvals)
        gfu = GridFunction(L2(self.mesh, order=self.order-1))
        gfu.Interpolate(qfu)
        for i in range(len(xvals)):
            for j in range(len(xvals)):
                evals[i,j] = gfu(self.mesh(t, xvals[i,j], yvals[i,j]))
        return evals
    
    