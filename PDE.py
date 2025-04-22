__all__ = [
    "PDE",
]

from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace, IntegrationRuleSpaceSurface
from ngsolve.webgui import Draw

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import abc


class PDE(abc.ABC):

    dim             = NotImplemented
    dirichlet_BCs   = NotImplemented
    initialB        = NotImplemented
    terminalB       = NotImplemented
    #nonLinearSolver = NotImplemented

    def __init__(
            self,
            order:    int,
            C:        int,
            beta:   float,
            printNum: int,
            maxIter:  int,
            nonLinearSolver: str
    ):
        

        if nonLinearSolver not in ["newton", "brent"]:
            raise Exception("Nonlinear solver must be 'newton' or 'brent'.")
        
        # if nonLinearSolver == "brent" and self.dim != 1:
        #     raise Exception("Brent method is current only available for dim = 1.")
            

        self.nonLinearSolver = nonLinearSolver    
        self.pdhgErr     = []
        self.stErr       = []
        self.terminalErr = []

        self.order, self.C, self.beta, self.printNum, self.maxIter = order, C, beta, printNum, maxIter
        self.iterList = list(range(0, self.maxIter, self.printNum))
        self.mesh = self.create_mesh()

        self.V     = H1(self.mesh, order = self.order)
        self.V0    = H1(self.mesh, order = self.order, dirichlet = self.dirichlet_BCs)
        self.fes   = self.V * self.V0 if self.dim == 1 else self.V * self.V0 * self.V0
        self.W     = IntegrationRuleSpace(self.mesh, order = self.order - 1)
        self.W_Vec = self.W ** self.dim
        self.M     = IntegrationRuleSpaceSurface(
                        self.mesh, 
                        order = self.order - 1, 
                        definedon = self.mesh.Boundaries(self.terminalB)
                    )

        self.rho = self.W.TrialFunction()

        self.dx  = dx(intrules  = self.W.GetIntegrationRules())
        self.dsr = ds(definedon = self.mesh.Boundaries(self.terminalB), intrules = self.M.GetIntegrationRules())
        self.dsl = ds(definedon = self.mesh.Boundaries(self.initialB),  intrules = self.M.GetIntegrationRules())

        self.gfu = GridFunction(self.fes)
        self.phih0    = GridFunction(self.V)

        self.sh       = GridFunction(self.W)
        self.sh0      = GridFunction(self.W)
        self.mh       = GridFunction(self.W_Vec)
        self.mh0      = GridFunction(self.W_Vec)
        self.nh       = GridFunction(self.W_Vec)
        self.nh0      = GridFunction(self.W_Vec)
        self.rhoh     = GridFunction(self.W)
        self.rhoh0    = GridFunction(self.W)
        self.rhohex   = GridFunction(self.W)
        self.rhoTh    = GridFunction(self.M)
        self.rhoTh0   = GridFunction(self.M)
        self.rhoThex  = GridFunction(self.M)
        
        
        self.drhoTh = self.rhoTh.vec.CreateVector()
        self.rhoT = self.M.TrialFunction()

        self.rhoh.Set (self.initial_guess())
        self.rhoTh.Set(
            self.initial_guess(), 
            definedon=self.mesh.Boundaries(self.terminalB)
        )
        self.rhohex.Interpolate (self.rho_exact())
        self.rhoThex.Interpolate(
            self.rho_exact(), 
            definedon=self.mesh.Boundaries(self.terminalB)
        )

        self.a = BilinearForm(self.fes)
        self.b = BilinearForm(self.W)
        self.c = BilinearForm(self.M)
        self.f = LinearForm(self.fes)

        self.phih, self.sigmah1 = self.gfu.components[:2]
        if self.dim == 2: self.sigmah2 = self.gfu.components[-1]

        self.sigmah01, self.sigmah02 = GridFunction(self.V0), GridFunction(self.V0)

        self.mh1  = self.mh.components[0]
        self.nh1  = self.nh.components[0]
        self.mh01 = self.mh0.components[0]
        self.nh01 = self.nh0.components[0]

        if self.dim == 2: self.mh2  = self.mh.components[1]
        if self.dim == 2: self.nh2  = self.nh.components[1]
        if self.dim == 2: self.mh02  = self.mh0.components[1]
        if self.dim == 2: self.nh02  = self.nh0.components[1]
        
        if self.dim == 2:
            (self.phi, self.sigma1, self.sigma2), (self.psi, self.tau1, self.tau2) = self.fes.TnT()
        else:
            (self.phi, self.sigma1), (self.psi, self.tau1) = self.fes.TnT()

        self.div_tau   = grad(self.tau1)[1]   + (grad(self.tau2)[2]   if self.dim == 2 else 0)
        self.div_sigma = grad(self.sigma1)[1] + (grad(self.sigma2)[2] if self.dim == 2 else 0)

        # phi terms
        ## space derivative + time derivative terms
        self.a += grad(self.phi) * grad(self.psi) * dx
        ## reaction term
        self.a += self.phi * self.psi * dx
        ## terminal term
        self.a += self.phi * self.psi * ds(self.terminalB)

        # sigma terms
        ## divergence term
        self.a += self.beta ** 2 * self.div_sigma * self.div_tau * dx
        ##
        self.a += self.sigma1 * self.tau1 * dx 
        if self.dim == 2: self.a += self.sigma2 * self.tau2 * dx 

        # Coupling terms
        self.a += self.beta * self.div_tau * grad(self.phi)[0] * dx

        self.a.Assemble()
        self.inva = self.a.mat.Inverse(self.fes.FreeDofs())

        if self.nonLinearSolver == 'newton':
            self.b += Variation(
                (   self.mh01 ** 2 / (1 + self.V1(self.rho))       
                +  (self.mh02 ** 2 / (1 + self.V1(self.rho))  if self.dim == 2 else 0 )
                +   self.nh01 ** 2 / (1 + self.V3(self.rho,))
                +  (self.nh02 ** 2 / (1 + self.V3(self.rho,)) if self.dim == 2 else 0 )
                +   self.sh0  ** 2 / (1 + self.V2(self.rho,self.C))
                +   self.beta ** 2 * self.dE(self.rho) ** 2 * self.V2(self.rho,self.C)
                + ( self.rho - self.rhoh0) ** 2
                ) *   self.dx
            )
        
        # self.c += Variation((self.beta * self.E(self.rhoT) + 0.5 * (self.rhoT-self.rhoTh0) ** 2) * self.dsr)

        # phi terms
        ## rho 
        self.f += -self.rhoh * grad(self.psi)[0] * self.dx
        ## m
        self.f += -self.mh1 * grad(self.psi)[1] * self.dx
        if self.dim == 2:
            self.f += -self.mh2 * grad(self.psi)[2] * self.dx
        ## s
        self.f += -self.sh * self.psi * self.dx
        ## terminal and initial
        self.f += self.rhoTh*self.psi*self.dsr - self.rho_exact()*self.psi*self.dsl

        # sigma term
        ## rho
        self.f += -self.beta * self.rhoh * self.div_tau * self.dx
        ## n
        self.f += -self.nh1 * self.tau1 * self.dx
        if self.dim == 2:
            self.f += -self.nh2 * self.tau2 * self.dx
    
    # Abstract methods --------------------------------------------------------
    @abc.abstractmethod
    def create_mesh(self):
        """Create the spatial finite element mesh."""
        raise NotImplementedError

    @abc.abstractmethod
    def rho_exact(self):
        raise NotImplementedError 
    
    @abc.abstractmethod
    def initial_guess(self):
        raise NotImplementedError 
    
    @abc.abstractmethod
    def V1(self):
        raise NotImplementedError 

    @abc.abstractmethod
    def V2(self):
        raise NotImplementedError 
    
    @abc.abstractmethod
    def V3(self):
        raise NotImplementedError

    @abc.abstractmethod
    def E(self):
        raise NotImplementedError 

    @abc.abstractmethod
    def dE(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def d2E(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def solveRho(self, m, s, n, rhoBar):
        raise NotImplementedError
    

# Solver ---------------------------------------------------------------
    def PDHGstep(self):
        # Step 1: solve for Lagrange multipliers
        self.f.Assemble()
        ## solve for (dphi, dsigma) and add
        self.gfu.vec.data += self.inva*self.f.vec

        # Step 2: extrapolation
        self.phih0.vec.data    += 2*self.phih.vec
        self.sigmah01.vec.data += 2*self.sigmah1.vec
        if self.dim == 2: self.sigmah02.vec.data += 2*self.sigmah2.vec

        div_sigmah0 = grad(self.sigmah01)[1] + (grad(self.sigmah02)[2] if self.dim == 2 else 0)

        # Step 3: Non-linear minimization
        ## Get bar variables
        self.rhoh0.Interpolate(self.rhoh + grad(self.phih0)[0] + self.beta * div_sigmah0)
        if self.C > 0: self.sh0.Interpolate (self.sh  + self.phih0)
        self.mh01.Interpolate(self.mh1 + grad(self.phih0)[1])
        if self.dim == 2: self.mh02.Interpolate(self.mh2 + grad(self.phih0)[2])
        self.nh01.Interpolate(self.nh1 + self.sigmah01)
        if self.dim == 2: self.nh02.Interpolate(self.nh2 + self.sigmah02)

        if self.nonLinearSolver == "newton":
            solvers.Newton(self.b, self.rhoh, printing=False) 
        else: 
            if self.dim == 1: 
                self.rhoh.vec.FV().NumPy()[:] = self.solveRho(self.mh01, self.sh0, self.nh01, self.rhoh0)
            else:
                self.rhoh.vec.FV().NumPy()[:] = self.solveRho((self.mh01, self.mh02), self.sh0, (self.nh01, self.nh02), self.rhoh0)

        ### Update mh, nh, sh
        self.mh1.Interpolate(self.V1(self.rhoh) * self.mh01 / ( 1 + self.V1(self.rhoh)))
        if self.dim == 2: self.mh2.Interpolate(self.V1(self.rhoh) * self.mh02 / (1 + self.V1(self.rhoh)) )
        self.nh1.Interpolate(self.V3(self.rhoh) * self.nh01 / (1 + self.V3(self.rhoh)))
        if self.dim == 2: self.nh2.Interpolate(self.V3(self.rhoh) * self.nh02 / (1 + self.V3(self.rhoh)))
        if self.C > 0: self.sh.Interpolate(self.V2(self.rhoh,self.C)*self.sh0 / (1 + self.V2(self.rhoh, self.C))) 

        ## solve for rhoT
        # self.rhoTh0.Interpolate(self.rhoTh - self.phih0, definedon=self.mesh.Boundaries(self.terminalB))
        # self.drhoTh.data = self.rhoTh.vec 
        # solvers.Newton(self.c, self.rhoTh, printing=False)
        # self.drhoTh.data -= self.rhoTh.vec
        
        self.drhoTh.data = self.rhoTh.vec 
        self.rhoTh.Interpolate(self.rhoTh - self.phih0, definedon=self.mesh.Boundaries(self.terminalB))
        self.rhoTh.vec.FV().NumPy()[:] += - self.beta * self.dE(self.drhoTh.FV().NumPy()[:])
        self.drhoTh.data -= self.rhoTh.vec

        # Step 2: Extrapolation
        self.phih0.vec.data    = -self.phih.vec
        self.sigmah01.vec.data = -self.sigmah1.vec
        if self.dim == 2: self.sigmah02.vec.data = -self.sigmah2.vec

    def solve(self):
        with TaskManager():
            for i in range(1,self.maxIter+1):
                self.PDHGstep()
                if ( i % self.printNum)==0:
                    self.err = max(max(self.drhoTh), -min(self.drhoTh))
                    self.pdhgErr.append(self.err)

                    self.err1 = sqrt(Integrate((self.rhoh-self.rhohex)**2*self.dx,self.mesh))
                    self.stErr.append(self.err1)

                    self.err2 = sqrt(Integrate((self.rhoTh-self.rhoThex)**2*self.dsr,self.mesh))
                    self.terminalErr.append(self.err2)

                    print('Iteration: %4i PDHG error: %.8e spacetime error: %.8e terminal error %.8e'%(
                    i, self.err, self.err1, self.err2), end="\r")
            print("\n")

   # Utilities ---------------------------------------------------------------
    def draw(self,rho):
        gfu = GridFunction(L2(self.mesh, order=self.order-1))
        gfu.Set(rho)
        Draw(gfu)

    
    def plotErr(self, axs, label, color):
        x_ = [i*self.printNum for i in range(1,len(self.pdhgErr)+1)]
        for ax in axs:
            ax.set_xlabel("PDHG iteration")
        axs[0].set_ylabel("PDHG Error")
        axs[1].set_ylabel(r"Solution Error")
        axs[0].semilogy(x_, self.pdhgErr, "-", label = label, color = color)
        axs[1].semilogy(x_, self.stErr, "--", label = label, color = color)
        axs[1].semilogy(x_, self.terminalErr, ":", label = label, color = color)

    def saveVTK(self, filename):
        gfu = GridFunction(L2(self.mesh, order=self.order-1))
        gfu2 = GridFunction(L2(self.mesh, order=self.order-1))
        gfu.Set(self.rhoh)
        gfu2.Set(self.rhohex)
        vtk = VTKOutput(
            self.mesh, 
            coefs = [gfu, gfu2], 
            names = ["rho", "rhoex"], 
            filename = filename, subdivision = 4)
        vtk.Do()

    def getTimeIntPoints(self):
        # Return a list of time quadrature points
        #if self.dim != 1: raise NotImplementedError
        X = IntegrationRuleSpaceSurface(self.mesh, order = self.order - 1, definedon = self.mesh.Boundaries("bottom"))
        int_points = X.GetIntegrationRules()[ET.SEGM].points
        points = set()
        for el in self.mesh.Elements():
            trafo = self.mesh.GetTrafo(el)
            for p in int_points:
                points.add(trafo(p[0],0).point[0])
        points = list(points)
        points.sort()
        return points 
        

    def getSpaceIntPoints(self):
        # returns an array of space quadrature points
        if self.dim != 1: raise NotImplementedError
            
        int_points = self.M.GetIntegrationRules()[ET.SEGM].points
        points = set()
        for el in self.mesh.Elements():
            trafo = self.mesh.GetTrafo(el)
            for p in int_points:
                points.add(trafo(0,p[0]).point[1])
        points = list(points)
        points.sort()
        return points
    
    def evaluateQuadratureFun(self, t, xvals, qfu):
        # Evaluates a quadrature function on (t, xvals) for scalar t and array xvals
        if self.dim != 1: raise NotImplementedError
        yvals = []
        gfu = GridFunction(L2(self.mesh, order=self.order-1))
        gfu.Interpolate(qfu)
        for p in xvals:
            yvals.append(gfu(self.mesh(t, p)))
        return yvals
    


