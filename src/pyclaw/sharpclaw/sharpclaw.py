r"""
Module containing SharpClaw solvers for PyClaw/PetClaw

#  File:        sharpclaw.py
#  Created:     2010-03-20
#  Author:      David Ketcheson
"""
# Solver superclass
from pyclaw.solver import Solver, CFLError

# Reconstructor
try:
    # load c-based WENO reconstructor (PyWENO)
    from pyclaw.limiters import reconstruct as recon
except ImportError:
    # load old WENO5 reconstructor
    from pyclaw.limiters import recon


def before_step(solver,solution):
    r"""
    Dummy routine called before each step
    
    Replace this routine if you want to do something before each time step.
    """
    pass

class SharpClawSolver(Solver):
    r"""
    Superclass for all SharpClawND solvers.

    Implements Runge-Kutta time stepping and the basic form of a 
    semi-discrete step (the dq() function).  If another method-of-lines
    solver is implemented in the future, it should be based on this class,
    which then ought to be renamed to something like "MOLSolver".

    .. attribute:: before_step
    
        Function called before each time step is taken.
        The required signature for this function is:
        
        def before_step(solver,solution)

    .. attribute:: lim_type

        Limiter(s) to be used.
        0: No limiting.
        1: TVD reconstruction.
        2: WENO reconstruction.
        ``Default = 2``

    .. attribute:: weno_order

        Order of the WENO reconstruction. From 1st to 17th order (PyWENO)
        ``Default = 5``

    .. attribute:: time_integrator

        Time integrator to be used.
        Euler: forward Euler method.
        SSP33: 3-stages, 3rd-order SSP Runge-Kutta method.
        SSP104: 10-stages, 4th-order SSP Runge-Kutta method.
        ``Default = 'SSP104'``

    .. attribute:: char_decomp

        Type of WENO reconstruction.
        0: conservative variables WENO reconstruction (standard).
        1: characteristic-wise WENO reconstruction.
        2: transmission-based WENO reconstruction.
        ``Default = 0``

    .. attribute:: tfluct_solver

        Whether a total fluctuation solver have to be used. If True the function
        that calculates the total fluctuation must be provided.
        ``Default = False``

    .. attribute:: aux_time_dep

        Whether the auxiliary array is time dependent.
        ``Default = False``
    
    .. attribute:: kernel_language

        Specifies whether to use wrapped Fortran routines ('Fortran')
        or pure Python ('Python').  
        ``Default = 'Fortran'``.

    .. attribute:: num_ghost

        Number of ghost cells.
        ``Default = 3``

    .. attribute:: fwave
    
        Whether to split the flux jump (rather than the jump in Q) into waves; 
        requires that the Riemann solver performs the splitting.  
        ``Default = False``

    .. attribute:: cfl_desired

        Desired CFL number.
        ``Default = 2.45``

    .. attribute:: cfl_max

        Maximum CFL number.
        ``Default = 2.50``

    .. attribute:: dq_src

        Whether a source term is present. If it is present the function that 
        computes its contribution must be provided.
        ``Default = None``
    """
    
    # ========================================================================
    #   Initialization routines
    # ========================================================================
    def __init__(self):
        r"""
        Set default options for SharpClawSolvers and call the super's __init__().
        """
        self.limiters = [1]
        self.before_step = before_step
        self.lim_type = 2
        self.weno_order = 5
        self.time_integrator = 'SSP104'
        self.char_decomp = 0
        self.tfluct_solver = False
        self.aux_time_dep = False
        self.kernel_language = 'Fortran'
        self.num_ghost = 3
        self.fwave = False
        self.cfl_desired = 2.45
        self.cfl_max = 2.5
        self.dq_src = None
        self._mthlim = self.limiters
        self._method = None
        self._rk_stages = None
        self.upwind = 1
        
        # Call general initialization function
        super(SharpClawSolver,self).__init__()
        
    # ========== Time stepping routines ======================================
    def step(self,solution):
        """Evolve q over one time step.

        Take on Runge-Kutta time step using the method specified by
        self..time_integrator.  Currently implemented methods:

        'Euler'  : 1st-order Forward Euler integration
        'SSP33'  : 3rd-order strong stability preserving method of Shu & Osher
        'SSP104' : 4th-order strong stability preserving method Ketcheson
        """
        state = solution.states[0]

        self.before_step(self,solution)

        try:
            if self.time_integrator=='Euler':
                deltaq=self.dq(state)
                state.q+=deltaq

            elif self.time_integrator=='DWSSP22':
                deltaq = self.dq(state)
                # Downwind
                self.upwind = 0
                deltaq_dw = self.dq(state)
                # Upwind   
                self.upwind = 1
                self._rk_stages[0].q = state.q + 0.822875655532364 * deltaq
                self._rk_stages[0].t = state.t + 0.822875655532364 * self.dt
                deltaq = self.dq(self._rk_stages[0])
                state.q = 0.261583187659478 * state.q + 0.738416812340522 * self._rk_stages[0].q \
                        - 0.215250437021539 * deltaq_dw \
                        + 0.607625218510713 * deltaq


            elif self.time_integrator=='SSP33':
                deltaq=self.dq(state)
                self._rk_stages[0].q=state.q+deltaq
                self._rk_stages[0].t =state.t+self.dt
                deltaq=self.dq(self._rk_stages[0])
                self._rk_stages[0].q= 0.75*state.q + 0.25*(self._rk_stages[0].q+deltaq)
                self._rk_stages[0].t = state.t+0.5*self.dt
                deltaq=self.dq(self._rk_stages[0])
                state.q = 1./3.*state.q + 2./3.*(self._rk_stages[0].q+deltaq)


            elif self.time_integrator=='SSP43':
                deltaq=self.dq(state)
                self._rk_stages[0].q = state.q+0.5*deltaq
                self._rk_stages[0].t = state.t+0.5*self.dt
                deltaq = self.dq(self._rk_stages[0])
                self._rk_stages[0].q = self._rk_stages[0].q + 0.5*deltaq
                self._rk_stages[0].t = state.t+self.dt
                deltaq = self.dq(self._rk_stages[0])
                self._rk_stages[0].q = 2./3.*state.q + 1./3.*(self._rk_stages[0].q+0.5*deltaq)
                self._rk_stages[0].t = state.t+0.5*self.dt
                deltaq = self.dq(self._rk_stages[0])
                state.q = self._rk_stages[0].q + 0.5*deltaq



            elif self.time_integrator=='SSP104':
                s1=self._rk_stages[0]
                s2=self._rk_stages[1]
                s1.q = state.q.copy()

                deltaq=self.dq(state)
                s1.q = state.q + deltaq/6.
                s1.t = state.t + self.dt/6.

                for i in xrange(4):
                    deltaq=self.dq(s1)
                    s1.q=s1.q + deltaq/6.
                    s1.t =s1.t + self.dt/6.

                s2.q = state.q/25. + 9./25 * s1.q
                s1.q = 15. * s2.q - 5. * s1.q
                s1.t = state.t + self.dt/3.

                for i in xrange(4):
                    deltaq=self.dq(s1)
                    s1.q=s1.q + deltaq/6.
                    s1.t =s1.t + self.dt/6.

                deltaq = self.dq(s1)
                state.q = s2.q + 0.6 * s1.q + 0.1 * deltaq

            elif self.time_integrator=='DWSSP105':
                deltaq = self.dq(state)
                self._rk_stages[0].q = state.q + 0.173586107937995 * deltaq
                c0 = 0.173586107937995
                self._rk_stages[0].t = state.t + c0 * self.dt


                deltaq1=self.dq(self._rk_stages[0])
                self._rk_stages[1].q = 0.258168167463650 * state.q \
                                     + 0.741831832536350 * self._rk_stages[0].q \
                                     + 0.218485490268790 * deltaq1
                c1 = 0.741831832536350 * c0 + 0.218485490268790                   
                self._rk_stages[1].t = state.t + c1 * self.dt


                deltaq2 = self.dq(self._rk_stages[1])
                self._rk_stages[2].q = 0.037493531856076 * self._rk_stages[0].q \
                                     + 0.962506468143924 * self._rk_stages[1].q \
                                     + 0.011042654588541 * deltaq1 \
                                     + 0.283478934653295 * deltaq2
                c2 = 0.037493531856076 * c0 + 0.962506468143924 * c1 + 0.011042654588541 \
                   + 0.283478934653295
                self._rk_stages[2].t = state.t +  c2 * self.dt


                deltaq3 = self.dq(self._rk_stages[2])
                # Downwind
                self.upwind = 0
                deltaq3_dw = self.dq(self._rk_stages[2])
                # Upwind
                self.upwind = 1
                self._rk_stages[3].q = 0.595955269449077 * state.q \
                                     + 0.404044730550923 * self._rk_stages[1].q \
                                     + 0.118999896166647 * deltaq2
                c3 = 0.404044730550923 * c1 + 0.118999896166647                   
                self._rk_stages[3].t = state.t +  c3 * self.dt


                deltaq4 = self.dq(self._rk_stages[3])
                self._rk_stages[4].q = 0.331848124368345 * state.q \
                                     + 0.008466192609453 * self._rk_stages[2].q \
                                     + 0.659685683022202 * self._rk_stages[3].q \
                                     + 0.025030881091201 * deltaq \
                                     - 0.002493476502164 * deltaq3_dw \
                                     + 0.194291675763785 * deltaq4
                c4 = 0.008466192609453 * c2 + 0.659685683022202 * c3 + 0.025030881091201 \
                   + 0.002493476502164 + 0.194291675763785
                self._rk_stages[4].t = state.t +  c4 * self.dt

                deltaq5 = self.dq(self._rk_stages[4])
                self._rk_stages[5].q = 0.086976414344414 * state.q \
                                     + 0.913023585655586 * self._rk_stages[4].q \
                                     + 0.268905157462563 * deltaq5
                c5 = 0.913023585655586 * c4 + 0.268905157462563                   
                self._rk_stages[5].t = state.t + c5 * self.dt


                deltaq6 = self.dq(self._rk_stages[5])
                self._rk_stages[6].q = 0.075863700003186 * state.q \
                                     + 0.267513039663395 * self._rk_stages[1].q \
                                     + 0.656623260333419 * self._rk_stages[5].q \
                                     + 0.066115378914543 * deltaq2 \
                                     + 0.193389726166555 * deltaq6
                c6 = 0.267513039663395 * c1 + 0.656623260333419 * c5 + 0.066115378914543 \
                   + 0.193389726166555
                self._rk_stages[6].t = state.t +  c6 * self.dt


                deltaq7 = self.dq(self._rk_stages[6])
                self._rk_stages[7].q = 0.005212058095597 * state.q \
                                     + 0.407430107306541 * self._rk_stages[2].q \
                                     + 0.587357834597862 * self._rk_stages[6].q \
                                     - 0.119996962708895 * deltaq3_dw \
                                     + 0.172989562899406 * deltaq7
                c7 = 0.407430107306541 * c2 + 0.587357834597862 * c6 + 0.119996962708895 \
                   + 0.172989562899406
                self._rk_stages[7].t = state.t + c7 * self.dt


                deltaq8 = self.dq(self._rk_stages[7])
                self._rk_stages[8].q = 0.122832051947995 * state.q \
                                     + 0.877167948052005 * self._rk_stages[7].q \
                                     + 0.000000000000035 * deltaq \
                                     + 0.258344898092277 * deltaq8
                c8 = 0.877167948052005 * c7 + 0.000000000000035 + 0.258344898092277                    
                self._rk_stages[8].t = state.t + c8 * self.dt


                deltaq9 = self.dq(self._rk_stages[8])
                state.q = 0.075346276482673 * state.q \
                        + 0.000425904246091 * self._rk_stages[0].q \
                        + 0.064038648145995 * self._rk_stages[4].q \
                        + 0.354077936287492 * self._rk_stages[5].q \
                        + 0.506111234837749 * self._rk_stages[8].q \
                        + 0.016982542367506 * deltaq \
                        + 0.018860764424857 * deltaq5 \
                        + 0.098896719553054 * deltaq6 \
                        + 0.149060685217562 * deltaq9
                

            else:
                raise Exception('Unrecognized time integrator')
        except CFLError:
            return False


    def set_mthlim(self):
        self._mthlim = self.limiters
        if not isinstance(self.limiters,list): self._mthlim=[self._mthlim]
        if len(self._mthlim)==1: self._mthlim = self._mthlim * self.num_waves
        if len(self._mthlim)!=self.num_waves:
            raise Exception('Length of solver.limiters is not equal to 1 or to solver.num_waves')
 
       
    def dq(self,state):
        """
        Evaluate dq/dt * (delta t)
        """

        deltaq = self.dq_hyperbolic(state)

        # Check here if we violated the CFL condition, if we did, return 
        # immediately to evolve_to_time and let it deal with picking a new
        # dt
        if self.cfl.get_cached_max() > self.cfl_max:
            raise CFLError('cfl_max exceeded')

        if self.dq_src is not None:
            deltaq+=self.dq_src(self,state,self.dt)

        return deltaq

    def dq_hyperbolic(self,state):
        raise NotImplementedError('You must subclass SharpClawSolver.')

         
    def dqdt(self,state):
        """
        Evaluate dq/dt.  This routine is used for implicit time stepping.
        """

        self.dt = 1
        deltaq = self.dq_hyperbolic(state)

        if self.dq_src is not None:
            deltaq+=self.dq_src(self,state,self.dt)

        return deltaq.flatten('f')


    def set_fortran_parameters(self,state,clawparams,workspace,reconstruct):
        """
        Set parameters for Fortran modules used by SharpClaw.
        The modules should be imported and passed as arguments to this function.

        """
        grid = state.grid
        clawparams.num_dim       = grid.num_dim
        clawparams.lim_type      = self.lim_type
        clawparams.weno_order    = self.weno_order
        clawparams.char_decomp   = self.char_decomp
        clawparams.tfluct_solver = self.tfluct_solver
        clawparams.fwave         = self.fwave
        clawparams.index_capa         = state.index_capa+1

        clawparams.num_waves     = self.num_waves
        clawparams.alloc_clawparams()
        for idim in range(grid.num_dim):
            clawparams.xlower[idim]=grid.dimensions[idim].lower
            clawparams.xupper[idim]=grid.dimensions[idim].upper
        clawparams.dx       =grid.delta
        clawparams.mthlim   =self._mthlim

        maxnx = max(grid.num_cells)+2*self.num_ghost
        workspace.alloc_workspace(maxnx,self.num_ghost,state.num_eqn,self.num_waves,self.char_decomp)
        reconstruct.alloc_recon_workspace(maxnx,self.num_ghost,state.num_eqn,self.num_waves,
                                            clawparams.lim_type,clawparams.char_decomp)

    def allocate_rk_stages(self,solution):
        r"""
        Instantiate State objects for Runge--Kutta stages.

        This routine is only used by method-of-lines solvers (SharpClaw),
        not by the Classic solvers.  It allocates additional State objects
        to store the intermediate stages used by Runge--Kutta time integrators.

        If we create a MethodOfLinesSolver subclass, this should be moved there.
        """
        if self.time_integrator   == 'Euler':  nregisters=1
        elif self.time_integrator == 'DWSSP22':  nregisters=2    
        elif self.time_integrator == 'SSP33':  nregisters=2
        elif self.time_integrator == 'SSP43':  nregisters=2
        elif self.time_integrator == 'SSP104': nregisters=3
        elif self.time_integrator == 'DWSSP105': nregisters=10

        

        state = solution.states[0]
        # use the same class constructor as the solution for the Runge Kutta stages
        State = type(state)
        self._rk_stages = []
        for i in xrange(nregisters-1):
            #Maybe should use State.copy() here?
            self._rk_stages.append(State(state.patch,state.num_eqn,state.num_aux))
            self._rk_stages[-1].problem_data       = state.problem_data
            self._rk_stages[-1].set_num_ghost(self.num_ghost)
            self._rk_stages[-1].t                = state.t
            if state.num_aux > 0:
                self._rk_stages[-1].aux              = state.aux




# ========================================================================
class SharpClawSolver1D(SharpClawSolver):
# ========================================================================
    """
    SharpClaw solver for one-dimensional problems.
    
    Used to solve 1D hyperbolic systems using the SharpClaw algorithms,
    which are based on WENO reconstruction and Runge-Kutta time stepping.
    """
    def __init__(self):
        r"""
        See :class:`SharpClawSolver1D` for more info.
        """   
        self.num_dim = 1
        super(SharpClawSolver1D,self).__init__()


    def setup(self,solution):
        """
        Allocate RK stage arrays and fortran routine work arrays.
        """
        self.num_ghost = (self.weno_order+1)/2

        # This is a hack to deal with the fact that petsc4py
        # doesn't allow us to change the stencil_width (num_ghost)
        state = solution.state
        state.set_num_ghost(self.num_ghost)
        # End hack

        self.allocate_rk_stages(solution)
        self.set_mthlim()
 
        state = solution.states[0]

        if self.kernel_language=='Fortran':
            from sharpclaw1 import clawparams, workspace, reconstruct
            import sharpclaw1
            state.set_cparam(sharpclaw1)
            self.set_fortran_parameters(state,clawparams,workspace,reconstruct)

        self.allocate_bc_arrays(state)

    def teardown(self):
        r"""
        Deallocate F90 module arrays.
        Also delete Fortran objects, which otherwise tend to persist in Python sessions.
        """
        if self.kernel_language=='Fortran':
            from sharpclaw1 import clawparams, workspace, reconstruct
            clawparams.dealloc_clawparams()
            workspace.dealloc_workspace(self.char_decomp)
            reconstruct.dealloc_recon_workspace(clawparams.lim_type,clawparams.char_decomp)
            import sharpclaw1
            print 'deleting sharpclaw1 object'
            del sharpclaw1, clawparams, workspace, reconstruct


    def dq_hyperbolic(self,state):
        r"""
        Compute dq/dt * (delta t) for the hyperbolic hyperbolic system.

        Note that the capa array, if present, should be located in the aux
        variable.

        Indexing works like this (here num_ghost=2 as an example)::

         0     1     2     3     4     mx+num_ghost-2     mx+num_ghost      mx+num_ghost+2
                     |                        mx+num_ghost-1 |  mx+num_ghost+1
         |     |     |     |     |   ...   |     |     |     |     |
            0     1  |  2     3            mx+num_ghost-2    |mx+num_ghost       
                                                  mx+num_ghost-1   mx+num_ghost+1

        The top indices represent the values that are located on the grid
        cell boundaries such as waves, s and other Riemann problem values, 
        the bottom for the cell centered values such as q.  In particular
        the ith grid cell boundary has the following related information::

                          i-1         i         i+1
                           |          |          |
                           |   i-1    |     i    |
                           |          |          |

        Again, grid cell boundary quantities are at the top, cell centered
        values are in the cell.

        """
    
        import numpy as np

        self.apply_q_bcs(state)
        q = self.qbc 

        grid = state.grid
        mx = grid.num_cells[0]

        ixy=1

        if self.kernel_language=='Fortran':
            from sharpclaw1 import flux1
            dq,cfl=flux1(q,self.auxbc,self.dt,state.t,ixy,mx,self.num_ghost,mx,self.upwind)

        elif self.kernel_language=='Python':

            dtdx = np.zeros( (mx+2*self.num_ghost) ,order='F')
            dq   = np.zeros( (state.num_eqn,mx+2*self.num_ghost) ,order='F')

            # Find local value for dt/dx
            if state.index_capa>=0:
                dtdx = self.dt / (grid.delta[0] * state.aux[state.index_capa,:])
            else:
                dtdx += self.dt/grid.delta[0]
 
            aux=self.auxbc
            if aux.shape[0]>0:
                aux_l=aux[:,:-1]
                aux_r=aux[:,1: ]
            else:
                aux_l = None
                aux_r = None

            #Reconstruct (wave reconstruction uses a Riemann solve)
            if self.lim_type==-1: #1st-order Godunov
                ql=q; qr=q
            elif self.lim_type==0: #Unlimited reconstruction
                raise NotImplementedError('Unlimited reconstruction not implemented')
            elif self.lim_type==1: #TVD Reconstruction
                raise NotImplementedError('TVD reconstruction not implemented')
            elif self.lim_type==2: #WENO Reconstruction
                if self.char_decomp==0: #No characteristic decomposition
                    ql,qr=recon.weno(5,q)
                elif self.char_decomp==1: #Wave-based reconstruction
                    q_l=q[:,:-1]
                    q_r=q[:,1: ]
                    wave,s,amdq,apdq = self.rp(q_l,q_r,aux_l,aux_r,state.problem_data)
                    ql,qr=recon.weno5_wave(q,wave,s)
                elif self.char_decomp==2: #Characteristic-wise reconstruction
                    raise NotImplementedError

            # Solve Riemann problem at each interface
            q_l=qr[:,:-1]
            q_r=ql[:,1: ]
            wave,s,amdq,apdq = self.rp(q_l,q_r,aux_l,aux_r,state.problem_data)

            # Loop limits for local portion of grid
            # THIS WON'T WORK IN PARALLEL!
            LL = self.num_ghost - 1
            UL = grid.num_cells[0] + self.num_ghost + 1

            # Compute maximum wave speed
            cfl = 0.0
            for mw in xrange(self.num_waves):
                smax1 = np.max( dtdx[LL  :UL]  *s[mw,LL-1:UL-1])
                smax2 = np.max(-dtdx[LL-1:UL-1]*s[mw,LL-1:UL-1])
                cfl = max(cfl,smax1,smax2)

            #Find total fluctuation within each cell
            wave,s,amdq2,apdq2 = self.rp(ql,qr,aux,aux,state.problem_data)

            # Compute dq
            for m in xrange(state.num_eqn):
                dq[m,LL:UL] = -dtdx[LL:UL]*(amdq[m,LL:UL] + apdq[m,LL-1:UL-1] \
                                + apdq2[m,LL:UL] + amdq2[m,LL:UL])

        else: raise Exception('Unrecognized value of solver.kernel_language.')

        self.cfl.update_global_max(cfl)
        return dq[:,self.num_ghost:-self.num_ghost]
    

# ========================================================================
class SharpClawSolver2D(SharpClawSolver):
# ========================================================================
    """SharpClaw evolution routine in 2D
    
    This class represents the 2D SharpClaw solver.  Note that there are 
    routines here for interfacing with the fortran time stepping routines only.
    """
    def __init__(self):
        r"""
        Create 2D SharpClaw solver
        
        See :class:`SharpClawSolver2D` for more info.
        """   
        self.num_dim = 2

        super(SharpClawSolver2D,self).__init__()


    def setup(self,solution):
        """
        Allocate RK stage arrays and fortran routine work arrays.
        """
        self.num_ghost = (self.weno_order+1)/2

        # This is a hack to deal with the fact that petsc4py
        # doesn't allow us to change the stencil_width (num_ghost)
        state = solution.state
        state.set_num_ghost(self.num_ghost)
        # End hack

        self.allocate_rk_stages(solution)
        self.set_mthlim()

        state = solution.states[0]
 
        if self.kernel_language=='Fortran':
            from sharpclaw2 import clawparams, workspace, reconstruct
            import sharpclaw2
            state.set_cparam(sharpclaw2)
            self.set_fortran_parameters(state,clawparams,workspace,reconstruct)

        self.allocate_bc_arrays(state)

    def teardown(self):
        r"""
        Deallocate F90 module arrays.
        Also delete Fortran objects, which otherwise tend to persist in Python sessions.
        """
        if self.kernel_language=='Fortran':
            from sharpclaw2 import clawparams, workspace, reconstruct
            workspace.dealloc_workspace(self.char_decomp)
            reconstruct.dealloc_recon_workspace(clawparams.lim_type,clawparams.char_decomp)
            clawparams.dealloc_clawparams()
            import sharpclaw2
            del sharpclaw2


    def dq_hyperbolic(self,state):
        """Compute dq/dt * (delta t) for the hyperbolic hyperbolic system

        Note that the capa array, if present, should be located in the aux
        variable.

        Indexing works like this (here num_ghost=2 as an example)::

         0     1     2     3     4     mx+num_ghost-2     mx+num_ghost      mx+num_ghost+2
                     |                        mx+num_ghost-1 |  mx+num_ghost+1
         |     |     |     |     |   ...   |     |     |     |     |
            0     1  |  2     3            mx+num_ghost-2    |mx+num_ghost       
                                                  mx+num_ghost-1   mx+num_ghost+1

        The top indices represent the values that are located on the grid
        cell boundaries such as waves, s and other Riemann problem values, 
        the bottom for the cell centered values such as q.  In particular
        the ith grid cell boundary has the following related information::

                          i-1         i         i+1
                           |          |          |
                           |   i-1    |     i    |
                           |          |          |

        Again, grid cell boundary quantities are at the top, cell centered
        values are in the cell.

        """
        self.apply_q_bcs(state)
        q = self.qbc 

        grid = state.grid

        num_ghost=self.num_ghost
        mx=grid.num_cells[0]
        my=grid.num_cells[1]
        maxm = max(mx,my)

        if self.kernel_language=='Fortran':
            from sharpclaw2 import flux2
            dq,cfl=flux2(q,self.auxbc,self.dt,state.t,num_ghost,maxm,mx,my)

        else: raise Exception('Only Fortran kernels are supported in 2D.')

        self.cfl.update_global_max(cfl)
        return dq[:,num_ghost:-num_ghost,num_ghost:-num_ghost]
