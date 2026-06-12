
import numpy as np
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Params:
    # Geometry / discretization
    L: float = 50.0 #mm
    dx: float = 0.1
    n_rows: int = 2
    row_width: int = 4 #mm
    resin_gap: float =  0.1 #mm

    # Loading
    eps0: float = 0.01  # applied macroscopic strain (displacement control)

    # Tow properties (axial)
    E_tow: float = 140e9
    D_f = 4
    t_f = .1
    A_tow: float = D_f * t_f # cross-sectional area of a single tow


    # Matrix/interface (shear transfer)
    G_m: float = 1.0e9
    t_eff: float = 1.0e-4
    b_eff: float = 1.0e-3  # effective width for patch area

    # Phase 1: hard-coded overlap region (in x)
    overlap_x0: float = .0*L
    overlap_x1: float = 1.0 * L


#======================================================================================
# indexing help
#=======================================================================================

def build_mesh(L: float, dx: float) -> np.ndarray:
    # Ensure endpoint included
    n = int(np.round(L / dx))
    #x = np.linspace(0.0, L, n + 1)
    x=  np.arange(n+1, dtype=float) * dx
    x[-1] = L
    return x


def dof_index(row: int, node: int, n_nodes: int) -> int:
    # global DOF index for 1 DOF per node per row
    return row * n_nodes + node
#======================================================================================
#element assembly
#=====================================================================================


print( build_mesh(Params.L, Params.dx)  )
print( dof_index(1, 3, 10) )
def add_2node_spring(K, a: int, b: int, k: float)-> None:
    K[a, a] += k
    K[b, b] += k
    K[a, b] -= k
    K[b, a] -= k


def k_axial(E_tow: float, A_tow: float, dx: float) -> float:
    return (E_tow * A_tow) / dx

def k_shear(G_m: float, b_eff: float, t_eff: float, dx: float) -> float:
    # shear spring stiffness per unit length
    k = G_m * b_eff * dx / t_eff
    return k

#======================================================================================
#boundary conditions
#======================================================================================
def apply_dirichlet_elimination(K, f, prescribed: dict[int, float]):
    """
    Dirichlet elimination:
      - prescribed: {dof_index: value}
    Returns:
      - K_ff, f_eff, free_dofs, u_full (with prescribed entries filled)
    """
    n = f.shape[0] # we dont actually import force displacemnts just there in form ku = f 
    u_full = np.zeros(n)


    # fill prescribed dof in u_full
    for dof_index, value in prescribed.items():
        u_full[dof_index] = value

    prescribed_dofs = np.array(sorted(prescribed.keys())) #converts dof indices and sorts them in array
    is_prescribed = np.zeros(n, dtype=bool) # creates boolean array of size n
    is_prescribed[prescribed_dofs] = True # where no dof is prescribed, set to True
    free_dofs = np.where(~is_prescribed)[0] #flips booleans and sptis out indicies of free dofs
    # get submatricies k_ff = k[free, free] kfp = K free, prescribed
    K_ff = K[free_dofs][:,free_dofs] #select row and col corresponding to free dofs
    K_gp = K[free_dofs][:, prescribed_dofs] # select row corresponding to free dofs and col corresponding to prescribed dofs


    #rhs f_f = K_fp*u+p
    u_p = u_full[prescribed_dofs] # get prescribed displacements
    f_f = f[free_dofs] # global forces corresponding with global free dofs #typicall 0 
    f_eff = f_f - K_gp @ u_p # effective rhs # equivalent load resulting from displacement
    return K_ff, f_eff, free_dofs, u_full

#======================================================================================
#build phase 1 model
#=====================================================================================
def assemble_phase1(params: Params, x:np.ndarray):
    n_nodes = len(x)
    n_dofs = params.n_rows * n_nodes


    #list of list sparse matrix
    K = lil_matrix((n_dofs, n_dofs))
    f = np.zeros(n_dofs, dtype= float)


    #axial springs in both rows (continuous to start) 
    kax = k_axial(params.E_tow, params.A_tow, params.dx)
    for row in range(params.n_rows):
        for i in range(n_nodes-1):
            a = dof_index(row, i, n_nodes)
            b = dof_index(row, i+1, n_nodes)
            add_2node_spring(K, a, b, kax)

    #shear springs in ovlerap resgion
    for i, xi in enumerate(x):
        ksh = k_shear(params.G_m, params.b_eff, params.t_eff, params.dx)
        if params.overlap_x0<= xi <= params.overlap_x1:
            # add shear spring between row 0 and row 1 at node i between overlap length

            a = dof_index(0, i, n_nodes)
            b = dof_index(1, i, n_nodes)
            add_2node_spring(K, a, b, ksh)

    return K.tocsr(), f # global stiffness matrix and global applied force vector correspondign for ku=f
        
def build_dirichlet_bcs(params: Params, x:np.ndarray) -> dict[int, float]:
    '''u(left end) = 0, u(right end) = eps0*L '''
    n_nodes = len(x)
    u_right= params.eps0 * params.L
    left_i = 0 
    right_i = n_nodes-1
    prescribed = {}
    for row in range(params.n_rows):
        prescribed[dof_index(row, left_i, n_nodes)] = 0.0
        prescribed[dof_index(row, right_i, n_nodes)] = u_right
    return prescribed


#postprocessing
def postprocess_placeholders(params: Params, x: np.ndarray, u: np.ndarray):
    n_nodes = len(x)

    # Split rows (1 DOF per node)
    u0 = u[0 * n_nodes : 1 * n_nodes]
    u1 = u[1 * n_nodes : 2 * n_nodes]

    # Element midpoints
    x_mid = 0.5 * (x[:-1] + x[1:])

    # Element strains (piecewise constant per element)
    eps0 = np.diff(u0) / params.dx
    eps1 = np.diff(u1) / params.dx

    # Element stresses
    sig0 = params.E_tow * eps0/1e9  # GPa
    sig1 = params.E_tow * eps1/1e9  # GPa

    # Sanity check still OK
    max_diff = np.max(np.abs(u0 - u1))
    print(f"Max |u_row0 - u_row1| = {max_diff:.3e}")

    # Plot
    plt.figure()
    plt.ylim(-10,10)
    plt.plot(x_mid, sig0, label="Row 0")
    plt.plot(x_mid, sig1, label="Row 1")
    plt.xlabel("x")
    plt.ylabel("Axial stress in tow")
    plt.title("Phase 1 axial stress distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    params = Params()
    x = build_mesh(params.L, params.dx)

    K, f = assemble_phase1(params, x)
    prescribed = build_dirichlet_bcs(params, x)

    K_ff, f_eff, free_dofs, u_full = apply_dirichlet_elimination(K, f, prescribed)

    # Solve reduced system
    u_free = spsolve(K_ff, f_eff)

    # Reconstruct full displacement vector
    u_full[free_dofs] = u_free

    postprocess_placeholders(params, x, u_full)


if __name__ == "__main__":
    main()
