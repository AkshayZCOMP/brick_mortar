
import numpy as np
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Params:
    # Geometry / discretization
    
    L_fiber: float = 50
    n_fibers: int = 4
    dx: float = 0.1
    n_rows: int = 5
    row_width: int = 4 #mm
    resin_gap: float =  0.1 #mm
    L: float = (L_fiber+resin_gap)*n_fibers #mm
    # Loading
    eps0: float = 0.01  # applied macroscopic strain (displacement control)

    # Tow properties (axial)
    E_tow: float = 140e9
    D_f = 4
    t_f = .1
    A_tow: float = D_f * t_f # cross-sectional area of a single tow

    row_break = [L*.49, L*.51]
    # Matrix/interface (shear transfer)
    G_m: float = 1.0e9
    E_m: float = 3.0e9
    t_eff: float = .10 #mm
    b_eff: float = 1.0 #mm  # effective width for patch area


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
    ksh = k_shear(params.G_m, params.b_eff, params.t_eff, params.dx)
    kbridge = params.E_m*params.A_tow/(params.dx) # spring constant for shear spring in break region
    elems = [] 

    for row in range(params.n_rows):
        if row == 0:
            for i in range(n_nodes-1):
                #introduce break in row 0 
                xmid = (x[i]+x[i+1])/2
                if xmid< params.row_break[0] or xmid >params.row_break[1]:
                    a = dof_index(row, i, n_nodes)
                    b = dof_index(row, i+1, n_nodes)
                    add_2node_spring(K, a, b, kax)
                    elems.append({"etype": "tow_axial", "row": row, "i": i, "a": a, "b": b, "k": kax, "A": params.A_tow})
                if params.row_break[0]<= xmid <= params.row_break[1]:
                    a = dof_index(row, i, n_nodes)
                    b = dof_index(row, i+1, n_nodes)
                    add_2node_spring(K, a, b, kbridge)
                    elems.append({"etype": "tow_bridge", "row": row, "i": i, "a": a, "b": b, "k": kbridge, "A": params.A_tow})
        else:
            for i in range(n_nodes-1):
                a = dof_index(row, i, n_nodes)
                b = dof_index(row, i+1, n_nodes)
                add_2node_spring(K, a, b, kax)
                elems.append({"etype": "tow_axial", "row": row, "i": i, "a": a, "b": b, "k": kax, "A": params.A_tow})
                      
        '''
        for i in range(n_nodes-1):
            a = dof_index(row, i, n_nodes)
            b = dof_index(row, i+1, n_nodes)
            add_2node_spring(K, a, b, kax)
            '''

    #shear springs in ovlerap resgion
    for r in range(params.n_rows - 1):
        for i, xi in enumerate(x):
                a = dof_index(r, i, n_nodes)
                b = dof_index(r+1, i, n_nodes)
                add_2node_spring(K, a, b, ksh)
                elems.append({"etype": "shear", "i": i, "a": a, "b": b,  "k": ksh})

    return K.tocsr(), f, elems # global stiffness matrix and global applied force vector correspondign for ku=f
        
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
def postprocess_by_element_type(params: Params, x: np.ndarray, u: np.ndarray, elems: list[dict]):
    n_nodes = len(x)
    x_mid = 0.5 * (x[:-1] + x[1:])

    # Axial stress arrays (Pa). Start as NaN so missing elements stay blank.
    sig_tow = np.full((params.n_rows, n_nodes - 1), np.nan, dtype=float)
    sig_bridge = np.full((params.n_rows, n_nodes - 1), np.nan, dtype=float)

    # Inter-row shear force / traction at nodes (shear elems are attached at nodes in your model)
    F_shear = np.full(n_nodes, np.nan, dtype=float)
    tau_shear = np.full(n_nodes, np.nan, dtype=float)  # Pa

    # Populate from element list
    for e in elems:
        etype = e["etype"]
        a = e["a"]
        b = e["b"]
        k = e["k"]
        du = u[b] - u[a]
        F = k * du

        if etype == "tow_axial":
            row = e["row"]
            i = e["i"]
            A = e["A"]
            sig_tow[row, i] = F / A

        elif etype == "tow_bridge":
            row = e["row"]
            i = e["i"]
            A = e["A"]  # ideally A_bridge; start with A_tow if that's your assumption
            sig_bridge[row, i] = F / A

        elif etype == "shear":
            i = e["i"]  # node index
            F_shear[i] = F
            tau_shear[i] = F / (params.b_eff * params.dx)

    # Convert to GPa for plotting
    sig_tow_GPa = sig_tow / 1e9
    sig_bridge_GPa = sig_bridge / 1e9

    # Plot tow stress (only tow elements)
    plt.figure()
    for row in range(params.n_rows):
        plt.plot(x_mid, sig_tow_GPa[row, :], label=f"Tow axial stress row {row}")
    plt.xlabel("x")
    plt.ylabel("Tow axial stress (GPa)")
    plt.title("Tow stress from tow_axial elements only")
    plt.legend()
    plt.grid(True)

    # Plot bridge stress separately (only bridge elements)
    plt.figure()
    for row in range(params.n_rows):
        plt.plot(x_mid, sig_bridge_GPa[row, :], label=f"Bridge axial stress row {row}")
    plt.xlabel("x")
    plt.ylabel("Bridge axial stress (GPa)")
    plt.title("Bridge stress from tow_bridge elements only")
    plt.legend()
    plt.grid(True)

    # Plot shear traction (MPa) along interface
    plt.figure()
    plt.plot(x, tau_shear / 1e6, label="Interface shear traction")
    plt.xlabel("x")
    plt.ylabel("Shear traction (MPa)")
    plt.title("Inter-row shear traction (from shear elements)")
    plt.legend()
    plt.grid(True)

    plt.show()


def main():
    params = Params()
    x = build_mesh(params.L, params.dx)

    K, f ,elems= assemble_phase1(params, x)
    prescribed = build_dirichlet_bcs(params, x)

    K_ff, f_eff, free_dofs, u_full = apply_dirichlet_elimination(K, f, prescribed)

    # Solve reduced system
    u_free = spsolve(K_ff, f_eff)

    # Reconstruct full displacement vector
    u_full[free_dofs] = u_free

    postprocess_by_element_type(params, x, u_full, elems)


if __name__ == "__main__":
    main()
