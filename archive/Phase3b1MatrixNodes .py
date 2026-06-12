
import numpy as np
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Params:
    # Geometry / discretization
    
    L_fiber: float = 50e-3
    n_fibers: int = 4
    dx: float = 0.1e-3
    n_rows: int = 5
    row_width: int = 4e-3 #mm
    resin_gap: float =  0.5e-3 #mm
    L: float = (L_fiber+resin_gap)*n_fibers #mm
    # Loading
    eps0: float = 0.01  # applied macroscopic strain (displacement control)

    # Tow properties (axial)
    E_tow: float = 140e9
    D_f = 4e-3
    t_f = .1e-3
    A_tow: float = D_f * t_f # cross-sectional area of a single tow

    row_break = [L*.49, L*.51]
    # Matrix/interface (shear transfer)
    G_m: float = 1.0e9
    E_m: float = 3.0e9
    t_eff: float = .10e-3 #mm
    b_eff: float = 1.0e-3 #mm  # effective width for patch area
#======================================================================================
#staggering helper

def staggering_coords(params: Params, odd_shift: float = None):
    """
    Return x-intervals (start,end) for resin gaps for:
      - even rows: no shift
      - odd rows: shifted by odd_shift (default = L_fiber/2)

    All intervals are clipped to [0, params.L].
    """
    Lf = float(params.L_fiber)
    g  = float(params.resin_gap)
    L  = float(params.L)

    if odd_shift is None:
        odd_shift = 0.5 * Lf  # classic half-brick staggering

    pitch = Lf + g

    def clip_interval(a, b):
        a2 = max(0.0, a)
        b2 = min(L, b)
        if b2 <= a2:
            return None
        return (round(a2, 6), round(b2, 6))

    # ---- Even-row gaps (no shift) ----
    even_gaps = []
    # gap k sits right after tow segment k: [end_of_tow_k, end_of_tow_k + g]
    # end_of_tow_k = (k+1)*Lf + k*g
    # choose k range large enough to cover domain
    k_max = int(np.ceil(L / pitch)) + 2

    for k in range(k_max):
        x0 = (k + 1) * Lf + k * g
        x1 = x0 + g
        iv = clip_interval(x0, x1)
        if iv is not None:
            even_gaps.append(iv)

    # ---- Odd-row gaps (shifted) ----
    odd_gaps = []
    # shift the entire pattern left by odd_shift (equivalently: subtract odd_shift)
    for k in range(k_max):
        x0 = (k + 1) * Lf + k * g - odd_shift
        x1 = x0 + g
        iv = clip_interval(x0, x1)
        if iv is not None:
            odd_gaps.append(iv)

    return even_gaps, odd_gaps

#======================================================================================
# indexing help
#=======================================================================================
def build_mesh(L: float, dx: float) -> np.ndarray:
    # Ensure endpoint included
    n = int(np.round(L / dx))
    #x = np.linspace(0.0, L, n + 1)
    x=  np.arange(n+1, dtype=float) * dx
    x[-1] = L
    return x.round(6)


def dof_index(row: int, node: int, n_nodes: int) -> int:
    # global DOF index for 1 DOF per node per row
    return int(row * n_nodes + node)
def dof_interface(n_rows: int, r_int: int, node: int, n_nodes: int) -> int:
    # r_int indexes the interface between row r_int and r_int+1
    # r_int = 0..n_rows-2 (only inbetween tow rows)
    tow_offset = n_rows * n_nodes
    return tow_offset + r_int * n_nodes + node

#======================================================================================
#element assembly
#=====================================================================================


print( build_mesh(Params.L, Params.dx)  )

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
    n_tow = params.n_rows *n_nodes
    n_interfaces = (params.n_rows -1) * n_nodes
    n_dofs = n_tow + n_interfaces


    #list of list sparse matrix
    K = lil_matrix((n_dofs, n_dofs))
    f = np.zeros(n_dofs, dtype= float)


    #axial springs in both rows (continuous to start) 
    kax = k_axial(params.E_tow, params.A_tow, params.dx)
    ksh = k_shear(params.G_m, params.b_eff, params.t_eff, params.dx)
    k_tm = 2*ksh
    
    A_m = params.b_eff * params.t_eff
    kbridge = params.E_m*A_m/(params.dx) # spring constant for shear spring in break region
    k_m_ax = params.E_m * A_m / (params.dx)
    elems = [] 
    evenoffset, oddoffset = staggering_coords(params)
    for row in range(params.n_rows):
        if row %2 ==0:

            for i in range(n_nodes-1):
                 xmid = (x[i]+x[i+1])/2 
                 inside = any(xstart<=xmid<=xend for (xstart, xend) in evenoffset)
                 if not inside:
                          a = dof_index(row, i, n_nodes)
                          b = dof_index(row, i+1, n_nodes)
                          add_2node_spring(K, a, b, kax)
                          elems.append({"etype": "tow_axial", "row": row, "i": i, "a": a, "b": b, "k": kax, "A": params.A_tow})
                 if inside: 
                        a = dof_index(row, i, n_nodes)
                        b = dof_index(row, i+1, n_nodes)
                        add_2node_spring(K, a, b, kbridge)
                        elems.append({"etype": "tow_bridge", "row": row, "i": i, "a": a, "b": b, "k": kbridge, "A": params.A_tow})
                 
        if row %2 !=0:
            for i in range(n_nodes-1):
                xmid = (x[i]+x[i+1])/2
                inside = any(xstart <= xmid <= xend for (xstart, xend) in oddoffset)
                if not inside: 
                    a = dof_index(row, i, n_nodes)
                    b = dof_index(row, i+1, n_nodes)
                    add_2node_spring(K, a, b, kax)
                    elems.append({"etype": "tow_axial", "row": row, "i": i, "a": a, "b": b, "k": kax, "A": params.A_tow})
                if inside:
                    a = dof_index(row, i, n_nodes)
                    b = dof_index(row, i+1, n_nodes)
                    add_2node_spring(K, a, b, kbridge)
                    elems.append({"etype": "tow_bridge", "row": row, "i": i, "a": a, "b": b, "k": kbridge, "A": params.A_tow})


      
    for r_int in range(params.n_rows - 1):
        for i in range(n_nodes):
                m = dof_interface(params.n_rows, r_int, i, n_nodes)
                a = dof_index(r_int, i, n_nodes)
                b = dof_index(r_int+1, i, n_nodes)
                add_2node_spring(K, a, m, k_tm)
                add_2node_spring(K, b, m, k_tm)
                elems.append({"etype": "tm_shear", "r_int": r_int, "row": r_int,     "i": i, "a": a, "m": m, "k": k_tm})
                elems.append({"etype": "tm_shear", "r_int": r_int, "row": r_int + 1, "i": i, "a": b, "m": m, "k": k_tm})
          
    for r_int in range(params.n_rows - 1):
        for i in range(n_nodes-1):
                m0 = dof_interface(params.n_rows, r_int, i, n_nodes)
                m1 = dof_interface(params.n_rows, r_int, i+1, n_nodes)

                add_2node_spring(K, m0, m1, k_m_ax)
                elems.append({"etype":"m_axial", "r_int": r_int, "i": i, "a": m0, "b": m1, "k": k_m_ax, "A": A_m})


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

    # ----------------------------
    # Storage (Pa unless converted)
    # ----------------------------
    sig_tow = np.full((params.n_rows, n_nodes - 1), np.nan, dtype=float)     # tow_axial only
    sig_bridge = np.full((params.n_rows, n_nodes - 1), np.nan, dtype=float)  # tow_bridge only

    # Interface traction (between row r_int and r_int+1), node-based
    tau_tm = np.full((params.n_rows - 1, n_nodes), np.nan, dtype=float)
    u_m = np.full((params.n_rows - 1, n_nodes), np.nan, dtype=float)

    # Matrix axial stress along each interface (segment-based) from m_axial elements
    sig_m = np.full((params.n_rows - 1, n_nodes - 1), np.nan, dtype=float)

    # Effective matrix area used for bridge/matrix elements (consistent with your kbridge, k_m_ax)
    A_m = params.b_eff * params.t_eff

    # ----------------------------
    # Populate from element list
    # ----------------------------
    for e in elems:
        etype = e["etype"]

        # -------- Tow axial / bridge elements (2-node along x)
        if etype in ("tow_axial", "tow_bridge"):
            a = e["a"]
            b = e["b"]
            k = e["k"]
            F = k * (u[b] - u[a])  # axial force in the element

            row = e["row"]
            i = e["i"]

            # Use element-provided area if present; otherwise fall back.
            # NOTE: If your "bridge" is actually matrix replacement (as in your current kbridge),
            # you should use A_m, not A_tow.
            if etype == "tow_axial":
                A = e.get("A", params.A_tow)
                sig_tow[row, i] = F / A
            else:
                # Robust handling: if you forgot to store the bridge area correctly, force A_m here
                A = e.get("A", A_m)
                # If your elems still store A_tow for bridge, override to A_m to stay consistent with kbridge choice.
                # Comment out the next line ONLY if your bridge truly has tow area and tow modulus.
                A = A_m
                sig_bridge[row, i] = F / A

        # -------- Tow–matrix coupling springs (tow node <-> interface node at same x)
        elif etype == "tm_shear":
            r_int = e["r_int"]   # interface index
            row = e["row"]       # tow row index for this spring
            i = e["i"]

            a = e["a"]           # tow dof
            m = e["m"]           # interface dof
            k = e["k"]

            F_am = k * (u[a] - u[m])  # spring force

            # Store traction once per interface location (use only the "lower-row" spring)
            if row == r_int:
                tau_tm[r_int, i] = F_am / (params.b_eff * params.dx)  # Pa
                u_m[r_int, i] = u[m]

        # -------- Matrix axial chain elements along interface (m_axial)
        elif etype == "m_axial":
            r_int = e["r_int"]
            i = e["i"]
            a = e["a"]
            b = e["b"]
            k = e["k"]
            A = e.get("A", A_m)

            F = k * (u[b] - u[a])
            sig_m[r_int, i] = F / A

    # ----------------------------
    # Unit conversions for plotting
    # ----------------------------
    sig_tow_GPa = sig_tow / 1e9
    sig_bridge_GPa = sig_bridge / 1e9
    tau_tm_MPa = tau_tm / 1e6
    sig_m_GPa = sig_m / 1e9

    # ----------------------------
    # Build a constant-denominator "row stress" field (fixes NaN-average spikes)
    # Interpretation: per x-segment, each row carries either tow stress, bridge stress, or 0 if absent.
    # ----------------------------
    sig_row = np.zeros((params.n_rows, n_nodes - 1), dtype=float)  # Pa, constant denominator
    mask_tow = ~np.isnan(sig_tow)
    mask_bridge = ~np.isnan(sig_bridge)

    sig_row[mask_tow] = sig_tow[mask_tow]
    sig_row[mask_bridge] = sig_bridge[mask_bridge]  # overwrite where bridge exists

    sig_row_GPa = sig_row / 1e9
    sig_avg_x = np.mean(sig_row_GPa, axis=0)  # average across ALL rows (constant denominator)

    # Diagnostics: how many rows are tow vs bridge at each x segment
    n_tow = np.sum(mask_tow, axis=0)
    n_bridge = np.sum(mask_bridge, axis=0)

    # ----------------------------
    # Plot styling helpers
    # ----------------------------
    styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "d", "x", "+", "v", ">", "<", "p", "h"]

    # ----------------------------
    # Plot 1: tow axial stress per row
    # ----------------------------
    plt.figure()
    for row in range(params.n_rows):
        style = styles[row % len(styles)]
        marker = markers[row % len(markers)]
        plt.plot(
            x_mid,
            sig_tow_GPa[row, :],
            linestyle=style,
            marker=marker,
            markevery=max(1, (n_nodes - 1) // 30),
            linewidth=1.5,
            label=f"Tow row {row}"
        )
    plt.xlabel("x")
    plt.ylabel("Tow axial stress (GPa)")
    plt.title("Tow stress from tow_axial elements")
    plt.legend()
    plt.grid(True)

    # ----------------------------
    # Plot 2: bridge axial stress per row (scatter)
    # ----------------------------
    plt.figure()
    for row in range(params.n_rows):
        y = sig_bridge_GPa[row, :]
        mask = ~np.isnan(y)
        if np.any(mask):
            marker = markers[row % len(markers)]
            plt.scatter(
                x_mid[mask],
                y[mask],
                s=25,
                marker=marker,
                label=f"Bridge row {row}"
            )
    plt.xlabel("x")
    plt.ylabel("Bridge axial stress (GPa)")
    plt.title("Bridge stress from tow_bridge elements")
    plt.legend()
    plt.grid(True)

    # ----------------------------
    # Plot 3: tow–matrix traction per interface
    # ----------------------------
    plt.figure()
    for r_int in range(params.n_rows - 1):
        plt.plot(x, tau_tm_MPa[r_int, :], label=f"Interface {r_int}-{r_int+1}")
    plt.xlabel("x")
    plt.ylabel("Tow–matrix traction (MPa)")
    plt.title("Tow–matrix traction along each interface")
    plt.legend()
    plt.grid(True)

    # ----------------------------
    # Plot 4: laminate-average stress (constant denominator, no NaN spikes)
    # ----------------------------
    plt.figure()
    plt.plot(x_mid, sig_avg_x, linewidth=2.0, label="Average (tow+bridge, all rows)")
    plt.xlabel("x")
    plt.ylabel("Average axial stress (GPa)")
    plt.title("Average axial stress along length (constant-denominator)")
    plt.legend()
    plt.grid(True)

    # ----------------------------
    # Plot 5: diagnostic counts of active segments
    # ----------------------------
    plt.figure()
    plt.plot(x_mid, n_tow, label="# tow_axial rows active")
    plt.plot(x_mid, n_bridge, label="# tow_bridge rows active")
    plt.xlabel("x")
    plt.ylabel("Count")
    plt.title("How many rows contribute tow vs bridge at each x-segment")
    plt.legend()
    plt.grid(True)

    # ----------------------------
    # Plot 6 (optional): matrix axial stress along each interface
    # ----------------------------
    plt.figure()
    for r_int in range(params.n_rows - 1):
        plt.plot(x_mid, sig_m_GPa[r_int, :], label=f"Matrix axial (interface {r_int}-{r_int+1})")
    plt.xlabel("x")
    plt.ylabel("Matrix axial stress (GPa)")
    plt.title("Matrix axial stress along each interface (m_axial)")
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
#======================================================================================
# Add Axial Nodes
