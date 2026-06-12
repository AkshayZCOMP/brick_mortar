def add_2node_spring(K, a: int, b: int, k: float) -> None:
    K[a, a] += k
    K[b, b] += k
    K[a, b] -= k
    K[b, a] -= k


def k_axial(E: float, A: float, dx: float) -> float:
    return E * A / dx


def k_shear(G_m: float, b_eff: float, t_eff: float, dx: float) -> float:
    return G_m * b_eff * dx / t_eff
