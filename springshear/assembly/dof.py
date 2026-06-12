from __future__ import annotations

from dataclasses import dataclass

from springshear.params import Params


@dataclass
class DofMap:
    params: Params
    n_nodes: int

    def n_intf_y(self) -> int:
        if self.params.periodic_y:
            return self.params.n_rows
        return max(0, self.params.n_rows - 1)

    def ply_block_size(self) -> int:
        return self.params.n_rows * self.n_nodes + self.n_intf_y() * self.n_nodes

    def inter_ply_block_size(self) -> int:
        return self.params.n_rows * self.n_nodes

    def n_boundaries(self) -> int:
        if self.params.n_plies <= 1:
            return 1 if self.params.periodic_z else 0
        n = self.params.n_plies - 1
        if self.params.periodic_z:
            n += 1
        return n

    def total_dofs(self) -> int:
        return (
            self.params.n_plies * self.ply_block_size()
            + self.n_boundaries() * self.inter_ply_block_size()
        )

    def ply_base(self, ply: int) -> int:
        return ply * (self.ply_block_size() + self.inter_ply_block_size())

    def inter_ply_base(self, boundary: int) -> int:
        return (boundary + 1) * self.ply_block_size() + boundary * self.inter_ply_block_size()

    def dof_tow(self, ply: int, row: int, node: int) -> int:
        return self.ply_base(ply) + row * self.n_nodes + node

    def dof_intf_y(self, ply: int, r_int: int, node: int) -> int:
        return self.ply_base(ply) + self.params.n_rows * self.n_nodes + r_int * self.n_nodes + node

    def dof_inter_ply(self, boundary: int, row: int, node: int) -> int:
        return self.inter_ply_base(boundary) + row * self.n_nodes + node

    def boundary_plies(self, boundary: int) -> tuple[int, int]:
        """Return ply indices (lower, upper) connected by a boundary."""
        if self.params.periodic_z and boundary == self.n_boundaries() - 1:
            return self.params.n_plies - 1, 0
        return boundary, boundary + 1
