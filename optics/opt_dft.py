import sys

sys.path.append("./")

import torch
from warnings import warn


def torch_matmul_complex32(A, B):
    """Matrix multiplication for complex32 data type."""

    C = torch.view_as_complex(
        torch.stack(
            (
                torch.matmul(A.real, B.real) - torch.matmul(A.imag, B.imag),
                torch.matmul(A.real, B.imag) + torch.matmul(A.imag, B.real),
            ),
            dim=-1,
        )
    )

    return C


def prepare_mdft(x, y, fx, fy):
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-2)
    fx = fx.unsqueeze(-2)
    fy = fy.unsqueeze(-1)

    mx = torch.exp(-2 * torch.pi * 1j * torch.matmul(x, fx))
    my = torch.exp(-2 * torch.pi * 1j * torch.matmul(fy, y))

    nx = torch.numel(x)
    ny = torch.numel(y)
    if nx == 1:
        dx = 1
    else:
        dx = (torch.squeeze(x)[-1] - torch.squeeze(x)[0]) / (nx - 1)

    if ny == 1:
        dy = 1
    else:
        dy = (torch.squeeze(y)[-1] - torch.squeeze(y)[0]) / (ny - 1)

    return mx, my, dx, dy


def prepare_midft(x, y, fx, fy):
    x = x.unsqueeze(-2)
    y = y.unsqueeze(-1)
    fx = fx.unsqueeze(-1)
    fy = fy.unsqueeze(-2)

    mx = torch.exp(2 * torch.pi * 1j * torch.matmul(fx, x))
    my = torch.exp(2 * torch.pi * 1j * torch.matmul(y, fy))

    nfx = torch.numel(fx)
    nfy = torch.numel(fy)
    if nfx == 1:
        dfx = 1
    else:
        dfx = (torch.squeeze(fx)[-1] - torch.squeeze(fx)[0]) / (nfx - 1)

    if nfy == 1:
        dfy = 1
    else:
        dfy = (torch.squeeze(fy)[-1] - torch.squeeze(fy)[0]) / (nfy - 1)

    return mx, my, dfx, dfy


def mdft(mat_in, mx, my, dx, dy, dtype):
    """2D DFT realized by matrix triple product (MTP)"""

    mat_in = (mat_in * dx * dy).to(dtype)

    mx, my = mx.to(dtype), my.to(dtype)

    if dtype == torch.complex32:
        mat_out = torch_matmul_complex32(torch_matmul_complex32(my, mat_in), mx)
    else:
        mat_out = torch.matmul(torch.matmul(my, mat_in), mx)

    return mat_out


def midft(mat_in, mx, my, dfx, dfy, dtype):
    """2D IDFT realized by MTP"""

    mat_in = (mat_in * dfx * dfy).to(dtype)

    mx, my = mx.to(dtype), my.to(dtype)

    if dtype == torch.complex32:
        mat_out = torch_matmul_complex32(torch_matmul_complex32(my, mat_in), mx)
    else:
        mat_out = torch.matmul(torch.matmul(my, mat_in), mx)

    return mat_out


@torch.no_grad()
def field_scale_factor(range_min, range_max, total_energy, dx, dy, Nx, Ny):
    """calculate the scaling factor for amplitude"""

    Lx, Ly = Nx * dx, Ny * dy  # size in spatial domain

    c_tmp = torch.sqrt(Lx * Ly * total_energy)
    sf_min = range_min * Nx * Ny / c_tmp
    sf_max = range_max / c_tmp

    if sf_min > sf_max:
        warn("\nThe field can not be accurately represented.", stacklevel=2)
        # i.e., range_max/range_min < Nx*Ny, which means that the dynamic range should be larger than the space bandwidth product

    eps = 5.0e-324  # smallest positive number to avoid zero division
    sf = torch.sqrt(sf_max * (sf_min + eps))  # geometric mean

    return sf
