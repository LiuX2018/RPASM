import sys

sys.path.append("./")

import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import optics.wave_lib as wave_lib
import optics.aperture_lib as aperture_lib
import optics.diffraction_simulator as diff_sim
from optics.constant import mm
from optics.config import WVL, FL, zf, fd, r, dl


def main_func(numerical_mode, iter_num, zo, thetaX, thetaY, l):
    propagator = "LSASM"
    # propagator = "RS"

    range_x = range_y = (-r, r)  # Coordinate range of aperture
    OVER_SF = 10  # Oversampling factor for LSASM

    DEVICE = "cuda:0"
    # DEVICE = 'cpu'

    ############################ Observation window ############################
    dx = dy = dl / 10
    Mx, My = int(l / dx), int(l / dy)

    xc = (
        zf
        * torch.sin(thetaX)
        / torch.sqrt(1 - torch.sin(thetaX) ** 2 - torch.sin(thetaY) ** 2)
    )
    yc = (
        zf
        * torch.sin(thetaY)
        / torch.sqrt(1 - torch.sin(thetaX) ** 2 - torch.sin(thetaY) ** 2)
    )

    if numerical_mode == "single":
        real_dtype = torch.float32
    elif numerical_mode == "half":
        real_dtype = torch.float16
    else:
        real_dtype = torch.float64

    x = torch.linspace(-l / 2 + xc, l / 2 + xc, Mx, dtype=real_dtype)
    y = torch.linspace(-l / 2 + yc, l / 2 + yc, My, dtype=real_dtype)
    print(f"observation window size = {l / mm:.2f} mm.")

    ############################ Source field ############################
    Uin = wave_lib.SphericalWavePlusThinLens(
        WVL, range_x, range_y, OVER_SF, thetaX, thetaY, -zo, FL, DEVICE, numerical_mode
    )
    Uin.forward()
    Uin.zo = 1 / (1 / FL - 1 / zo)
    print(f"Samples = {Uin.field.shape[0], Uin.field.shape[1]}.")

    CircAperture = aperture_lib.CircAperture(
        range_x, range_y, Uin.field.shape[-1], Uin.field.shape[-2], 0, 0, r, DEVICE
    )
    Uin.field = CircAperture(Uin.field)
    Uin.field_on_axis = CircAperture(Uin.field_on_axis)

    if propagator == "RS":
        prop2 = diff_sim.RS(Uin, x, y, zf, DEVICE, numerical_mode)
    elif propagator == "LSASM":
        prop2 = diff_sim.RPASM(Uin, x, y, zf, DEVICE, numerical_mode)

    ############################ Propagation ############################
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(iter_num), desc="Repeating evaluation", position=0):
        U2 = prop2(Uin)
    torch.cuda.synchronize()
    end = time.time()
    runtime = end - start
    print(f"Time elapsed: {runtime:.2f} seconds for {iter_num} iterations.")

    return Uin, U2, runtime


if __name__ == "__main__":
    ############################ Configuration ############################
    # numerical_mode = 'double'
    # numerical_mode = 'single'
    # numerical_mode = 'half'
    # numerical_mode = "double2single"
    numerical_mode = "double2half"

    iter_num = 1
    # iter_num = 1000

    # thetaX, thetaY, l = (
    #     torch.as_tensor(0),
    #     torch.as_tensor(0),
    #     10 * dl,
    # )  # Incident angle in degree

    thetaX, thetaY, l = (
        torch.as_tensor(3),
        torch.as_tensor(3),
        100 * dl,
    )  # Incident angle in degree

    # thetaX, thetaY, l = (
    #     torch.as_tensor(10),
    #     torch.as_tensor(10),
    #     700 * dl,
    # )  # Incident angle in degree

    ############################ Main function ############################
    Uin, U2, runtime = main_func(
        numerical_mode,
        iter_num,
        fd,
        torch.deg2rad(thetaX),
        torch.deg2rad(thetaY),
        l,
    )

    ############################ Visualization ############################
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    # Input amplitude
    im0 = axes[0, 0].imshow(torch.abs(Uin.field).cpu().detach(), origin="lower")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 0].set_title("input amplitude")

    # Input phase
    im1 = axes[0, 1].imshow(torch.angle(Uin.field).cpu().detach(), origin="lower")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].set_title("input phase")

    # Output amplitude
    im2 = axes[1, 0].imshow(
        torch.abs(U2).cpu().detach() ** 2, origin="lower", cmap="inferno"
    )
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title("output amplitude")

    # Output phase
    im3 = axes[1, 1].imshow(torch.angle(U2).cpu().detach(), origin="lower")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].set_title("output phase")

    plt.tight_layout()
    plt.show()
