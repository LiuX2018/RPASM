import math
import torch
from tqdm import tqdm

import optics.opt_dft as opt_dft


class DiffSim:
    def __init__(self, vec_x_ow, vec_y_ow, z, DEVICE):
        self.device = DEVICE
        self.vec_x_ow, self.vec_y_ow = (
            torch.as_tensor(vec_x_ow, device=DEVICE),
            torch.as_tensor(vec_y_ow, device=DEVICE),
        )
        self.z = torch.as_tensor(z, device=DEVICE)

    def __call__(self):
        pass


class RS(DiffSim):
    """Class for Rayleigh-Sommerfeld diffraction simulator"""

    def __init__(
        self, Uin, vec_x_ow, vec_y_ow, z, DEVICE, numerical_mode="double2single"
    ):
        super().__init__(vec_x_ow, vec_y_ow, z, DEVICE)

        self.numerical_mode = numerical_mode

        self.xx, self.yy = torch.meshgrid(self.vec_x_ow, self.vec_y_ow, indexing="xy")
        self.dx = Uin.vec_x[1] - Uin.vec_x[0]
        self.dy = Uin.vec_y[1] - Uin.vec_y[0]
        self.xx_flat, self.yy_flat = (
            self.xx.flatten().unsqueeze(0),
            self.yy.flatten().unsqueeze(0),
        )
        self.xx0_flat, self.yy0_flat = (
            Uin.xx.flatten().unsqueeze(0),
            Uin.yy.flatten().unsqueeze(0),
        )
        self.field0 = Uin.field.flatten().unsqueeze(0)
        self.batch_size = 1  # batch size for vectorized computation
        self.Uout = torch.zeros_like(
            self.xx_flat, dtype=Uin.field.dtype, device=self.device
        )

        if self.numerical_mode == "double2single":
            self.dtype_real = torch.float32
        elif self.numerical_mode == "double2half":
            self.dtype_real = torch.float16

    def __call__(self, Uin):
        num_elements = self.Uout.numel()
        for start in tqdm(
            range(0, num_elements, self.batch_size),
            desc="Sampling output field",
            position=0,
        ):
            end = min(start + self.batch_size, num_elements)
            idx_ii = range(start, end)
            r = torch.sqrt(
                (self.xx0_flat.t() - self.xx_flat[:, idx_ii]) ** 2
                + (self.yy0_flat.t() - self.yy_flat[:, idx_ii]) ** 2
                + self.z**2
            )
            E_tmp = torch.exp(1j * Uin.k * r).to(Uin.field.dtype)
            if (self.numerical_mode == "double2single") or (
                self.numerical_mode == "double2half"
            ):
                r_tmp, k_tmp, z_tmp = (
                    r.to(self.dtype_real),
                    Uin.k.to(self.dtype_real),
                    self.z.to(self.dtype_real),
                )
            else:
                r_tmp, k_tmp, z_tmp = r, Uin.k, self.z

            h = (
                1
                / (2 * torch.pi)
                * (1 / r_tmp - 1j * k_tmp)
                * E_tmp
                / r_tmp
                * z_tmp
                / r_tmp
            )

            self.Uout[:, idx_ii] = torch.matmul(self.field0, h)

        return self.Uout.reshape(self.xx.shape) * self.dx * self.dy


class RPASM(DiffSim):
    """Diffraction simulator using the reduced-precision angular spectrum method"""

    def __init__(
        self,
        Uin,
        vec_x_ow,
        vec_y_ow,
        z,
        DEVICE,
        numerical_mode="double2single",
        scale_field_flag=True,
    ):
        """
        Uin: input field (Class)
        vec_x_ow, vec_y_ow: coordinate vectors of observation window
        z: propagation distance
        """
        super().__init__(vec_x_ow, vec_y_ow, z, DEVICE)

        self.numerical_mode = numerical_mode
        self.scale_field_flag = scale_field_flag

        if numerical_mode == "single":
            real_dtype = torch.float32
            complex_dtype = torch.complex64
        elif numerical_mode == "half":
            real_dtype = torch.float16
            complex_dtype = torch.complex32
        else:
            real_dtype = torch.float64
            complex_dtype = torch.complex128

        # Center and size of observation window
        xc_ow, yc_ow = (
            (self.vec_x_ow[0] + self.vec_x_ow[-1]) / 2,
            (self.vec_y_ow[0] + self.vec_y_ow[-1]) / 2,
        )
        wx, wy = (
            self.vec_x_ow[-1] - self.vec_x_ow[0],
            self.vec_y_ow[-1] - self.vec_y_ow[0],
        )
        self.dx_ow, self.dy_ow = (
            wx / (len(self.vec_x_ow) - 1),
            wy / (len(self.vec_y_ow) - 1),
        )
        self.wx, self.wy = wx, wy

        # Combined phase gradient analysis
        gx1, gy1 = self.grad_H(
            Uin.wvl, self.z, Uin.fbx / 2 + Uin.fcx, Uin.fby / 2 + Uin.fcy
        )
        gx2, gy2 = self.grad_H(
            Uin.wvl, self.z, -Uin.fbx / 2 + Uin.fcx, -Uin.fby / 2 + Uin.fcy
        )
        FHcx = (gx1 + gx2) / (2 * torch.pi) / 2
        FHcy = (gy1 + gy2) / (2 * torch.pi) / 2

        # Specify the frequency sampling for each type of input field
        # del Uin.zo  # Remove the available attribute 'zo' in Uin
        if "zo" in dir(Uin):  # If Uin behaves like a spherical wave
            hx = Uin.k * Uin.zo * Uin.wvl**2 * Uin.fbx / 2
            hy = Uin.k * Uin.zo * Uin.wvl**2 * Uin.fby / 2
            FUHbx = torch.abs((hx + gx1) - (-hx + gx2)) / (2 * torch.pi)
            FUHby = torch.abs((hy + gy1) - (-hy + gy2)) / (2 * torch.pi)

            deltax = self.compute_shift_of_H(FHcx, FUHbx + 2 * Uin.size_x, xc_ow, wx)
            deltay = self.compute_shift_of_H(FHcy, FUHby + 2 * Uin.size_y, yc_ow, wy)

            FUHcx_shifted = FHcx + deltax
            FUHcy_shifted = FHcy + deltay

            tau_UHx = 2 * torch.abs(FUHcx_shifted) + FUHbx + 2 * Uin.size_x
            tau_UHy = 2 * torch.abs(FUHcy_shifted) + FUHby + 2 * Uin.size_y
        else:
            tau_UHx = tau_UHy = torch.inf

        # Upper bound
        FHbx = torch.abs(gx1 - gx2) / (2 * torch.pi)
        FHby = torch.abs(gy1 - gy2) / (2 * torch.pi)

        # Compute the optimal shift of the observation window
        deltax = self.compute_shift_of_H(FHcx, FHbx + Uin.size_x, xc_ow, wx)
        deltay = self.compute_shift_of_H(FHcy, FHby + Uin.size_y, yc_ow, wy)

        FHcx_shifted = FHcx + deltax
        FHcy_shifted = FHcy + deltay

        tau_fx_bound = 2 * torch.abs(FHcx_shifted) + FHbx + Uin.size_x
        tau_fy_bound = 2 * torch.abs(FHcy_shifted) + FHby + Uin.size_y

        # Final phase gradient
        tau_UHx = min(tau_UHx, tau_fx_bound) + Uin.over_sf * 41.2 / Uin.fbx / 2
        tau_UHy = min(tau_UHy, tau_fy_bound) + Uin.over_sf * 41.2 / Uin.fby / 2

        dfx_UH = 1 / tau_UHx
        dfy_UH = 1 / tau_UHy

        # Maximum sampling interval limited by OW
        dfx_ow = 1 / (2 * torch.abs(xc_ow - deltax) + wx)
        dfy_ow = 1 / (2 * torch.abs(yc_ow - deltay) + wy)

        # Minimum requirements of sampling interval in frequency domain
        dfx_tmp = max(dfx_UH, 1 / (1 / dfx_UH / 2 + 1 / dfx_ow))
        dfy_tmp = max(dfy_UH, 1 / (1 / dfy_UH / 2 + 1 / dfy_ow))

        # The least samples in frequency domain
        self.sp_fx = math.ceil(Uin.fbx / dfx_tmp)
        self.sp_fy = math.ceil(Uin.fby / dfy_tmp)

        if self.sp_fx > 5000 or self.sp_fy > 5000:
            print(f"Frequency samples = {self.sp_fx, self.sp_fy}.")

        dfx = Uin.fbx / self.sp_fx
        dfy = Uin.fby / self.sp_fy

        # Coordinates in frequency domain
        self.fx = torch.linspace(
            -Uin.fbx / 2,
            Uin.fbx / 2 - dfx,
            self.sp_fx,
            dtype=real_dtype,
            device=self.device,
        )
        self.fy = torch.linspace(
            -Uin.fby / 2,
            Uin.fby / 2 - dfy,
            self.sp_fy,
            dtype=real_dtype,
            device=self.device,
        )
        self.fx_shift, self.fy_shift = self.fx + Uin.fcx, self.fy + Uin.fcy
        fxx_shift, fyy_shift = torch.meshgrid(
            self.fx_shift, self.fy_shift, indexing="xy"
        )

        # Original longitudinal wavevector
        dir_cos_z = torch.sqrt(
            (1 - (fxx_shift * Uin.wvl) ** 2 - (fyy_shift * Uin.wvl) ** 2).to(
                complex_dtype
            )
        )
        kz_compensate = torch.tensor(0.0, dtype=real_dtype, device=self.device)

        kz_orgn = Uin.k * dir_cos_z

        # Linear phase due to observation window shift
        linear_phase = Uin.k * Uin.wvl * (fxx_shift * deltax + fyy_shift * deltay)

        # Express phase in double precision and then convert the complex H to single precision
        self.H = (
            torch.exp(1j * (kz_orgn * self.z + linear_phase))
            * torch.exp(1j * kz_compensate * self.z)
        ).to(Uin.field_on_axis.dtype)

        # Shift the observation window back to origin
        self.x, self.y = self.vec_x_ow - deltax, self.vec_y_ow - deltay

        # Linear phase on observation plane
        xx, yy = torch.meshgrid(self.x, self.y, indexing="xy")
        self.linear_phase_spat = torch.exp(
            1j * Uin.k * Uin.wvl * (Uin.fcx * xx + Uin.fcy * yy)
        ).to(Uin.field_on_axis.dtype)

        # Prepare the amplitude scaling factor
        if self.numerical_mode == "double":
            self.scale_field_flag = False
        elif (self.numerical_mode == "double2single") or (
            self.numerical_mode == "single"
        ):
            self.range_min, self.range_max = 1.4e-45, 3.4e38
        elif self.numerical_mode == "double2half" or (self.numerical_mode == "half"):
            self.range_min, self.range_max = 6e-8, 6.5e4

        # Prepare the DFT parameters in double precision
        self.mx, self.my, self.dx0, self.dy0 = opt_dft.prepare_mdft(
            Uin.vec_x, Uin.vec_y, self.fx, self.fy
        )
        self.imx, self.imy, self.dfx, self.dfy = opt_dft.prepare_midft(
            self.x, self.y, self.fx, self.fy
        )

    def __call__(self, Uin):
        if self.scale_field_flag:
            total_energy = (
                self.dx0
                * self.dy0
                * torch.abs(Uin.field_on_axis.to(torch.complex128)) ** 2
            ).sum()

            self.sf0 = opt_dft.field_scale_factor(
                self.range_min,
                self.range_max,
                total_energy,
                self.dx0,
                self.dy0,
                Uin.sample_x,
                Uin.sample_y,
            )
            self.sf1 = opt_dft.field_scale_factor(
                self.range_min,
                self.range_max,
                total_energy,
                self.dfx,
                self.dfy,
                self.sp_fx,
                self.sp_fy,
            )
        else:
            self.sf0 = self.sf1 = 1

        field_tmp = (Uin.field_on_axis).to(
            torch.complex128
        ) * self.sf0  # Amplitude pre-scaling in spatial domain

        # Spectrum of input field
        Fu = opt_dft.mdft(
            field_tmp, self.mx, self.my, self.dx0, self.dy0, Uin.field_on_axis.dtype
        )

        # Spectrum of output field
        Fout = (Fu * self.H).to(torch.complex128) / self.sf0  # Scale back
        Fout_ = Fout * self.sf1  # Amplitude pre-scaling in frequency domain

        # Output field
        Uout = opt_dft.midft(
            Fout_, self.imx, self.imy, self.dfx, self.dfy, Uin.field_on_axis.dtype
        )
        Uout = Uout.to(torch.complex128) / self.sf1  # Scale back
        Uout = Uout * self.linear_phase_spat

        return Uout

    def grad_H(self, lam, z, fx, fy):
        """
        Compute the gradient of H

        Parameters
        ----------
        lam: Uin.wvl
        z: propagation distance
        fx: x component of spatial frequency (scalar value)
        fy: y component of spatial frequency (scalar value)
        """

        # 'abs' avoids evanescent wave
        dir_cos_z = torch.abs(1 - (lam * fx) ** 2 - (lam * fy) ** 2)

        gradx = -z * 2 * torch.pi * lam * fx / torch.sqrt(dir_cos_z)
        grady = -z * 2 * torch.pi * lam * fy / torch.sqrt(dir_cos_z)
        return gradx, grady

    def compute_shift_of_H(self, C1, C2, pc, w, type="intuitive"):
        """
        Compute required virtual shift of observation window

        Parameters
        ----------
        C1: center of H's spectrum
        C2: bandwidth of H's spectrum
        pc: center of observation window
        w: width of observation window
        """
        if type == "intuitive":
            delta = pc
        else:
            if -2 * C1 - 2 * pc + C2 < w < 2 * C1 + 2 * pc + C2:
                delta = pc / 2 + w / 4 - C1 / 2 - C2 / 4
            elif 2 * C1 + 2 * pc + C2 < w < -2 * C1 - 2 * pc + C2:
                delta = pc / 2 - w / 4 - C1 / 2 + C2 / 4
            elif (w > 2 * C1 + 2 * pc + C2) and (w > -2 * C1 - 2 * pc + C2):
                delta = pc
            elif (w < 2 * C1 + 2 * pc + C2) and (w < -2 * C1 - 2 * pc + C2):
                delta = -C1

        return delta
