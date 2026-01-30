import math
import torch
import torch.nn as nn
from optics.constant import m, mm


class Waves(nn.Module):
    def __init__(self, wvl, range_x, range_y, DEVICE):
        super(Waves, self).__init__()
        """
        Parameters
        ----------
        wvl: wavelength
        range_x: x coordinate range of aperture
        range_y: y coordinate range of aperture
        DEVICE: computing device
        """

        self.device = DEVICE
        self.wvl = torch.as_tensor(wvl, device=DEVICE)
        self.k = 2 * torch.pi / self.wvl
        self.range_x, self.range_y = (
            torch.as_tensor(range_x, device=DEVICE),
            torch.as_tensor(range_y, device=DEVICE),
        )
        self.size_x = self.range_x[1] - self.range_x[0]
        self.size_y = self.range_y[1] - self.range_y[0]
        self.fbx_apt, self.fby_apt = self.aperture_bandwidth()

    def aperture_bandwidth(self):
        """
        Parameters
        ----------
        size_x: the aperture size along x direction
        size_y: the aperture size along y direction
        """

        fbx_apt, fby_apt = 41.2 / self.size_x, 41.2 / self.size_y

        return fbx_apt, fby_apt

    def compute_coordinates(self):
        if self.numerical_mode == "single":
            self.real_dtype = torch.float32
        elif self.numerical_mode == "half":
            self.real_dtype = torch.float16
        else:
            self.real_dtype = torch.float64

        vec_x = torch.linspace(
            self.range_x[0],
            self.range_x[-1],
            self.sample_x,
            dtype=self.real_dtype,
            device=self.device,
        )
        vec_y = torch.linspace(
            self.range_y[0],
            self.range_y[-1],
            self.sample_y,
            dtype=self.real_dtype,
            device=self.device,
        )
        xx, yy = torch.meshgrid(vec_x, vec_y, indexing="xy")

        return vec_x, vec_y, xx, yy

    def update_bandwidth(self):
        delta_x_update = (self.vec_x[-1] - self.vec_x[0]) / (self.sample_x - 1)
        delta_y_update = (self.vec_y[-1] - self.vec_y[0]) / (self.sample_y - 1)
        self.fbx, self.fby = 1 / delta_x_update, 1 / delta_y_update

    def least_samples(self):
        delta_x, delta_y = 1 / self.fbx, 1 / self.fby
        sample_x = math.ceil(self.size_x / delta_x)
        sample_y = math.ceil(self.size_y / delta_y)

        return sample_x, sample_y


class GaussianAmplitude(Waves):
    """Class for Gaussian beam representation"""

    def __init__(
        self,
        wvl,
        range_x,
        range_y,
        over_sf=50,
        wo=1 * mm,
        DEVICE="cpu",
        numerical_mode="double2single",
    ):
        """
        Parameters
        ----------
        wvl: wavelength
        range_x: x coordinate range of aperture
        range_y: y coordinate range of aperture
        wo: beam waist
        """
        super().__init__(wvl, range_x, range_y, DEVICE)
        self.over_sf = torch.as_tensor(over_sf, device=DEVICE)
        self.wo = torch.as_tensor(wo, device=DEVICE)
        self.numerical_mode = numerical_mode

        self.zr = torch.pi * self.wo**2 / self.wvl
        self.fcx, self.fcy, self.fbx_wave, self.fby_wave = self.spectrum_parameters()
        self.fbx, self.fby = (
            self.fbx_wave + self.over_sf * self.fbx_apt,
            self.fby_wave + self.over_sf * self.fby_apt,
        )

    def forward(self, sample_x=None, sample_y=None):
        if sample_x is None and sample_y is None:
            sample_x, sample_y = self.least_samples()

        self.sample_x, self.sample_y = sample_x, sample_y
        self.vec_x, self.vec_y, self.xx, self.yy = self.compute_coordinates()
        self.update_bandwidth()

        self.amp = torch.exp(-(self.xx**2 + self.yy**2) / self.wo**2)
        if self.numerical_mode == "double2single" or self.numerical_mode == "single":
            self.amp = self.amp.to(torch.float32)
        elif self.numerical_mode == "double2half" or self.numerical_mode == "half":
            self.amp = self.amp.to(torch.float16)
        else:
            self.amp = self.amp.to(torch.float64)

        phase = torch.zeros_like(self.amp)
        self.field_on_axis = self.field = self.amp * torch.exp(1j * phase)

    def spectrum_parameters(self):
        fcx = fcy = torch.as_tensor(0, device=self.device)

        fbx_wave = 1 / (torch.pi * self.wo)
        fby_wave = 1 / (torch.pi * self.wo)

        return fcx, fcy, fbx_wave, fby_wave


class SphericalWave(Waves):
    """Class for spherical wave representation"""

    def __init__(
        self,
        wvl,
        range_x,
        range_y,
        over_sf=50,
        theta_x=0,
        theta_y=0,
        zo=0,
        DEVICE="cpu",
        numerical_mode="double2single",
    ):
        """
        Parameters
        ----------
        wvl: wavelength
        range_x: x coordinate range of aperture
        range_y: y coordinate range of aperture
        theta_x: angle in x direction (degree)
        theta_y: angle in y direction (degree)
        zo: distance from source to aperture
        """
        super().__init__(wvl, range_x, range_y, DEVICE)
        self.over_sf = torch.as_tensor(over_sf, device=DEVICE)
        self.theta_x = torch.as_tensor(theta_x, device=DEVICE)
        self.theta_y = torch.as_tensor(theta_y, device=DEVICE)
        self.zo = torch.as_tensor(zo, device=DEVICE)
        self.numerical_mode = numerical_mode
        self.sign = 1 if self.zo < 0 else -1

        ro = self.zo / torch.sqrt(
            1 - torch.sin(self.theta_x) ** 2 - torch.sin(self.theta_y) ** 2
        )
        self.xo, self.yo = ro * torch.sin(self.theta_x), ro * torch.sin(self.theta_y)

        self.fcx, self.fcy, self.fbx_wave, self.fby_wave = self.spectrum_parameters()
        self.fbx, self.fby = (
            self.fbx_wave + self.over_sf * self.fbx_apt,
            self.fby_wave + self.over_sf * self.fby_apt,
        )

    def forward(self, sample_x=None, sample_y=None):
        if sample_x is None and sample_y is None:
            sample_x, sample_y = self.least_samples()

        self.sample_x, self.sample_y = sample_x, sample_y
        self.vec_x, self.vec_y, self.xx, self.yy = self.compute_coordinates()
        self.update_bandwidth()

        r = torch.sqrt((self.xx - self.xo) ** 2 + (self.yy - self.yo) ** 2 + self.zo**2)
        phase = self.sign * self.k * r

        amp = 1 / r
        amp /= amp.max()
        self.field = amp * torch.exp(1j * phase)
        self.linear_phase = self.compute_linear_phase()
        self.field_on_axis = self.field * torch.exp(-1j * self.linear_phase)

        if self.numerical_mode == "double2single":
            self.field = self.field.to(torch.complex64)
            self.field_on_axis = self.field_on_axis.to(torch.complex64)
        elif self.numerical_mode == "double2half":
            self.field = self.field.to(torch.complex32)
            self.field_on_axis = self.field_on_axis.to(torch.complex32)

    def compute_linear_phase(self):
        linear_phase = 2 * torch.pi * (self.fcx * self.xx + self.fcy * self.yy)

        return linear_phase

    def spectrum_parameters(self):
        xx_vertex, yy_vertex = torch.meshgrid(self.range_x, self.range_y, indexing="xy")
        rr_vertex = self.sign * torch.sqrt(
            (xx_vertex - self.xo) ** 2 + (yy_vertex - self.yo) ** 2 + self.zo**2
        )
        gradx = self.k * (xx_vertex - self.xo) / rr_vertex
        grady = self.k * (yy_vertex - self.yo) / rr_vertex
        gradx_min, gradx_max = torch.min(gradx), torch.max(gradx)
        grady_min, grady_max = torch.min(grady), torch.max(grady)

        fcx = (gradx_min + gradx_max) / 2 / (2 * torch.pi)
        fcy = (grady_min + grady_max) / 2 / (2 * torch.pi)

        fbx_wave = (gradx_max - gradx_min) / (2 * torch.pi)
        fby_wave = (grady_max - grady_min) / (2 * torch.pi)

        return fcx, fcy, fbx_wave, fby_wave


class ThinLens(Waves):
    """Class for thin lens representation"""

    def __init__(
        self,
        wvl,
        range_x,
        range_y,
        over_sf=50,
        fl=50 * mm,
        DEVICE="cpu",
        numerical_mode="double2single",
    ):
        """
        Parameters
        ----------
        wvl: wavelength
        range_x: x coordinate range of aperture
        range_y: y coordinate range of aperture
        theta_x: angle in x direction (degree)
        theta_y: angle in y direction (degree)
        fl: focal length of lens
        """
        super().__init__(wvl, range_x, range_y, DEVICE)
        self.over_sf = torch.as_tensor(over_sf, device=DEVICE)
        self.fl = torch.as_tensor(fl, device=DEVICE)
        self.numerical_mode = numerical_mode

        self.fcx, self.fcy, self.fbx_wave, self.fby_wave = self.spectrum_parameters()
        self.fbx, self.fby = (
            self.fbx_wave + self.over_sf * self.fbx_apt,
            self.fby_wave + self.over_sf * self.fby_apt,
        )

    def forward(self, sample_x=None, sample_y=None):
        if sample_x is None and sample_y is None:
            sample_x, sample_y = self.least_samples()

        self.sample_x, self.sample_y = sample_x, sample_y
        self.vec_x, self.vec_y, self.xx, self.yy = self.compute_coordinates()
        self.update_bandwidth()

        phase = -self.k * (self.xx**2 + self.yy**2) / (2 * self.fl)

        amp = torch.ones_like(phase)
        self.field = amp * torch.exp(1j * phase)
        self.field_on_axis = self.field

        if self.numerical_mode == "double2single":
            self.field = self.field.to(torch.complex64)
            self.field_on_axis = self.field_on_axis.to(torch.complex64)
        elif self.numerical_mode == "double2half":
            self.field = self.field.to(torch.complex32)
            self.field_on_axis = self.field_on_axis.to(torch.complex32)

    def spectrum_parameters(self):
        xx_vertex, yy_vertex = torch.meshgrid(self.range_x, self.range_y, indexing="xy")
        gradx = -self.k * xx_vertex / self.fl
        grady = -self.k * yy_vertex / self.fl
        gradx_min, gradx_max = torch.min(gradx), torch.max(gradx)
        grady_min, grady_max = torch.min(grady), torch.max(grady)

        fcx = (gradx_min + gradx_max) / 2 / (2 * torch.pi)
        fcy = (grady_min + grady_max) / 2 / (2 * torch.pi)

        fbx_wave = (gradx_max - gradx_min) / (2 * torch.pi)
        fby_wave = (grady_max - grady_min) / (2 * torch.pi)

        return fcx, fcy, fbx_wave, fby_wave


class SphericalWavePlusThinLens(Waves):
    def __init__(
        self,
        wvl,
        range_x,
        range_y,
        over_sf=50,
        theta_x=0,
        theta_y=0,
        zo=1 * m,
        fl=50.0 * mm,
        DEVICE="cpu",
        numerical_mode="double2single",
    ):
        """
        Parameters
        ----------
        wvl: wavelength
        range_x: x coordinate range of aperture
        range_y: y coordinate range of aperture
        theta_x: angle in x direction (degree)
        theta_y: angle in y direction (degree)
        zo: distance from source to aperture
        fl: focal length of lens
        """
        super().__init__(wvl, range_x, range_y, DEVICE)
        self.over_sf = torch.as_tensor(over_sf, device=DEVICE)
        self.theta_x = torch.as_tensor(theta_x, device=DEVICE)
        self.theta_y = torch.as_tensor(theta_y, device=DEVICE)
        self.zo = torch.as_tensor(zo, device=DEVICE)
        self.fl = torch.as_tensor(fl, device=DEVICE)
        self.numerical_mode = numerical_mode
        self.sign = 1 if self.zo < 0 else -1

        ro = self.zo / torch.sqrt(
            1 - torch.sin(self.theta_x) ** 2 - torch.sin(self.theta_y) ** 2
        )
        if torch.isnan(ro):
            raise ValueError("ro is NaN. Check incident angle for validity.")
        self.xo, self.yo = ro * torch.sin(self.theta_x), ro * torch.sin(self.theta_y)

        self.fcx, self.fcy, self.fbx_wave, self.fby_wave = self.spectrum_parameters()
        self.fbx, self.fby = (
            self.fbx_wave + self.over_sf * self.fbx_apt,
            self.fby_wave + self.over_sf * self.fby_apt,
        )

    def forward(self, sample_x=None, sample_y=None):
        if sample_x is None and sample_y is None:
            sample_x, sample_y = self.least_samples()

        self.sample_x, self.sample_y = sample_x, sample_y
        self.vec_x, self.vec_y, self.xx, self.yy = self.compute_coordinates()
        self.update_bandwidth()
        rationalize_r = True

        r = torch.sqrt((self.xx - self.xo) ** 2 + (self.yy - self.yo) ** 2 + self.zo**2)
        r_amp = r

        if rationalize_r:
            r_rationalized = ((self.xx - self.xo) ** 2 + (self.yy - self.yo) ** 2) / (
                torch.abs(self.zo) + r
            )
            r_phs = r_rationalized
            z_compensate = self.zo
        else:
            r_phs = r
            z_compensate = torch.tensor(0.0, dtype=self.real_dtype, device=self.device)

        phs_sph = self.k * self.sign * r_phs
        phs_lens = -self.k * (self.xx**2 + self.yy**2) / (2 * self.fl)
        phase = phs_sph + phs_lens

        amp = 1 / r_amp
        amp /= amp.max()
        self.field = (
            amp
            * torch.exp(1j * phase)
            * torch.exp(1j * self.k * self.sign * torch.abs(z_compensate))
        )
        self.linear_phase = self.compute_linear_phase()
        self.field_on_axis = self.field * torch.exp(-1j * self.linear_phase)

        if self.numerical_mode == "double2single":
            self.field = self.field.to(torch.complex64)
            self.field_on_axis = self.field_on_axis.to(torch.complex64)
        elif self.numerical_mode == "double2half":
            self.field = self.field.to(torch.complex32)
            self.field_on_axis = self.field_on_axis.to(torch.complex32)

    def compute_linear_phase(self):
        linear_phase = 2 * torch.pi * (self.fcx * self.xx + self.fcy * self.yy)

        return linear_phase

    def spectrum_parameters(self):
        xx_vertex, yy_vertex = torch.meshgrid(self.range_x, self.range_y, indexing="xy")
        rr_vertex = self.sign * torch.sqrt(
            (xx_vertex - self.xo) ** 2 + (yy_vertex - self.yo) ** 2 + self.zo**2
        )
        gradx = self.k * ((xx_vertex - self.xo) / rr_vertex - xx_vertex / self.fl)
        grady = self.k * ((yy_vertex - self.yo) / rr_vertex - yy_vertex / self.fl)
        gradx_min, gradx_max = torch.min(gradx), torch.max(gradx)
        grady_min, grady_max = torch.min(grady), torch.max(grady)

        fcx = (gradx_min + gradx_max) / 2 / (2 * torch.pi)
        fcy = (grady_min + grady_max) / 2 / (2 * torch.pi)

        fbx_wave = (gradx_max - gradx_min) / (2 * torch.pi)
        fby_wave = (grady_max - grady_min) / (2 * torch.pi)

        return fcx, fcy, fbx_wave, fby_wave


class GeneralWave(Waves):
    def __init__(
        self,
        wvl,
        range_x,
        range_y,
        DEVICE,
        amplitude,
        phase,
        numerical_mode="double2single",
        zo=None,
    ):
        super().__init__(wvl, range_x, range_y, DEVICE)

        self.device = DEVICE
        self.numerical_mode = numerical_mode
        self.over_sf = 10

        self.amplitude = amplitude
        self.phase = phase
        if self.numerical_mode == "double2single":
            self.amplitude = self.amplitude.to(torch.float32)
            self.phase = self.phase.to(torch.float32)
        elif self.numerical_mode == "double2half":
            self.amplitude = self.amplitude.to(torch.float16)
            self.phase = self.phase.to(torch.float16)

        self.sample_y, self.sample_x = self.amplitude.shape

        self.vec_x, self.vec_y, self.xx, self.yy = self.compute_coordinates()
        lx, ly = (
            range_x[1] - range_x[0],
            range_y[1] - range_y[0],
        )
        dx, dy = lx / (self.sample_x - 1), ly / (self.sample_y - 1)
        self.fcx = self.fcy = torch.as_tensor(0, dtype=self.real_dtype, device=DEVICE)
        self.fbx, self.fby = 1 / dx, 1 / dy

        if zo is not None:
            self.zo = torch.as_tensor(zo, dtype=self.real_dtype, device=DEVICE)

    def forward(self):
        self.field = self.field_on_axis = self.amplitude * torch.exp(1j * self.phase)
