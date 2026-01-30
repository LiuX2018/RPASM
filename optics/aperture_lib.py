"""This module contains various of apertures."""

import torch


class Aperture:
    """Base class for all apertures"""

    def __init__(self, range_x, range_y, sample_x, sample_y, DEVICE):
        """
        Parameters
        ----------
        range_x: x coordinate range of aperture
        range_y: y coordinate range of aperture
        sample_x: number of samples along x direction
        sample_y: number of samples along y direction
        """

        vec_x = torch.linspace(range_x[0], range_x[-1], sample_x, device=DEVICE)
        vec_y = torch.linspace(range_y[0], range_y[-1], sample_y, device=DEVICE)

        # Cartesian coordinates
        self.xx, self.yy = torch.meshgrid(vec_x, vec_y, indexing="xy")

        # Polar coordinates
        self.phi, self.rho = (
            torch.atan2(self.xx, self.yy),
            torch.sqrt(self.xx**2 + self.yy**2),
        )


class CircAperture(Aperture):
    """Circular aperture function"""

    def __init__(self, range_x, range_y, sample_x, sample_y, xo, yo, radius, DEVICE):
        """
        Parameters
        ----------
        range_x: x coordinate range of aperture
        range_y: y coordinate range of aperture
        sample_x: number of samples along x direction
        sample_y: number of samples along y direction
        xo: x coordinate of the center of the circular aperture
        yo: y coordinate of the center of the circular aperture
        radius: radius of the circular aperture
        """

        super().__init__(range_x, range_y, sample_x, sample_y, DEVICE)
        self.xo, self.yo = xo, yo
        self.radius = radius

    def __call__(self, field):
        mask = (
            torch.sqrt((self.xx - self.xo) ** 2 + (self.yy - self.yo) ** 2)
            > self.radius
        )
        field_new = torch.where(mask, torch.zeros_like(field), field)

        return field_new
