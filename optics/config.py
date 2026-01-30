from optics.constant import m, nm


############################ system parameters ############################
WVL = 550 * nm  # Wavelength of light in vacuum

FL = 1 * m  # Focal length of lens
FN = 16  # F-number of lens
zf = 7 * m  # Lens-sensor distance
r = FL / FN / 2  # Radius of aperture
fd = 1 / (1 / FL - 1 / zf)  # Focal distance of lens
dl = WVL * zf / r / 2

print("\nSystem parameters:")
print(f"Wavelength (WVL): {WVL / nm:.2f} nm")
print(f"Lens-sensor distance (zf): {zf / m:.2f} m")
print(f"Focal distance (fd): {fd / m:.2f} m")
