import numpy as np

from thermodynamics.func_SubstanceLibrary import SubstanceLibrary

Ru = 8.314    # R universal

def PengRobinson( T, Substance ):

    # getThermo Compute necessary thermodyamic parameters for the cubic
    # equation of state (N2 property is assumed).

    # N2 substance library
    MW, Tc, pc, _, _, _, omega, _, _, _, NASA_coefficients, _, _, _, _, _, _, _ = SubstanceLibrary(Substance)

    # Peng Robsinson coefficients
    if omega > 0.49:            # Accentric factor
        c      = 0.379642 + 1.48503*omega - 0.164423*(omega**2) + 0.016666*(omega**3)
    else:
        c      = 0.37464 + 1.54226*omega - 0.26992*(omega**2)

    R      = Ru/MW              # R specific
    a      = (0.457236*(R*Tc)**2/pc)*(1+c*(1-np.sqrt(T/Tc)))**2
    b      = 0.077796*R*Tc/pc
    G      = c*np.sqrt(T/Tc)/(1+c*(1-np.sqrt(T/Tc)))
    dadT   = -(1/T)*a*G
    d2adT2 = 0.457236*R**2/T/2*c*(1+c)*Tc/pc*np.sqrt(Tc/T)

    return a, b, R, dadT, d2adT2, NASA_coefficients
