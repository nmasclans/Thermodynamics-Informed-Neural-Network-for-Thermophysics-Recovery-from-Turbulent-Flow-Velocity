"""
Thermodynamics equations from article:
    Title:   Transcritical diffuse-interface hydrodynamics of propellants in 
             high-pressure combustors of chemical propulsion systems 
    Authors: Lluís Jofre, Javier Urzay
"""

import sys
import numpy as np

from thermodynamics.func_PengRobinson import PengRobinson
from thermodynamics.func_SubstanceLibrary import SubstanceLibrary

Ru = 8.314    # R universal [J / (K * mol)]

def SpecificHeatCapacities(bSolver, P, rho, T, Substance):

    """
        P, rho, T: scalar floats, not an array
    """

    MW    = SubstanceLibrary(Substance)[0]
    gamma = SubstanceLibrary(Substance)[7]
    R     = Ru/MW # R specific

    if bSolver == 'Real':
        
        a, b, R, dadT, d2adT2, NASA_coefficients = PengRobinson( T, Substance )

        ## -------------- Cp real gas -------------- 

        # Cp ideal gas, depending on temperature, from equation C.25, with variations because of coefficients dependency on temperature!
        if T >= 200 and T < 1000:
            c_p_ideal = R*(NASA_coefficients[7] + NASA_coefficients[8]*T + NASA_coefficients[9]*T**2 + NASA_coefficients[10]*T**3 + NASA_coefficients[11]*T**4)
        elif T >= 1000 and T < 6000:
            c_p_ideal = R*(NASA_coefficients[0] + NASA_coefficients[1]*T + NASA_coefficients[2]*T**2 + NASA_coefficients[3]*T**3 + NASA_coefficients[4]*T**4)
        elif T < 200:
            # Assume constant temperature below 200K
            c_p_ideal = R*(NASA_coefficients[7] + NASA_coefficients[8]*200 + NASA_coefficients[9]*200**2 + NASA_coefficients[10]*200**3 + NASA_coefficients[11]*200**4)
        else:
            sys.exit(f"Temperature T = {T:.3e} is too large, should be < 6000K.") 

        # Departure function Cp --> from Jofre and Urzay, appendix C.7
        # !!!!! ALERT: R specific should be Ru universal, or R0 as article notation!!!!!
        v       = 1/rho              # Molar volume
        Z       = P*v/(R*T)          # Compressibility factor, specific
        A       = a*P/(R*T)**2 
        B       = b*P/(R*T)
        M       = (Z**2 + 2*B*Z - B**2)/(Z - B)
        N       = dadT*(B/(b*R))
        dep_c_p = (R*(M - N)**2)/(M**2 - 2*A*(Z + B)) - (T*d2adT2/(2*np.sqrt(2)*b))*np.log((Z + (1 - np.sqrt(2))*B)/(Z + (1 + np.sqrt(2))*B)) - R

        # Cp real gas
        c_p     = c_p_ideal + dep_c_p

        ## -------------- Cv real gas -------------- 

        # Cv ideal gas
        c_v_ideal = c_p_ideal - R

        # Departure function Cv
        dep_c_v = -1*(T*d2adT2)/(2+np.sqrt(2)*b)*np.log((Z + (1 - np.sqrt(2))*B)/(Z + (1 + np.sqrt(2))*B))

        # Cv real gas
        c_v  = c_v_ideal + dep_c_v

    elif bSolver == "Ideal":

        # Ideal gas
        c_v   = R/(gamma - 1)
        c_p   = c_v*gamma

    else: 
        sys.exit(f"ErrorValue in bSolver = {bSolver}; admissible values of bSolver: 'Real', 'Ideal'")

    ## Gamma
    gamma = c_p / c_v

    return c_p, c_v, gamma
