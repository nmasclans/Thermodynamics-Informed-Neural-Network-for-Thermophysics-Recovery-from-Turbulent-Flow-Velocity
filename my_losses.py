import tensorflow as tf
import sys
import time

class Supervised_PINNS(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="Supervised_PINNS")
        self.loss_weights = args.Supervised_PINNS_weights
        self.loss_weights_first_epoch = args.Supervised_PINNS_weights_first_epoch
        assert sum(self.loss_weights) == 1.0

        # ---- For calculating Relative Errors in Physics Equations ----
        self.c_p_min    = args.targets_limits['c_p'][0]
        self.c_p_max    = args.targets_limits['c_p'][1] 
        self.rho_min    = args.targets_limits['rho'][0]
        self.rho_max    = args.targets_limits['rho'][1] 
        self.T_min      = args.targets_limits['T'][0]
        self.T_max      = args.targets_limits['T'][1] 
        self.min_value  = args.min_value
        self.max_value  = args.max_value
        self.eps        = args.eps
        self.P_constant = args.P_constant
        if args.Substance == 'N2':
            self.args  = args
            # ----------------- N2 -
            self.Ru    = 8.314            # R universal
            self.MW    = 2.80134e-2       # Molecular weight kg/mol
            self.R     = self.Ru/self.MW  # R specific
            self.Tc    = 126.19           # Critical temperature [k]
            self.pc    = 3.3958e+6;       # Critical pressure [Pa]
            self.omega = 0.03720          # Acentric factor
            self.NASA_coefficients =  [ 2.952576370000000000000, 0.001396900400000000000, -0.000000492631603000000, 0.000000000078601019000,
                                       -0.000000000000004607552, -923.9486880000000000000, 5.871887620000000000000, 3.531005280000000000000,
                                       -0.000123660980000000000, -0.000000502999433000000, 0.000000002435306120000, -0.000000000001408812400,
                                       -1046.976280000000000000, 2.967470380000000000000, 0.000000000000000000000]
            if self.omega > 0.49:         # Accentric factor
                self.c = 0.379642 + 1.48503*self.omega - 0.164423*(self.omega**2) + 0.016666*(self.omega**3)
            else:
                self.c = 0.37464 + 1.54226*self.omega - 0.26992*(self.omega**2)
            self.b = 0.077796*self.R*self.Tc/self.pc
        else:
            sys.exit("Not implemented Substance. Set Substance to 'N2'")

    @tf.function #(input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                  tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, epoch):
        
        # Add epsilon value to predicted values to prevent NaN values in RE-PINNS losses (when predicted values / model output = 0):
        y_pred += self.eps
        y_gt += self.eps

        # ----------------------- Supervised loss: RAE --------------------------
        # loss_Supervised = tf.reduce_mean(tf.math.abs((y_gt - y_pred)/y_gt)) # RAE
        loss_Supervised = tf.reduce_mean(tf.square(y_gt - y_pred))            # MSE

        # ------------------------- Unsupervised loss --------------------------
        c_p_scaled = y_pred[:,0]
        rho_scaled = y_pred[:,1]
        T_scaled   = y_pred[:,2]    
        c_p = (c_p_scaled - self.min_value) * (self.c_p_max - self.c_p_min) / (self.max_value - self.min_value) - self.c_p_min
        rho = (rho_scaled - self.min_value) * (self.rho_max - self.rho_min) / (self.max_value - self.min_value) - self.rho_min
        T   = (T_scaled   - self.min_value) * (self.T_max   - self.T_min)   / (self.max_value - self.min_value) - self.T_min
        v    = 1/rho

        # ------ Peng Robinson ------
        a      = (0.457236*(self.R*self.Tc)**2/self.pc) * tf.pow(1+self.c*(1-tf.math.sqrt(T/self.Tc)), 2)
        G      = self.c*tf.math.sqrt(T/self.Tc) / (1+self.c*(1-tf.math.sqrt(T/self.Tc)))
        dadT   = -(1/T)*a*G
        d2adT2 = 0.457236*self.R**2 / T / 2 * self.c * (1+self.c) * self.Tc / self.pc * tf.math.sqrt(self.Tc/T)

        # ------ Unsup. PINNS loss 1: RE of Equation of State of Real Gas ------
        P = self.R * T / (v - self.b) - a / (tf.math.pow(v,2) + 2*self.b*v - self.b**2)
        
        # Relative error on the Equation of State of Real Gas:
        loss_RE_StateRealGas = tf.reduce_mean(tf.math.abs((P-self.P_constant)/self.P_constant))
        # tf.print("Unsupervised PINNS loss of RE Real Gas State Equation:",loss_RE_StateRealGas)

        # ------ Unsup. PINNS loss 2: RE of Equation of  Specific Heat Capacity ------
        # Cp ideal gas, depending on temperature, from equation C.25, with variations because of coefficients dependency on temperature!
        # assert tf.math.reduce_all(tf.math.less_equal(T,200),T)
        # if T >= 200 and T < 1000:
        #     c_p_ideal = R*(self.NASA_coefficients[7] + self.NASA_coefficients[8]*T + self.NASA_coefficients[9]*T**2 + self.NASA_coefficients[10]*T**3 + self.NASA_coefficients[11]*T**4)
        # elif T >= 1000 and T < 6000:
        #     c_p_ideal = R*(self.NASA_coefficients[0] + self.NASA_coefficients[1]*T + self.NASA_coefficients[2]*T**2 + self.NASA_coefficients[3]*T**3 + self.NASA_coefficients[4]*T**4)
        # elif T < 200:
        #    # Assume constant temperature below 200K
        c_p_ideal = self.R*(self.NASA_coefficients[7] + self.NASA_coefficients[8]*200 + self.NASA_coefficients[9]*200**2 + self.NASA_coefficients[10]*200**3 + self.NASA_coefficients[11]*200**4)
        # else:
        #     sys.exit(f"Temperature T = {T:.3e} is too large, should be < 6000K.") 

        # Departure function Cp --> from Jofre and Urzay, appendix C.7
        Z       = self.P_constant*v/(self.R*T)                  # Compressibility factor, specific
        A       = a*self.P_constant/tf.math.pow(self.R*T,2) 
        B       = self.b*self.P_constant/(self.R*T)
        M       = (tf.math.pow(Z,2) + 2*B*Z - tf.math.pow(B,2))/(Z - B)
        N       = dadT*B/(self.b*self.R)
        sqrt_2  = tf.math.sqrt(2.0)
        c_p_dep = (self.R*tf.math.pow(M - N,2))/(tf.math.pow(M,2) - 2*A*(Z + B)) - (T*d2adT2/(2*sqrt_2*self.b))*tf.math.log((Z + (1 - sqrt_2)*B)/(Z + (1 + sqrt_2)*B)) - self.R

        # Cp real gas
        c_p_equation = c_p_ideal + c_p_dep

        # Relatife error between predicted c_p vs Real Gas c_p Equation (depending on T)
        loss_RE_CpEq = tf.reduce_mean(tf.math.abs((c_p-c_p_equation)/c_p_equation))

        # ------------------------------------ Total Loss -----------------------------------
        if epoch == 0:
            loss_supervised_PINNS = self.loss_weights_first_epoch[0]*loss_Supervised + self.loss_weights_first_epoch[1]*loss_RE_StateRealGas + self.loss_weights_first_epoch[2]*loss_RE_CpEq
            tf.print('loss_weights_first_epoch used')
        else: 
            loss_supervised_PINNS = self.loss_weights[0]*loss_Supervised + self.loss_weights[1]*loss_RE_StateRealGas + self.loss_weights[2]*loss_RE_CpEq
            tf.print('loss_weights (not first_epoch) used')

        return loss_supervised_PINNS


class MSE(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="MSE")

    @tf.function # (input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, **kwargs):
        return tf.reduce_mean(tf.square(y_gt - y_pred))


class RSE(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RSE")
        self.eps       = args.eps

    @tf.function # (input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, **kwargs):
        y_gt += self.eps
        return tf.reduce_mean(tf.square(y_gt - y_pred)/tf.square(y_gt)) 


class RAE(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RAE")
        self.eps       = args.eps

    @tf.function # (input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, **kwargs):
        y_gt += self.eps
        return tf.reduce_mean(tf.math.abs((y_gt - y_pred)/y_gt)) 

class RAE_target_0(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RAE_target_0")
        self.eps = args.eps

    @tf.function # (input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, **kwargs):
        y_gt += self.eps
        return tf.reduce_mean(tf.math.abs((y_gt[:,0] - y_pred[:,0])/y_gt[:,0])) 

class RAE_target_1(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RAE_target_1")
        self.eps = args.eps

    @tf.function # (input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, **kwargs):
        y_gt += self.eps
        return tf.reduce_mean(tf.math.abs((y_gt[:,1] - y_pred[:,1])/y_gt[:,1])) 

class RAE_target_2(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RAE_target_2")
        self.eps = args.eps

    @tf.function # (input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, **kwargs):
        y_gt += self.eps
        return tf.reduce_mean(tf.math.abs((y_gt[:,2] - y_pred[:,2])/y_gt[:,2])) 


class RelError_RealGasEquation(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RE_RealGasEq")
        self.rho_min   = args.targets_limits['rho'][0]
        self.rho_max   = args.targets_limits['rho'][1] 
        self.T_min     = args.targets_limits['T'][0]
        self.T_max     = args.targets_limits['T'][1] 
        self.min_value = args.min_value
        self.max_value = args.max_value
        self.eps       = args.eps
        if args.Substance == 'N2':
            self.args  = args
            # ----------------- N2 -
            self.Ru    = 8.314            # R universal
            self.MW    = 2.80134e-2       # Molecular weight kg/mol
            self.R     = self.Ru/self.MW  # R specific
            self.Tc    = 126.19           # Critical temperature [k]
            self.pc    = 3.3958e+6;       # Critical pressure [Pa]
            self.omega = 0.03720          # Acentric factor
            self.NASA_coefficients =  [ 2.952576370000000000000, 0.001396900400000000000, -0.000000492631603000000, 0.000000000078601019000,
                                       -0.000000000000004607552, -923.9486880000000000000, 5.871887620000000000000, 3.531005280000000000000,
                                       -0.000123660980000000000, -0.000000502999433000000, 0.000000002435306120000, -0.000000000001408812400,
                                       -1046.976280000000000000, 2.967470380000000000000, 0.000000000000000000000]
            if self.omega > 0.49:         # Accentric factor
                self.c = 0.379642 + 1.48503*self.omega - 0.164423*(self.omega**2) + 0.016666*(self.omega**3)
            else:
                self.c = 0.37464 + 1.54226*self.omega - 0.26992*(self.omega**2)
            self.b = 0.077796*self.R*self.Tc/self.pc
            self.P_constant = args.P_constant
        else:
            sys.exit("Not implemented Substance. Set Substance to 'N2'")

    @tf.function # (input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, **kwargs):
        rho_scaled = y_pred[:,1]
        T_scaled   = y_pred[:,2]    
        rho = (rho_scaled - self.min_value) * (self.rho_max - self.rho_min) / (self.max_value - self.min_value) - self.rho_min
        T   = (T_scaled   - self.min_value) * (self.T_max   - self.T_min)   / (self.max_value - self.min_value) - self.T_min
        rho += self.eps # in case rho = 0, prevent v = inf, rel_err = nan
        T   += self.eps # in case T   = 0, prevent dadT = inf, rel_err = nan
        v    = 1/rho

        # -------------------------- Peng Robinson -------------------------
        a      = (0.457236*(self.R*self.Tc)**2/self.pc) * tf.pow(1+self.c*(1-tf.math.sqrt(T/self.Tc)), 2)

        # ----------------------- Equation of Real Gas ----------------------
        P = self.R * T / (v - self.b) - a / (tf.math.pow(v,2) + 2*self.b*v - self.b**2)
        
        # Relative error on the Equation of Real Gas:
        rel_err_real_gas = tf.reduce_mean(tf.math.abs((P-self.P_constant)/self.P_constant))
        return rel_err_real_gas


class RelError_CpEquation(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RE_CpEquation")
        self.c_p_min   = args.targets_limits['c_p'][0]
        self.c_p_max   = args.targets_limits['c_p'][1] 
        self.rho_min   = args.targets_limits['rho'][0]
        self.rho_max   = args.targets_limits['rho'][1] 
        self.T_min     = args.targets_limits['T'][0]
        self.T_max     = args.targets_limits['T'][1] 
        self.min_value = args.min_value
        self.max_value = args.max_value
        self.eps       = args.eps
        if args.Substance == 'N2':
            self.args  = args
            # ----------------- N2 -
            self.Ru    = 8.314            # R universal
            self.MW    = 2.80134e-2       # Molecular weight kg/mol
            self.R     = self.Ru/self.MW  # R specific
            self.Tc    = 126.19           # Critical temperature [k]
            self.pc    = 3.3958e+6;       # Critical pressure [Pa]
            self.omega = 0.03720          # Acentric factor
            self.NASA_coefficients =  [ 2.952576370000000000000, 0.001396900400000000000, -0.000000492631603000000, 0.000000000078601019000,
                                       -0.000000000000004607552, -923.9486880000000000000, 5.871887620000000000000, 3.531005280000000000000,
                                       -0.000123660980000000000, -0.000000502999433000000, 0.000000002435306120000, -0.000000000001408812400,
                                       -1046.976280000000000000, 2.967470380000000000000, 0.000000000000000000000]
            if self.omega > 0.49:         # Accentric factor
                self.c = 0.379642 + 1.48503*self.omega - 0.164423*(self.omega**2) + 0.016666*(self.omega**3)
            else:
                self.c = 0.37464 + 1.54226*self.omega - 0.26992*(self.omega**2)
            self.b = 0.077796*self.R*self.Tc/self.pc
            self.P_constant = args.P_constant
        else:
            sys.exit("Not implemented Substance. Set Substance to 'N2'")
    
    @tf.function # (input_signature=(tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(None,args.num_targets), dtype=tf.float32)))
    def call(self, y_gt, y_pred, **kwargs):
        c_p_scaled = y_pred[:,0]
        rho_scaled = y_pred[:,1]
        T_scaled   = y_pred[:,2]    

        c_p = (c_p_scaled - self.min_value) * (self.c_p_max - self.c_p_min) / (self.max_value - self.min_value) - self.c_p_min
        rho = (rho_scaled - self.min_value) * (self.rho_max - self.rho_min) / (self.max_value - self.min_value) - self.rho_min
        T   = (T_scaled   - self.min_value) * (self.T_max   - self.T_min)   / (self.max_value - self.min_value) - self.T_min
        rho += self.eps # in case rho = 0, prevent v = inf, rel_err = nan
        T   += self.eps # in case T   = 0, prevent dadT = inf, rel_err = nan
        c_p += self.eps
        v   = 1/rho

        # -------------------------- Peng Robinson -------------------------
        a      = (0.457236*(self.R*self.Tc)**2/self.pc) * tf.pow(1+self.c*(1-tf.math.sqrt(T/self.Tc)), 2)
        G      = self.c*tf.math.sqrt(T/self.Tc) / (1+self.c*(1-tf.math.sqrt(T/self.Tc)))
        dadT   = -(1/T)*a*G
        d2adT2 = 0.457236*self.R**2 / T / 2 * self.c * (1+self.c) * self.Tc / self.pc * tf.math.sqrt(self.Tc/T)

        # --------------- Equation of  Specific Heat Capacity ---------------
        # Cp ideal gas, depending on temperature, from equation C.25, with variations because of coefficients dependency on temperature!
        # assert tf.math.reduce_all(tf.math.less_equal(T,200),T)
        # if T >= 200 and T < 1000:
        #     c_p_ideal = R*(self.NASA_coefficients[7] + self.NASA_coefficients[8]*T + self.NASA_coefficients[9]*T**2 + self.NASA_coefficients[10]*T**3 + self.NASA_coefficients[11]*T**4)
        # elif T >= 1000 and T < 6000:
        #     c_p_ideal = R*(self.NASA_coefficients[0] + self.NASA_coefficients[1]*T + self.NASA_coefficients[2]*T**2 + self.NASA_coefficients[3]*T**3 + self.NASA_coefficients[4]*T**4)
        # elif T < 200:
        #    # Assume constant temperature below 200K
        c_p_ideal = self.R*(self.NASA_coefficients[7] + self.NASA_coefficients[8]*200 + self.NASA_coefficients[9]*200**2 + self.NASA_coefficients[10]*200**3 + self.NASA_coefficients[11]*200**4)
        # else:
        #     sys.exit(f"Temperature T = {T:.3e} is too large, should be < 6000K.") 

        # Departure function Cp --> from Jofre and Urzay, appendix C.7
        Z       = self.P_constant*v/(self.R*T)                  # Compressibility factor, specific
        A       = a*self.P_constant/tf.math.pow(self.R*T,2) 
        B       = self.b*self.P_constant/(self.R*T)
        M       = (tf.math.pow(Z,2) + 2*B*Z - tf.math.pow(B,2))/(Z - B)
        N       = dadT*B/(self.b*self.R)
        sqrt_2  = tf.math.sqrt(2.0)
        dep_c_p = (self.R*tf.math.pow(M - N,2))/(tf.math.pow(M,2) - 2*A*(Z + B)) - (T*d2adT2/(2*sqrt_2*self.b))*tf.math.log((Z + (1 - sqrt_2)*B)/(Z + (1 + sqrt_2)*B)) - self.R

        # Cp real gas
        c_p_equation = c_p_ideal + dep_c_p

        # Relatife error between predicted c_p vs Real Gas c_p Equation (depending on T)
        rel_err_c_p_equation = tf.reduce_mean(tf.math.abs((c_p-c_p_equation)/c_p_equation))
        return rel_err_c_p_equation
