'''
Yisheng Tu, Tanaji Sen, Jean-Francois Ostiguy
'''

import numpy as np


class quadrupole():
    def __init__(self, a, b, theta_0, mode, Vp = 1, Z0=376.730313667, max_val = 200):
        '''

        Solves the semi-analytic equation for quadrupole mode.

        :param a: (float) outer ring
        :param b: (float) inner ring
        :param theta_0: (float) angle
        :param Vp: (float) the voltage on the plate
        :param Z0: (float) the characteristic impedance constant
        :param max_val: (int) max value of the index solver goes to, don't need to specify when using projection method

        The functions are the same as dipole odd (dipole even) mode, refer to the functions with the same names for documentation.

        '''

        self.a = float(a)
        self.b = float(b)
        self.b_a = self.b / self.a
        self.theta_0 = theta_0

        self.Vp = Vp
        self.max_val = 4 * int(max_val)

        self.__quad_X = []     # solution to the system of equations
        self.__quad_X_f = []
        self.__quad_A_mn = []  # A_mn
        self.__quad_B_n = []   # B_n

        self.__sum_X = []     # solution to the system of equations
        self.__sum_X_f = []
        self.__sum_A_mn = []  # A_mn
        self.__sum_B_n = []   # B_n

        self.solve()

        self.__quad_char_impedance = Z0/(4 * abs(sum([self.__quad_X_f[i] * self.__g(4.0*i+2) * np.sin((4.0*i+2)*self.theta_0) for i in range(0, len(self.__quad_X_f), 1)])))
        self.__sum_char_impedance = 2 * Z0/np.pi * np.log(self.a/self.b)/abs(self.__sum_X_f[0])
        self.__geo_char_impedance = (self.__quad_char_impedance * self.__sum_char_impedance) ** 0.5

        self.mode = mode
        if self.mode == 'quad':
            self.X = self.__quad_X
            self.X_f = self.__quad_X_f
            self.A_mn = self.__quad_A_mn
            self.B_n = self.__quad_B_n
        elif self.mode == 'sum':
            self.X = self.__sum_X
            self.X_f = self.__sum_X_f
            self.A_mn = self.__sum_A_mn
            self.B_n = self.__sum_B_n
        else:  # self.mode != 'quad' and self.mode != 'sum':
            raise ValueError('mode must be \'sum\' or \'quad\'')

    def __filter_fun(self, sigma):
        sigma_f = 1.0/sigma
        return lambda x: np.exp(- x ** 2 / (2 * sigma_f ** 2))

    def __B(self, m, n):
        if m != n:
            return (1.0/(n-m))* np.sin((n-m)*self.theta_0) + (1.0/(n+m))* np.sin((n+m)*self.theta_0)
        else:
            return self.theta_0 + (1.0/(2*m)) * np.sin(2 * m * self.theta_0)

    def __g(self, m):
        '''
        computes equation 17 in paper

        :param m: (int) index
        :return: equation 17
        '''
        a, b = self.a, self.b
        return 1/(1-(b/a)**(2*m))

    def __quad_phi(self, r, theta, Cm_list = None, number_of_item=200):
        phi_tot = 0
        if Cm_list is None:
            Cm_list = self.__quad_X_f
        if r <= self.b:
            for i in np.linspace(0, len(Cm_list) - 1, len(Cm_list))[:number_of_item]:
                Cm = Cm_list[int(i)]
                m = i * 4 + 2
                phi_tot += Cm * (r / self.b) ** m * np.cos(m * theta)
        elif self. b <= r <= self.a:
            for i in np.linspace(0, len(Cm_list) - 1, len(Cm_list))[:number_of_item]:
                Cm = Cm_list[int(i)]
                m = i * 4 + 2
                phi_tot += Cm / (1 - (self.a / self.b) ** (2 * m)) * \
                           ((r / self.b) ** m - (self.a ** 2 / (self.b * r)) ** m) * np.cos(m * theta)
        else:
            return 0

        return self.Vp * phi_tot

    def __quad_phi_xy(self, x, y, Cm_list = None, number_of_item=200):
        r = (x ** 2 + y ** 2) ** 0.5
        if x >= 0 and y >= 0:
            theta = np.arctan(y/x)
        elif x <= 0:
            theta = np.pi + np.arctan(y/x)
        else:
            theta = 2*np.pi - np.arctan(y/x)
        return self.__quad_phi(r, theta, Cm_list, number_of_item)

    def __sum_phi(self, r, theta, Cm_list = None, number_of_item=200):
        phi_tot = 0
        if Cm_list is None:
            Cm_list = self.X_f
        if r <= self.b:
            for i in np.linspace(0, len(Cm_list) - 1, len(Cm_list))[:number_of_item]:
                Cm = Cm_list[int(i)]
                m = i * 4
                if i == 0:
                    phi_tot += Cm
                else:
                    phi_tot += Cm * (r / self.b) ** m * np.cos(m * theta)
        elif self.b <= r <= self.a:
            for i in np.linspace(0, len(Cm_list) - 1, len(Cm_list))[:number_of_item]:
                Cm = Cm_list[int(i)]
                m = i * 4
                if i == 0:
                    phi_tot += np.log(r/self.a)/np.log(self.b/self.a) * Cm
                else:
                    phi_tot += Cm / (1 - (self.a / self.b) ** (2 * m)) * \
                               ((r / self.b) ** m - ((self.a ** 2) / (self.b * r)) ** m) * np.cos(m * theta)
        else:
            return 0

        return self.Vp * phi_tot

    def __sum_phi_xy(self, x, y, Cm_list = None, number_of_item=200):
        r = (x ** 2 + y ** 2) ** 0.5
        if x >= 0 and y >= 0:
            theta = np.arctan(y/x)
        elif x <= 0:
            theta = np.pi + np.arctan(y/x)
        else:
            theta = 2*np.pi - np.arctan(y/x)

        return self.__sum_phi(r, theta, Cm_list, number_of_item)

    def phi(self, r, theta, Cm_list = None, number_of_item=200):
        '''

        Get the potential for the mode specified at (r, theta) in spherical coordinate

        :param r: radius
        :param theta: angle
        :param Xm_list: default leave it ``None''. The program will use X_f, the filtered coefficients. One could also
                        customize to any array one would like
        :param number_of_item: number of items in Xm_list to be used to compute phi
        :return: phi(r, theta)
        '''

        if self.mode == 'quad':
            return self.__quad_phi(r, theta, Cm_list, number_of_item)
        elif self.mode == 'sum':
            return self.__sum_phi(r, theta, Cm_list, number_of_item)

    def phi_xy(self, x, y, Cm_list = None, number_of_item=200):
        '''

        Get the potential phi at (x, y) for the specified mode in cartesian coordinate.

        :return: phi at(x, y)
        '''
        if self.mode == 'quad':
            return self.__quad_phi_xy(x, y, Cm_list, number_of_item)
        elif self.mode == 'sum':
            return self.__sum_phi_xy(x, y, Cm_list, number_of_item)

    def __quad_filter_fun(self, sigma):
        sigma_f = 1.0/sigma
        return lambda x: np.exp(- x ** 2 / (2 * sigma_f ** 2))

    def __quad_proj_get_Aij(self, i, j):
        '''

        Get the matrix element A_ij for matrix equation AX = B

        :param i: (int) row index
        :param j: (int) column index
        :return: (float) value of A_ij
        '''
        if i == j:
            n = i
            return lambda theta_0: (1 + 1.0 * n * (-1.0) * self.__g(n)) * self.__B(n, n) - (n * (np.pi / 4.0) * (-1.0) * self.__g(n))
        else:
            n = i
            m = j
            return lambda theta_0: (1 + m * (-1.0) * self.__g(m)) * self.__B(m, n)

    def __quad_proj_get_Bi(self, i):
        n = i
        return lambda theta_0: - 2.0/n * np.sin(n*theta_0)

    def __quad_proj_solve(self):
        # solve by QR algo
        self.A_mn = []
        self.B_n = []
        for i in range(2, self.max_val, 4):
            row = []
            for j in range(2, self.max_val, 4):
                row.append(self.__quad_proj_get_Aij(i, j)(self.theta_0))
            self.A_mn.append(row)
            self.B_n.append(self.__quad_proj_get_Bi(i)(self.theta_0))
        q, r = np.linalg.qr(self.A_mn, mode='complete')
        Qb = np.dot(q.T, self.B_n)

        self.__quad_X = np.linalg.solve(r, Qb)

    def __quad_solve(self):
        self.__quad_proj_solve()
        self.__quad_apply_filter_fun()

    def __quad_apply_filter_fun(self, sigma = 2.0 * np.pi / 100.0):
        index = 0
        while index < len(self.__quad_X):
            self.__quad_X_f.append(self.__quad_X[index] * self.__filter_fun(sigma=sigma)(4 * index + 2))
            index += 1

    def __sum_lstsq_get_A_mn(self, m, n):
        pass

    def __sum_lstsq_get_B_n(self, m):
        pass

    def __sum_lstsq_solve(self):
        pass

    def __sum_proj_get_Aij(self, i, j):
        '''

        Get the matrix element A_ij for matrix equation AX = B

        :param i: (int) row index
        :param j: (int) column index
        :return: (float) value of A_ij
        '''
        if i == j:
            n = i
            return lambda theta_0: 2 * np.log(self.b/self.a) * n * self.__g(n) * (self.__B(n, n) - np.pi/4.0) + self.__B(n, n)
        else:
            n = i
            m = j
            return lambda theta_0: (2 * np.log(self.b/self.a) * m * self.__g(m) + 1) * self.__B(m, n)

    def __sum_proj_get_Bi(self, i):
        n = i
        return lambda theta_0: 2.0/n * np.sin(n*self.theta_0)

    def __sum_proj_solve(self):
        # solve by QR algo
        self.A_mn = []
        self.B_n = []
        for i in range(4, self.max_val, 4):
            row = []
            for j in range(4, self.max_val, 4):
                row.append(self.__sum_proj_get_Aij(i, j)(self.theta_0))
            self.A_mn.append(row)
            self.B_n.append(self.__sum_proj_get_Bi(i)(self.theta_0))
        q, r = np.linalg.qr(self.A_mn, mode='complete')
        Qb = np.dot(q.T, self.B_n)

        self.__sum_X = np.linalg.solve(r, Qb)
        X0 = 1 - sum(self.__sum_X[i] * 1.0/(4 * i + 4) * np.sin((4 * i + 4) * self.theta_0) for i in range(0, len(self.__sum_X), 1)) * 1.0/self.theta_0
        self.__sum_X = [X0] + list(self.__sum_X)

    def __sum_apply_filter_fun(self, sigma = 2.0 * np.pi / 100.0):
        index = 0
        while index < len(self.__sum_X):
            self.__sum_X_f.append(self.__sum_X[index] * self.__filter_fun(sigma=sigma)(4 * index + 2))
            index += 1

    def __sum_solve(self):
        self.__sum_proj_solve()
        self.__sum_apply_filter_fun()

    def solve(self):
        ''' solves the matrix equation and apply the filter function'''
        self.__quad_solve()
        self.__sum_solve()

    def __characteristic_impedance_thickness(self, thickness, kf_sum, kf_geo):
        '''

        Get the estimated characteristic impedance for a specified thickness

        :param thickness: (float) thickness of the plate
        :param kf_sum: (float) the scaling coefficient of the sum mode
        :param kf_geo:  (float) the scaling coefficient of the geometric mode
        :return: (list), odd mode, sum mode, geometric mean (characteristic impedance)
        '''

        thick_sum_imp = self.__quad_char_impedance * np.log((self.a - thickness/kf_sum)/self.b) / np.log(self.a/self.b)
        thick_geo_imp = self.__geo_char_impedance * np.log((self.a - thickness/kf_geo)/self.b) / np.log(self.a/self.b)

        return None, thick_sum_imp, thick_geo_imp

    def get_characteristic_impedance(self, thickness = None, kf_sum = 3.7, kf_geo = 3.7):
        '''

        print the characteristic impedance

        :param thickness: if None or 0, then assume infinitesimal plate, else apply thickness;
                          when thickness !=0, potential and field are not available.
        :param kf_sum (float) the scaling coefficient of the sum mode
        :param kf_geo:  (float) the scaling coefficient of the geometric mode
        :return: None
        '''

        if thickness is None or thickness == 0.0:
            print ('a = ', self.a, ', b = ', self.b, ', theta_0 = ', self.theta_0, ', thickness = ', thickness)
            print ('quad mode: Z = ', self.__quad_char_impedance)
            print ('sum mode: Z = ', self.__sum_char_impedance())
            print ('geo (Z_q^2 + Z_s^2)^0.5: Z = ', self.__geo_char_impedance)
        else:
            Zquad, Zsum, Zgeo = self.__characteristic_impedance_thickness(thickness=thickness, kf_sum = kf_sum, kf_geo = kf_geo)
            print ('a = ', self.a, ', b = ', self.b, ', theta_0 = ', self.theta_0, ', thickness = ', thickness)
            # print ('quad mode: Z = ', self.__odd_char_impedance)
            print ('sum mode: Z = ', Zsum)
            print ('geo (Z_q^2 + Z_s^2)^0.5: Z = ', Zgeo)

    def change_mode(self, mode):
        '''

        change the mode

        :param mode: 'sum' or 'quad'
        :return: None
        '''

        if self.mode == 'sum':
            self.mode = mode
            self.X = self.__sum_X
            self.X_f = self.__sum_X_f
            self.A_mn = self.__sum_A_mn
            self.B_n = self.__sum_B_n
        elif self.mode == 'quad':
            self.mode = mode
            self.X = self.__quad_X
            self.X_f = self.__quad_X_f
            self.A_mn = self.__quad_A_mn
            self.B_n = self.__quad_B_n
        else:  # self.mode != 'sum' and self.mode != 'quad':
            raise ValueError('mode must be \'quad\' or \'sum\'')


