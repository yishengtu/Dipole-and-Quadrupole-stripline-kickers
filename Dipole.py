'''
Yisheng Tu, Tanaji Sen, Jean-Francois Ostiguy
'''

import numpy as np


class dipole():
    def __init__(self, a, b, theta_0, mode, Vp = 1, Z0=376.730313667, max_val = 200):
        '''

        Solves for the dipole mode, when called, the object constructs the matrix equations specified by the kicker parameters for 
        both even and sum mode. Then automatically solves the matrix equation (result stores in self.X) and apply a filter function to smooth
        the result (stores in self.X_f). method Phi_*() and E_*() can be called to compute the electric field/electric potential
        at any location on the kicker plate.

        :param a: (float) outer ring
        :param b: (float) inner ring
        :param theta_0: (float) angle
        :param Vp: (float) voltage of the plate, default 1V
        :param Z0: (float) a constant in the characteristic impedance expression
        :param max_val: (int) the maximum number of terms solves for
        '''
        
        self.a = float(a)       # the radius of the beampipe
        self.b = float(b)       # the radius of the arc plates
        self.theta_0 = theta_0  # half of the cover angle of each arc plate

        self.Vp = Vp                         # The potential on each plate
        self.max_val = 2 * int(max_val) + 1  # the maximum index of the efficient

        self.__odd_X = []     # solution to the system of equations
        self.__odd_X_f = []   # filtered coefficients (filtered solution to the system of equations)
        self.__odd_A_mn = []  # A_mn in matrix equation
        self.__odd_B_n = []   # B_n in matrix equation

        self.__even_X = []     # solution to the system of equations
        self.__even_X_f = []   # filtered coefficients (filtered solution to the system of equations)
        self.__even_A_mn = []  # A_mn in matrix equation
        self.__even_B_n = []   # B_n in matrix equation

        self.solve()

        self.__odd_char_impedance = Z0/(4 * abs(sum([self.__odd_X[i] * self.__g(2.0 * i + 1) * np.sin((2.0*i+1)*self.theta_0) for i in range(0, len(self.__odd_X), 1)])))  # calculate the characteristic impedance
        self.__even_char_impedance = Z0/np.pi * np.log(self.a/self.b)/abs(self.__even_X_f[0])  # calculate the characteristic impedance
        self.__geo_char_impedance = (self.__odd_char_impedance * self.__even_char_impedance)**0.5

        self.mode = mode
        if self.mode == 'odd':
            self.X = self.__odd_X
            self.X_f = self.__odd_X_f
            self.A_mn = self.__odd_A_mn
            self.B_n = self.__odd_B_n
        elif self.mode == 'even':
            self.X = self.__even_X
            self.X_f = self.__even_X_f
            self.A_mn = self.__even_A_mn
            self.B_n = self.__even_B_n
        else:  # self.mode != 'odd' and self.mode != 'even':
            raise ValueError('mode must be \'even\' or \'odd\'')

    def __filter_fun(self, sigma):
        '''

        The filter function, using a gaussian function: e^(-x^2/(2*sigma^2))

        :param sigma: (float), a parameter in gaussian function
        :return: the filter function, with a parameter x (float), the original value to be filtered.
        '''

        sigma_f = 1.0/sigma
        return lambda x: np.exp(- x ** 2 / (2 * sigma_f ** 2))

    def __B(self, m, n):
        ''' One expression in the matrix equation '''
        if m != n:
            return (1.0/(n-m))* np.sin((n-m)*self.theta_0) + (1.0/(n+m))* np.sin((n+m)*self.theta_0)
        else:
            return self.theta_0 + (1.0/(2*m)) * np.sin(2 * m * self.theta_0)

    def __g(self, m):
        '''
        One expression in the matrix equation
        '''
        a, b = self.a, self.b
        return 1/(1-(b/a)**(2*m))

    def __odd_phi(self, r, theta, Xm_list = None, number_of_item=200):
        '''

        computes the potential at a position (r, theta) for the odd mode

        :param r: (float) r-value in polar coordinate
        :param theta: (float) theta-value in polar coordinate
        :param Xm_list: (list) the list of solution, default using the X_f in this object.
        :param number_of_item: (int) number of items to be calculated the infinite series of phi. This value may cause
                overflow problem
        :return: potential at location (r, theta)
        '''
        if Xm_list is None:
            Xm_list = self.__odd_X_f
        phi_tot = 0
        if r <= self.b:
            for i in np.linspace(0, len(Xm_list) - 1, len(Xm_list))[:number_of_item]:
                Cm = Xm_list[int(i)]
                m = float(2 * i + 1)
                phi_tot += Cm * (r / self.b) ** m * np.cos(m * theta)
        elif r <= self.a:
            for i in np.linspace(0, len(Xm_list) - 1, len(Xm_list))[:number_of_item]:
                Cm = Xm_list[int(i)]
                m = float(2 * i + 1)
                phi_tot += Cm / (1 - (self.a / self.b) ** (2 * m)) * \
                           ((r / self.b) ** m - (self.a ** 2 / (self.b * r)) ** m) * np.cos(m * theta)
        else:
            return 0

        return self.Vp * phi_tot

    def __odd_phi_xy(self, x, y, Xm_list = None, number_of_item=200):
        '''

        computes the potential at a position (x, y) for the odd mode

        :param x: (float) x-value in polar coordinate
        :param y: (float) y-value in polar coordinate
        :param Xm_list: (list) the list of solution, default using the X_f in this object.
        :param number_of_item: (int) number of items to be calculated the infinite series of phi. This value may cause
                overflow problem
        :return: potential at location (x, y)
        '''

        r = (x ** 2 + y ** 2) ** 0.5
        if x >= 0 and y >= 0:
            theta = np.arctan(y/x)
        elif x <= 0:
            theta = np.pi + np.arctan(y/x)
        else:
            theta = 2*np.pi - np.arctan(y/x)
        return self.__odd_phi(r, theta, Xm_list, number_of_item)

    def __even_phi(self, r, theta, Xm_list = None, number_of_item=200):
        '''

        computes the potential at a position (r, theta) for the even mode

        :param r: (float) r-value in polar coordinate
        :param theta: (float) theta-value in polar coordinate
        :param Xm_list: (list) the list of solution, default using the X_f in this object.
        :param number_of_item: (int) number of items to be calculated the infinite series of phi. This value may cause
                overflow problem
        :return: potential at location (r, theta)
        '''

        if Xm_list is None:
            Xm_list = self.__even_X_f
        phi_tot = 0
        if r <= self.b:
            for i in np.linspace(0, len(Xm_list) - 1, len(Xm_list))[:number_of_item]:
                Cm = Xm_list[int(i)]
                m = float(2 * i)
                if int(m) == 0:
                    phi_tot += Cm
                else:
                    phi_tot += Cm * (r/self.b) ** m * np.cos(m * theta)
        elif self.b < r <= self.a:
            for i in np.linspace(0, len(Xm_list) - 1, len(Xm_list))[:number_of_item]:
                Cm = Xm_list[int(i)]
                m = float(2 * i)
                if int(m) == 0:
                    phi_tot += Cm * np.log(r/self.a)/np.log(self.b/self.a)
                else:
                    phi_tot += Cm/(1.0 - (self.a/self.b) ** (2 * m)) * ((r/self.b) ** m - (self.a ** 2 / (self.b * r)) ** m) * np.cos(m * theta)
        else:
            return 0

        return self.Vp * phi_tot

    def __even_phi_xy(self, x, y, Xm_list = None, number_of_item = 200):
        '''

        computes the potential at a position (x, y) for the even mode

        :param x: (float) x-value in polar coordinate
        :param y: (float) y-value in polar coordinate
        :param Xm_list: (list) the list of solution, default using the X_f in this object.
        :param number_of_item: (int) number of items to be calculated the infinite series of phi. This value may cause
                overflow problem
        :return: potential at location (x, y)
        '''
        r = (x ** 2 + y ** 2) ** 0.5
        theta = np.arctan(y/x)
        return self.__even_phi(r, theta, Xm_list, number_of_item)

    def phi(self, r, theta, Xm_list = None, number_of_item=200):
        '''
        
        Get the potential for the mode specified at (r, theta) in spherical coordinate
        
        :param r: radius
        :param theta: angle
        :param Xm_list: default leave it ``None''. The program will use X_f, the filtered coefficients. One could also 
                        customize to any array one would like
        :param number_of_item: number of items in Xm_list to be used to compute phi
        :return: phi(r, theta)
        '''

        if self.mode == 'odd':
            return self.__odd_phi(r, theta, Xm_list, number_of_item)
        elif self.mode == 'even':
            return self.__even_phi(r, theta, Xm_list, number_of_item)

    def phi_xy(self, x, y, Xm_list = None, number_of_item=200):
        '''

        Get the potential phi at (x, y) for the specified mode in cartesian coordinate.

        :return: phi at(x, y)
        '''

        if self.mode == 'odd':
            return self.__odd_phi_xy(x, y, Xm_list, number_of_item)
        elif self.mode == 'even':
            return self.__even_phi_xy(x, y, Xm_list, number_of_item)

    def __odd_proj_get_Aij(self, i, j):
        '''

        Get the matrix element A_ij for the projection method for odd mode

        :param i: (int) row index
        :param j: (int) column index
        :return: (float) value of A_ij
        '''
        if i == j:
            n = i
            return lambda theta_0: (1 + 1.0 * n * (-1.0) * self.__g(n)) * self.__B(n, n) - (n * (np.pi / 2.0) * (-1.0) * self.__g(n))
        else:
            n = i
            m = j
            return lambda theta_0: (1 + m * (-1.0) * self.__g(m)) * self.__B(m, n)

    def __odd_proj_get_Bi(self, i):
        '''
        computes matrix element for the projection method for odd mode

        :param i: (int) i in matrix A
        :return: element at location (m, n) of the matrix
        '''
        n = i
        return lambda theta_0: - 2.0/n * np.sin(n*theta_0)

    def __odd_proj_solve(self):
        '''
        Solves the system of matrix equation for odd mode
        updates A_mn: the matrix element of the LHS of the matrix equation
        updates B_m: the matrix element of the RHS of the matrix equation
        updates X: solution of the system of equations

        The function could be modified to solve the matrix equation using other algorithms,
        such as QR decomposition or SVD algorithm.

        :return: None
        '''

        self.__odd_A_mn = []
        self.__odd_B_n = []
        for i in range(1, self.max_val, 2):
            row = []
            for j in range(1, self.max_val, 2):
                row.append(self.__odd_proj_get_Aij(i, j)(self.theta_0))
            self.__odd_A_mn.append(row)
            self.__odd_B_n.append(self.__odd_proj_get_Bi(i)(self.theta_0))
        q, r = np.linalg.qr(self.__odd_A_mn, mode='complete')

        Qb = np.dot(q.T, self.__odd_B_n)

        self.__odd_X = np.linalg.solve(r, Qb)

    def __odd_solve(self):
        '''
        solve the system of equations for odd mode, then apply the filter function
        '''

        self.__odd_proj_solve()
        self.__odd_apply_filter_fun()

    def __odd_apply_filter_fun(self, sigma = 2.0 * np.pi / 100.0):
        '''

        Apply the filter function to the solution of system of equations (self.X) and stores the result in self.X_f

        :param sigma: (float), see self.__filter_fun for more detail
        :return: None
        '''
        index = 0
        while index < len(self.__odd_X):
            self.__odd_X_f.append(self.__odd_X[index] * self.__filter_fun(sigma=sigma)(2 * index + 1))
            index += 1

    def __even_proj_get_Aij(self, i, j):
        '''

        Get the matrix element A_ij for matrix equation AX = B, using projection method

        :param i: (int) row index
        :param j: (int) column index
        :return: (float) value of A_ij
        '''
        if i == j:
            n = i
            return (1 + 2.0 * np.log(self.b / self.a) * n * self.__g(n)) * self.__B(n, n) - n * np.pi * self.__g(n) * np.log(self.b / self.a)
        else:
            n = i
            m = j
            return (1 + 2.0 * np.log(self.b / self.a) * m * self.__g(m)) * self.__B(m, n)

    def __even_proj_get_Bi(self, i):
        '''
        computes matrix element for the projection method

        :param i: (int) i in matrix A
        :return: element at location (m, n) of the matrix
        '''

        n = i
        return 2.0 / n * np.sin(n * self.theta_0)

    def __even_proj_solve(self):
        '''
        Solves the system of matrix equation
        updates A_mn: the matrix element of the LHS of the matrix equation
        updates B_m: the matrix element of the RHS of the matrix equation
        updates X: solution of the system of equations

        The function could be modified to solve the matrix equation using other algorithms,
        such as QR decomposition or SVD algorithm.

        :return: None
        '''

        # solve by QR algo
        self.__even_A_mn = []
        self.__even_B_n = []
        for i in range(2, self.max_val - 1, 2):
            row = []
            for j in range(2, self.max_val - 1, 2):
                row.append(self.__even_proj_get_Aij(i, j))
            self.__even_A_mn.append(row)
            self.__even_B_n.append(self.__even_proj_get_Bi(i))
        q, r = np.linalg.qr(self.__even_A_mn, mode='complete')

        Qb = np.dot(q.T, self.__even_B_n)

        self.__even_X = np.linalg.solve(r, Qb)

        X0 = - 1.0/(np.pi - 2.0 * self.theta_0) * 4 * np.log(self.b/self.a) * \
             sum(self.__even_X[i] * self.__g(2*i+2) * np.sin((2 * i + 2) * self.theta_0) for i in range(0, len(self.__even_X), 1))
        self.__even_X = [X0] + list(self.__even_X)

    def __even_solve(self):
        '''
        Solves the system of matrix equation
        updates A_mn: the matrix element of the LHS of the matrix equation
        updates B_m: the matrix element of the RHS of the matrix equation
        updates X: solution of the system of equations

        The function could be modified to solve the matrix equation using other algorithms,
        such as QR decomposition or SVD algorithm.

        :return: None
        '''

        self.__even_proj_solve()
        self.__even_apply_filter_fun()

    def __even_apply_filter_fun(self, sigma = 2.0 * np.pi / 100.0):
        '''

        Apply the filter function to the solution of system of equations (self.__even_X) and stores the result in self.__even_X_f

        :param sigma: (float), see self.__filter_fun for more detail
        :return: None
        '''

        index = 0
        while index < len(self.__even_X):
            self.__even_X_f.append(self.__even_X[index] * self.__filter_fun(sigma=sigma)(2 * index))
            index += 1

    def solve(self):
        ''' solves the matrix equation and apply the filter function'''
        self.__odd_solve()
        self.__even_solve()

    def __characteristic_impedance_thickness(self, thickness, kf_even, kf_geo):
        '''

        Get the estimated characteristic impedance for a specified thickness

        :param thickness: (float) thickness of the plate
        :param kf_even: (float) the scaling coefficient of the even mode
        :param kf_geo:  (float) the scaling coefficient of the geometric mode
        :return: (list), odd mode, even mode, geometric mean (characteristic impedance)
        '''

        thick_even_imp = self.__odd_char_impedance * np.log((self.a - thickness/kf_even)/self.b) / np.log(self.a/self.b)
        thick_geo_imp = self.__geo_char_impedance * np.log((self.a - thickness/kf_geo)/self.b) / np.log(self.a/self.b)

        return None, thick_even_imp, thick_geo_imp

    def get_characteristic_impedance(self, thickness = None, kf_even = 5.5, kf_geo = 5.5):
        '''

        print the characteristic impedance

        :param thickness: if None or 0, then assume infinitesimal plate, else apply thickness;
                          when thickness !=0, potential and field are not available.
        :param kf_even: (float) the scaling coefficient of the even mode
        :param kf_geo:  (float) the scaling coefficient of the geometric mode
        :return: None
        '''
        
        if thickness is None or thickness == 0.0:
            print ('a = ', self.a, ', b = ', self.b, ', theta_0 = ', self.theta_0, ', thickness = ', thickness)
            print ('odd mode: Z = ', self.__odd_char_impedance)
            print ('even mode: Z = ', self.__even_char_impedance)
            print ('geo (Z_o^2 + Z_e^2)^0.5: Z = ', self.__geo_char_impedance)
        else:
            Zodd, Zeven, Zgeo = self.__characteristic_impedance_thickness(thickness=thickness, kf_even = kf_even, kf_geo = kf_geo)
            print ('a = ', self.a, ', b = ', self.b, ', theta_0 = ', self.theta_0, ', thickness = ', thickness)
            # print ('odd mode: Z = ', self.__odd_char_impedance)
            print ('even mode: Z = ', Zeven)
            print ('geo (Z_o^2 + Z_e^2)^0.5: Z = ', Zgeo)

    def change_mode(self, mode):
        '''

        change the mode

        :param mode: 'odd' or 'even'
        :return: None
        '''
        if mode == 'odd':
            self.mode = mode
            self.X = self.__odd_X
            self.X_f = self.__odd_X_f
            self.A_mn = self.__odd_A_mn
            self.B_n = self.__odd_B_n
        elif mode == 'even':
            self.mode = mode
            self.X = self.__even_X
            self.X_f = self.__even_X_f
            self.A_mn = self.__even_A_mn
            self.B_n = self.__even_B_n
        else:  # self.mode != 'odd' and self.mode != 'even':
            raise ValueError('mode must be \'even\' or \'odd\'')



