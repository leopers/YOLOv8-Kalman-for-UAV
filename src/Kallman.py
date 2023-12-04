import numpy as np

class KalmanFilter():
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):

        self.dt = dt
        self.u = np.array([[u_x],[u_y]])
        #initial state guess
        self.x = np.array([[0], [0], [0], [0]])
        
        self.A = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        self.B = np.array([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])
        
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        
        self.R = np.array([[x_std_meas**2,0],
                           [0, y_std_meas**2]])
        
        #initial cov matrix can be initialized as an identity matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self, u : np.array):
        self.x = self.A@self.x + self.B@u
        self.P = self.A@self.P@self.A.T + self.Q
        return self.x, self.P
    
    def filter(self, z):
        K = self.P@self.P.T@np.linalg.inv(self.R+self.C@self.P@self.C.T)
        self.x = self.x + K@(z-self.C@self.x)
        self.P = self.P - K@self.C@self.P
        return self.x, self.P