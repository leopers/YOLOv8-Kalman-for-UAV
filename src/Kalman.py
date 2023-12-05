import numpy as np

class KalmanFilter():
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):

        self.dt = dt
        self.u = np.matrix([[u_x],[u_y]])
        #initial state guess
        self.x = np.matrix([[0], [0], [0], [0]])
        
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])
        
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])
        
        #initial cov matrix can be initialized as an identity matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]
    
    def filter(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P
        return self.x[0:2]
    
