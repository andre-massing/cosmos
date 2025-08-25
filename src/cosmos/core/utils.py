import numpy as np

def MandBP(gfu_vec, dt = None, weights=None, BP=None, MP=False, mass0=None):

    tol = 1e-10

    if BP and not MP:

        gfu_data = np.clip(gfu_vec, BP[0], BP[1])  

    elif MP:

        def F(xsi):

            result = 0

            temp = gfu_vec + dt * xsi

            result -= mass0
            dummy = np.clip(temp, BP[0], BP[1])
            result += np.sum(weights * dummy)

            return result

        xsi_old0 = 0
        xsi_old1 = -dt
        xsi2 = -1

        while abs(F(xsi_old1) - F(xsi_old0))>tol or xsi2==-1:

            F1 = F(xsi_old1)
            F0 = F(xsi_old0)

            xsi2 = xsi_old1-F1*(xsi_old1 - xsi_old0)/(F1 - F0)

            xsi_old0 = xsi_old1
            xsi_old1 = xsi2

        threshold = dt * xsi2
        gfu_data = np.clip(gfu_vec + threshold, BP[0], BP[1]) 

    return gfu_data