from ngsolve import *
import numpy as np
import csv

def compute_error(data, gfu, u_ex, norm, domain, VorB):

    ns = specialcf.normal(data.mesh.dim)
    Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)

    if norm[:2] == 'L2':
        if VorB == BND:
            if norm == 'L2norm':
                aux = InnerProduct(ns*(gfu-u_ex()), ns*(gfu-u_ex()))
                err = sqrt(Integrate(aux, mesh = data.mesh, order = gfu.space.globalorder*2, 
                                VOL_or_BND=BND, 
                                definedon = domain))
            elif norm == 'L2tang':
                aux = InnerProduct(Ps*(gfu-u_ex()), Ps*(gfu-u_ex()))
                err = sqrt(Integrate(aux, mesh = data.mesh, order = gfu.space.globalorder*2, 
                                VOL_or_BND=BND, 
                                definedon = domain))
            else:
                aux = InnerProduct(gfu-u_ex(), gfu-u_ex())
                err = sqrt(Integrate(aux, mesh = data.mesh, order = gfu.space.globalorder*2, 
                                VOL_or_BND=BND, 
                                definedon = domain))
        else:
            aux = InnerProduct(gfu-u_ex(), gfu-u_ex())
            err = sqrt(Integrate(aux, mesh = data.mesh, order = gfu.space.globalorder*2,
                            definedon = domain))
    else:
        raise ValueError('The norm given is not implemented')
    
    return err

def gradient(f,P):

    m, _ = P.dims
    l = len(f.dims)

    if l == 0:
        # scalar gradient is defined traditionally

        if m == 2:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y)))
        elif m == 3:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y), f.Diff(z))) 
            
    elif l == 1:
        # vector gradient is defined component by component
        # and disposed along columns

        if m == 2:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            output = CoefficientFunction((aux1[0], aux2[0], \
                aux1[1], aux2[1]), dims = (m,m))
        elif m == 3:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            aux3 = gradient(f[2], P)
            output = CoefficientFunction((aux1[0], aux2[0], aux3[0], \
                aux1[1], aux2[1], aux3[1],\
                    aux1[2], aux2[2], aux3[2]), dims = (m,m))
            
    else:

        raise RuntimeError("Don't know how to take the gradient. Only scalars and vectors are accepted.")
    
    return output

def sphere_transformation(A, b, t):

    phi = A*CF((x,y,z)) + b
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y,z)) - b)

    w_phi = A.Diff(t)*inv_phi + b.Diff(t)

    e1 = A[:,0]
    e2 = A[:,1]
    e3 = A[:,2]

    n_ex = Cross(e2, e3)*inv_phi[0]+Cross(e3, e1)*inv_phi[1] + Cross(e1, e2)*inv_phi[2]
    n_ex = n_ex/Norm(n_ex)

    return phi, inv_phi, w_phi, detJ, n_ex


class SaveError():

    def __init__(self, ex_sol = None, filename = 'errors.csv', folderpath = '.', norm = 'L2'):

        self.ex_sol = ex_sol
        self.norm = norm
        self.folderpath = folderpath
        self.filename = filename

        file_path = os.path.join(self.folderpath, filename)
        self.filepath = file_path

    def Initialize(self, data, pde):

        os.makedirs(self.folderpath, exist_ok=True)

        if hasattr(data, 't'):
            columns = ['Time'] + pde.name
        else:
            columns = pde.name

        with open(self.filepath, mode='w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(columns)

    def Save(self, data, pde):

        err = pde.get_error(data, self.ex_sol, self.norm)

        if hasattr(data, 't'):
            towrite = [data.t.Get()] + err
        else:
            towrite = err

        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(towrite)


class SaveSolution():

    def __init__(self, filename = 'sol', folderpath = '.', sample_rate = 1, subdivision = 1):

        self.filename = filename
        self.folderpath = folderpath
        self.sample_rate = sample_rate
        self.subdivision = subdivision

        file_path = os.path.join(self.folderpath, filename)
        self.filepath = file_path

    def Initialize(self, data, pde):

        os.makedirs(self.folderpath, exist_ok=True)

        if data.mesh.ne == 0:
            self.v_or_b = BND
        else:
            self.v_or_b = VOL

        os.makedirs(self.folderpath, exist_ok=True)

        self.vtk = VTKOutput(data.mesh, coefs= pde.gfu_save,
                names=pde.name, filename = self.filepath,
                subdivision = self.subdivision)
        
    def Save(self, data, pde):

        if hasattr(data, 't'):
            if data.iter%self.sample_rate == 0:
                self.vtk.Do(time = data.t.Get(), vb = self.v_or_b)
        else:
            self.vtk.Do(vb = self.v_or_b)

def MandBP(gfu_vec, dt = None, weights=None, BP=None, MP=False, mass0=None):

    tol = 1e-10

    if BP and not MP:

        gfu_data = np.minimum(np.maximum(gfu_vec, BP[0]), BP[1])  

    elif MP:

        def F(xsi):

            result = 0

            temp = gfu_vec + dt * xsi

            result -= mass0
            dummy = np.minimum(np.maximum(temp, BP[0]), 
                                BP[1])
            result += np.sum(weights * dummy)

            return result

        xsi_old0 = 0
        xsi_old1 = -dt
        xsi2 = 1e100

        while abs(F(xsi_old1) - F(xsi_old0))>tol:

            F1 = F(xsi_old1)
            F0 = F(xsi_old0)

            xsi2 = xsi_old1-F1*(xsi_old1 - xsi_old0)/(F1 - F0)

            xsi_old0 = xsi_old1
            xsi_old1 = xsi2

        threshold = dt * xsi2
        gfu_data = np.minimum(np.maximum(gfu_vec + threshold, 
                                            BP[0]), BP[1]) 
    return gfu_data