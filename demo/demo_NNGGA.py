import torch
import numpy as np
import math
import pylibxc

def __getArrayType__(x):
    moduleName = type(x).__module__
    if moduleName == np.__name__:
        return "np"

    elif moduleName == torch.__name__:
        return "torch"

    else:
        return None

class PBE():
    def __init__(self, dtype="float64", device=torch.device("cpu")):
        self.X = pylibxc.LibXCFunctional("gga_x_pbe", "polarized")
        self.C = pylibxc.LibXCFunctional("gga_c_pbe", "polarized")
        self.device = device
        self.nptype = np.float64
        if dtype == "float32":
            self.nptype = np.float32

    def eval(self, rho, sigma, evalGrad=False, outType='np'):
        rhoArrayType = __getArrayType__(rho)
        sigmaArrayType = __getArrayType__(sigma)
        if rhoArrayType not in ["np", "torch"]:
            raise ValueError('''Invalid array type provided. '''\
                             '''Valid types are numpy.ndarray and torch.Tensor''')

        if sigmaArrayType not in ["np", "torch"]:
            raise ValueError('''Invalid array type provided. '''\
                             '''Valid types are numpy.ndarray and torch.Tensor''')

        if rhoArrayType != sigmaArrayType:
            raise ValueError('''Mismatch is array type of rho and sigma. '''\
                             '''Both should either be numpy.ndarray or torch.Tensor.''')
        if outType not in ["np", "torch"]:
            raise ValueError('''Invalid outType provided. '''\
                             '''Valid types are numpy.ndarray and torch.Tensor''')

        rhoFlattened = None
        sigmaFlattened = None
        N = rho.shape[0]
        if isinstance(rho, np.ndarray):
            rhoFlattened = np.empty(2*N, dtype=self.nptype)
            sigmaFlattened = np.empty(3*N, dtype=self.nptype)
            rhoFlattened[0::2] = rho[:,0]
            rhoFlattened[1::2] = rho[:,1]
            sigmaFlattened[0::3]= sigma[:,0]
            sigmaFlattened[1::3]= sigma[:,1]
            sigmaFlattened[2::3]= sigma[:,2]

        if isinstance(rho, torch.Tensor):
            rhoAlpha = (torch.flatten(rho[:,0])).numpy()
            rhoBeta= (torch.flatten(rho[:,1])).numpy()
            rhoFlattened = np.empty(2*N, dtype=self.nptype)
            sigmaFlattened = np.empty(3*N, dtype=self.nptype)
            rhoFlattened[0::2] = rhoAlpha
            rhoFlattened[1::2] = rhoBeta
            sigmaFlattened[0::3]= sigma[:,0].numpy()
            sigmaFlattened[1::3]= sigma[:,1].numpy()
            sigmaFlattened[2::3]= sigma[:,2].numpy()

        inp = {}
        inp["rho"] = rhoFlattened
        inp["sigma"] = sigmaFlattened
        vsigmax = None
        vsigmac = None
        XVals = self.X.compute(inp, do_exc=True, do_vxc=evalGrad, do_fxc=False)
        CVals = self.C.compute(inp, do_exc=True, do_vxc=evalGrad, do_fxc=False)
        ex = (XVals["zk"])[:,0]
        ec = (CVals["zk"])[:,0]
        exc = None
        vrho = None
        vsigma = None
        if outType == 'np':
            exc = ex+ec
        if outType == 'torch':
            exc = torch.from_numpy(ex+ec).to(self.device)

        if evalGrad == True:
            vrhox = XVals["vrho"]
            vsigmax = XVals["vsigma"]
            vrhoc = CVals["vrho"]
            vsigmac = CVals["vsigma"]
            if outType == 'np':
                vrho = vrhox+vrhoc,
                vsigma =  vsigmax+vsigmac
            if outType == 'torch':
                vrho = torch.from_numpy(vrhox+vrhoc).to(self.device)
                vsigma = torch.from_numpy(vsigmax+vsigmac).to(self.device)

        return {'exc': exc, 'vrho': vrho, 'vsigma': vsigma}


class NNGGA:
    r"""!
    @brief Class to encapsulate the NNGGA and NNGGA-UEG functional based on a input .ptc file.
    @note This class assumes a spin-unrestricted (spin-polarized) form. Thus, even for a spin-restricted
    calculation, the user needs to provide the up-spin and down-spin densities.
    """
    def __init__(self, modelFile, dtype="float64", device=torch.device("cpu")):
        r"""!
        @brief Constructor
        @param[in] modelFile The .ptc file for the NNGGA or NNGGA-UEG model
        @param[in] dtype String providing the floating point precision used in evaluating the functional.
                   Valid values are: "float64" for double precision and "float32" for single precision.
        @param[in] device torch.device object specifying the device (CPU or GPU) on which the major
                   computation and memory allocation should occur.
        """
        self.model = torch.jit.load(modelFile).to(device)
        self.torchdtype = torch.float64
        self.pbe = PBE(dtype=dtype, device=device)
        self.device = device
        if dtype == "float32":
            self.torchdtype = torch.float32

    def eval(self, rho, sigma, evalGrad=False, outType = 'np', tol=1e-8):
        r"""!
        @brief Evaluate the XC energy density (per unit charge) and the derivative
        of the XC energy (per unit volume) with respect to the spin-density and sigma.
        This function allows evaluation for a batch of N points at a time, for efficiency.
        @param[in] rho numpy.ndarray or torch.Tensor of shape (N,2) containing the spin density.
               That is rho[i, 0] and rho[i,1] are the up-spin and down-spin density for the i-th point, respectively
        @param[in] sigma numpy.ndarray or torch.Tensor of shape (N,3) containing the information of the gradient
                   of the density. We follow the libxc convention. That is,
                   sigma[i,0] = \f$ \nabla (\rho^{\uparrow}_i) \cdot \nabla (\rho^{\uparrow}_i)\f$
                   sigma[i,1] = \f$ \nabla (\rho^{\uparrow}_i) \cdot \nabla (\rho^{\downarrow}_i)\f$
                   sigma[i,2] = \f$ \nabla (\rho^{\downarrow}_i) \cdot \nabla (\rho^{\downarrow}_i)\f$,
                   where \f$ \rho^{\uparrow}_i)\f$ and \f$ \rho^{\downarrow}_i) \f$ are the up-spin
                   and down-spin densities for the i-th point.
        @param[in] evalGrad Boolean to specify whether to evaluate the derivative with respect to rho and sigma or not.
        @param[in] outType String specifying the type of the output array.
                   Valid values are: "np" for numpy.ndarray and "torch" for torch.Tensor
        @param[in] tol Specifies a tolerance for rho and sigma to avoid division by zero
        @return A dictionary of the form {"exc": exc, "vrho": vrho, "vsigma": vsigma"}
                where exc is the XC energy density per unit charge,
                vrho is the partial derivative of the XC energy density per unit volume with respect to the spin-density,
                vsigma is the partial derivative of the XC energy density per unit volume with respect to sigma.
                exc, vrho, vsigma are arrays of type defined by outType.
                exc is of shape (N,), i.e., exc[i] is the XC energy density per unit charge for the i-th point
                vrho is of shape (N,2): vrho[i,0] = \f$ \frac{\partial e_i}{\partial \rho^{\uparrow}_i}\f$
                and vrho[i,1] = \f$ \frac{\partial e}{\partial \rho^{\downarrow}_i}\f$,
                where \f$ e_i = (\rho^{\uparrow}_i + \rho^{\downarrow}_i)*exc_i\f$ is the XC energy density per unit volume.
                vsigma is of shape (N,3): vsigma[i,j] = \f$ \frac{\partial e_i}{\partial \sigma_{i,j}}f$, where
                \f$ \sigma_{i,j} = \f$ sigma[i,j] defined above.
                If evalGrad is False, then vrho and vsigma are set to None.
        """
        # sanity checks
        rhoArrayType = __getArrayType__(rho)
        sigmaArrayType = __getArrayType__(sigma)
        if rhoArrayType not in ["np", "torch"]:
            raise ValueError('''Invalid array type provided. '''\
                             '''Valid types are numpy.ndarray and torch.Tensor''')

        if sigmaArrayType not in ["np", "torch"]:
            raise ValueError('''Invalid array type provided. '''\
                             '''Valid types are numpy.ndarray and torch.Tensor''')

        if rhoArrayType != sigmaArrayType:
            raise ValueError('''Mismatch is array type of rho and sigma. '''\
                             '''Both should either be numpy.ndarray or torch.Tensor.''')

        if outType not in ["np", "torch"]:
            raise ValueError('''Invalid outType passed. Valid types are:'''\
                             ''' 'np'for numpy.ndarray and 'torch' for torch.Tensor''')

        N = rho.shape[0]
        modGradRho = (sigma[:,0] + 2*sigma[:,1] + sigma[:,2])**0.5
        inp = torch.empty((N,3), dtype=self.torchdtype, device=self.device)
        if isinstance(rho, np.ndarray):
            rho_ = torch.from_numpy(rho).to(self.device)
            modGradRho_ = torch.from_numpy(modGradRho).to(self.device)
            inp[:,0:2] = rho_
            inp[:,2] = modGradRho_

        if isinstance(rho, torch.Tensor):
            inp[:,0:2] = rho.to(self.device)
            inp[:,2] = modGradRho.to(self.device)


        # add tolerance to rho and modGradRho
        inp += tol

        rhoTotal = inp[:,0] + inp[:,1]
        modGradRho = inp[:,2]

        # if evalGrad is True, make inp have requires_grad as True, as we need
        # to perform autograd
        if evalGrad:
            inp = inp.requires_grad_(True)

        excUnitVol = self.model(inp)
        excUnitCharge = excUnitVol/rhoTotal
        pbeData = self.pbe.eval(rho, sigma, evalGrad, outType)
        if outType == 'np':
            excUnitCharge = excUnitCharge.cpu().detach().numpy() + pbeData['exc']

        if outType == 'torch':
            excUnitCharge = excUnitCharge + pbeData['exc']

        vrho = None
        vsigma = None
        if evalGrad:
            excOnes = torch.ones_like(excUnitVol).to(self.device)
            g = torch.autograd.grad(excUnitVol, inp, grad_outputs = excOnes)[0]
            vrho = g[:,0:2]
            vsigma = torch.empty((N,3), dtype=self.torchdtype, device=self.device)
            sigma_ = None
            if isinstance(sigma, np.ndarray):
                sigma_ = torch.from_numpy(sigma).to(self.device)
            elif isinstance(sigma, torch.Tensor):
                sigma_ = sigma.to(self.device)

            vsigma[:,0] = 0.5*g[:,2]*sigma_[:,0]/modGradRho
            vsigma[:,1] = g[:,2]*sigma_[:,1]/modGradRho
            vsigma[:,2] = 0.5*g[:,2]*sigma_[:,2]/modGradRho

            if outType == 'np':
                vrho = vrho.cpu().detach().numpy() + pbeData['vrho']
                vsigma = vsigma.cpu().detach().numpy() + pbeData['vsigma']

            if outType == 'torch':
                vrho = vrho + pbeData['vrho']
                vsigma = vsigma + pbeData['vsigma']

        return {'exc': excUnitCharge, 'vrho': vrho, 'vsigma': vsigma}


if __name__ == "__main__":
    # provide path to the NNGGA  or NNGGA-UEG .ptc file
    modelFile = "NNGGA.ptc"

    # Set the torch.device (CPU or GPU).
    # It is relevant only while using torch
    # comment or uncomment the following as per requirement
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    # Set the output array type.
    # It can be either "np" (for numpy.ndarray) or
    # "torch" (for torch.Tensor).
    outType = "torch"

    # Set the floating point precision
    # Valid values are "float64" for double precision
    # and "float32" for single precision
    dtype = "float64"

    # demo with N points
    N = 5

    # Provide the input the spin density rho
    # rho can be either np.ndarray or torch.Tensor.
    # comment or uncomment the following two lines as per requirement
    rho = torch.rand((N,2)) ## for torch.Tensor
    #rho = np.random.random((N,2)) ## for np.ndarray

    # Provide the input the sigma
    # sigma can be either np.ndarray or torch.Tensor.
    # comment or uncomment the following two lines as per requirement
    sigma = torch.rand((N,3)) ## for torch.Tensor
    #sigma = np.random.random((N,3)) ## for np.ndarray

    # Boolean to specify whether to evaluate the derivatives
    # of the XC energy density (per unit volume) with respect to rho or not
    evalGrad = True

    nngga = NNGGA(modelFile, dtype = dtype, device = device)
    ret = nngga.eval(rho, sigma, evalGrad=evalGrad, outType = outType)
    print(ret)
