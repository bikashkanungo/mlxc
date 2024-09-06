import numpy as np
import math
import pylibxc

def getArrayType(x):
    moduleName = type(x).__module__
    if moduleName == np.__name__:
        return "np"

    elif moduleName == torch.__name__:
        return "torch"

    else:
        return None

class PW92():
    def __init__(self, dtype="float64", device=torch.device("cpu")):
        self.X = pylibxc.LibXCFunctional("lda_x", "polarized")
        self.C = pylibxc.LibXCFunctional("lda_c_pw", "polarized")
        self.device = device
        self.nptype = np.float64
        if dtype == "float32":
            self.nptype = np.float32

    def eval(self, rho, evalGrad=False, outType='np'):
        rhoArrayType = getArrayType(rho)
        if rhoArrayType not in ["np", "torch"]:
            raise ValueError('''Invalid array type provided. '''\
                             '''Valid types are numpy.ndarray and torch.Tensor''')
        if outType not in ["np", "torch"]:
            raise ValueError('''Invalid outType passed. Valid types are:'''\
                             ''' 'np'for numpy.ndarray and 'torch' for torch.Tensor''')

        rhoFlattened = None
        N = rho.shape[0]
        if isinstance(rho, np.ndarray):
            rhoFlattened = np.empty(2*N, dtype=self.nptype)
            rhoFlattened[0::2] = rho[:,0]
            rhoFlattened[1::2] = rho[:,1]

        if isinstance(rho, torch.Tensor):
            rhoAlpha = (torch.flatten(rho[:,0])).numpy()
            rhoBeta= (torch.flatten(rho[:,1])).numpy()
            rhoFlattened = np.empty(2*N, dtype=self.nptype)
            rhoFlattened[0::2] = rhoAlpha
            rhoFlattened[1::2] = rhoBeta

        inp = {}
        inp["rho"] = rhoFlattened
        XVals = self.X.compute(inp, do_exc=True, do_vxc=evalGrad, do_fxc=False)
        CVals = self.C.compute(inp, do_exc=True, do_vxc=evalGrad, do_fxc=True)
        ex = (XVals["zk"])[:,0]
        ec = (CVals["zk"])[:,0]
        exc = None
        vrho = None
        if outType == 'np':
            exc = ex + ec
        if outType == 'torch':
            exc = torch.from_numpy(ex+ec).to(self.device)

        if evalGrad == True:
            vrhox = XVals["vrho"]
            vrhoc = CVals["vrho"]
            if outType == 'np':
                vrho = vrhox+vrhoc
            if outType == 'torch':
                vrho =  torch.from_numpy(vrhox+vrhoc).to(self.device)

        return {'exc': exc, 'vrho': vrho}


class NNLDA:
    def __init__(self, modelFile, dtype="float64", device=torch.device("cpu")):
        r"""!
        @brief Constructor
        @param[in] modelFile The .ptc file for the NNLDA
        @param[in] dtype String providing the floating point precision used in evaluating the functional.
                   Valid values are: "float64" for double precision and "float32" for single precision.
        @param[in] device torch.device object specifying the device (CPU or GPU) on which the major
                   computation and memory allocation should occur.
        """
        self.model = torch.jit.load(modelFile).to(device)
        self.torchdtype = torch.float64
        self.pw92 = PW92(dtype=dtype, device=device)
        self.device = device
        if dtype == "float32":
            self.torchdtype = torch.float32

    def eval(self, rho, evalGrad=False, outType = 'np', tol=1e-8):
        r"""!
        @brief Evaluate the XC energy density (per unit charge) and the derivative
        of the XC energy (per unit volume) with respect to the spin-density .
        This function allows evaluation for a batch of N points at a time, for efficiency.
        @param[in] rho numpy.ndarray or torch.Tensor of shape (N,2) containing the spin density.
               That is rho[i, 0] and rho[i,1] are the up-spin and down-spin density for the i-th point, respectively
        @param[in] evalGrad Boolean to specify whether to evaluate the derivative with respect to rho or not.
        @param[in] outType String specifying the type of the output array.
                   Valid values are: "np" for numpy.ndarray and "torch" for torch.Tensor
        @param[in] tol Specifies a tolerance for rho and sigma to avoid division by zero
        @return A dictionary of the form {"exc": exc, "vrho": vrho}
                where exc is the XC energy density per unit charge,
                vrho is the partial derivative of the XC energy density per unit volume with respect to the spin-density,
                exc and vrho are arrays of type defined by outType.
                exc is of shape (N,), i.e., exc[i] is the XC energy density per unit charge for the i-th point
                vrho is of shape (N,2): vrho[i,0] = \f$ \frac{\partial e_i}{\partial \rho^{\uparrow}_i}\f$
                and vrho[i,1] = \f$ \frac{\partial e}{\partial \rho^{\downarrow}_i}\f$,
                where \f$ e_i = (\rho^{\uparrow}_i + \rho^{\downarrow}_i)*exc_i\f$ is the XC energy density per unit volume.
                If evalGrad is False, then vrho is set to None.
        """
        # sanity checks
        rhoArrayType = getArrayType(rho)
        if rhoArrayType not in ["np", "torch"]:
            raise ValueError('''Invalid array type provided. '''\
                             '''Valid types are numpy.ndarray and torch.Tensor''')

        if outType not in ["np", "torch"]:
            raise ValueError('''Invalid outType passed. Valid types are:'''\
                             ''' 'np'for numpy.ndarray and 'torch' for torch.Tensor''')

        N = rho.shape[0]
        inp = torch.empty((N,2), dtype=self.torchdtype, device=self.device)
        if isinstance(rho, np.ndarray):
            rho_ = torch.from_numpy(rho).to(self.device)
            inp[:,0:2] = rho_

        if isinstance(rho, torch.Tensor):
            inp[:,0:2] = rho.to(self.device)

        # add tolerance to rho and modGradRho
        inp += tol

        rhoTotal = inp[:,0] + inp[:,1]

        # if evalGrad is True, make inp have requires_grad as True, as we need
        # to perform autograd
        if evalGrad:
            inp = inp.requires_grad_(True)

        excUnitVol = self.model(inp)
        excUnitCharge = excUnitVol/rhoTotal
        pw92Data = self.pw92.eval(rho, evalGrad, outType)
        if outType == 'np':
            excUnitCharge = excUnitCharge.cpu().detach().numpy() + pw92Data['exc']
        if outType == 'torch':
            excUnitCharge = excUnitCharge + pw92Data['exc']

        vrho = None
        if evalGrad:
            excOnes = torch.ones_like(excUnitVol).to(self.device)
            g = torch.autograd.grad(excUnitVol, inp, grad_outputs = excOnes)[0]
            vrho = g[:,0:2]
            if outType == 'np':
                vrho = vrho.cpu().detach().numpy() + pw92Data['vrho']

            if outType == 'torch':
                vrho = vrho + pw92Data['vrho']

        return {'exc': excUnitCharge, 'vrho': vrho}


if __name__ == "__main__":
    # provide path to the NNLDA .ptc file
    modelFile = "NNLDA.ptc"

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

    # Boolean to specify whether to evaluate the derivatives
    # of the XC energy density (per unit volume) with respect to rho or not
    evalGrad = True

    nnlda = NNLDA(modelFile, dtype = dtype, device = device)
    ret = nnlda.eval(rho, evalGrad=evalGrad, outType=outType)
    print(ret)
