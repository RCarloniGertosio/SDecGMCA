import numpy as np
import utils
from sdecgmca import SDecGMCA
import healpytools as hpyt

# Initialize the parameters of the problem
n = 4                       # number of sources
m = 8                       # number of observations
nside = 128                 # Healpix nside
lmax = np.int(3*nside)      # maximum frequency of the projection in the spherical harmonic space
cutmin = np.int(0.5*nside)  # source generation: frequency at which the band-limiting filter starts to cut
cutmax = np.int(3*nside)    # source generation: frequency above which the alm are fixed to 0
nscales = 3                 # source generation: number of WT scales
sparseLvl = 2               # source generation: desired sparsity level in WT domain (corresponds to a k*std per scale)
condn = 2                   # mixing matrix generation: condition number of the mixing matrix
max0s = 0                   # mixing matrix generation: max nb of zeros (may be relaxed)
minResol = np.int(lmax/8)   # fwhm in the spherical harmonic space of the observation with the worse resolution
maxResol = np.int(lmax)     # fwhm in the spherical harmonic space of the observation with the best resolution
snr = 10                    # SNR in dB
verb = 3                    # verbosity level

# Generate the joint deconvolution and BSS problem
A0, S0, Hl, X, N, Y = utils.generate_problem(n=n, m=m, nside=nside, lmax=lmax, cutmin=cutmin, cutmax=cutmax,
                                             nscales=nscales, sparseLvl=sparseLvl, condn=condn, max0s=max0s,
                                             minResol=minResol, maxResol=maxResol, snr=snr, verb=verb)
S0c = hpyt.convolve(S0, Hl[-1, :])

# View the data
utils.view_data(A0, S0, Y, Hl)

# Initialize the parameters of SDecGMCA
minWuIt = 100                       # minimum number of iterations at warm-up
c_wu = 0.5*np.array([1, 10])        # Tikhonov regularization hyperparameter at warm-up
c_ref = 0.5                         # Tikhonov regularization hyperparameter at refinement
cwuDec = 50                         # number of iterations for the decrease of c_wu
nStd = np.std(N)                    # noise standard deviation
useMad = False                      # use mad to estimate noise std in source space
nscales = 3                         # number of detail scales
k = 3                               # parameter of the k-std thresholding
K_max = 0.5                         # maximal L0 norm of the sources
L1 = True                           # L1 penalization
doRw = True                         # do l1 reweighing during refinement
eps = np.array([1e-2, 1e-6, 1e-4])  # stopping criteria

# Set the algorithm
sdecgmca = SDecGMCA(Y, Hl/Hl[-1, :], n, minWuIt=minWuIt, c_wu=c_wu, c_ref=c_ref, cwuDec=cwuDec, nStd=nStd,
                    useMad=useMad, nscales=nscales, k=k, K_max=K_max, L1=L1, doRw=doRw, eps=eps, verb=verb)

# Run the algorithm
sdecgmca.run()

S = sdecgmca.S
A = sdecgmca.A

utils.asses_solution(A0, S0c, A, S, corrPerm=True, perScale=True, nscales=nscales, view=False)
