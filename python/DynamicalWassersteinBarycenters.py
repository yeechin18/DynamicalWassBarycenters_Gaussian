# -*- coding: utf-8 -*-
import torch
import scipy.io as scio
import numpy as np
import DiffWassersteinLib as dwl
import SimplexRandomWalkUtils as util
import PSD_RiemannianOptimization as psd
import matplotlib.pyplot as plt
import DataSetParameters as dsp
import CostBarycenterLib as cbl
import argparse
import os, sys
import time

class TimeSeriesCost():
    def __init__(self, params, y, pi, mu0, cov0, geoMean, geoCov, x0=None, gamma=None, A_Beta=None, B_Beta=None, A_Beta0=None, B_Beta0=None, w_Beta=None):
        self.p = params #these are set up in DataSetParameters.py, and obtained based on whatever dataset is used
        self.y = y #the empirical gaussians for each window. each element is a list of 2, 0: mean, 1: covariance. exploit the decoupling of mean and variance in equation2 of wass2 dist between gaussians. 
        self.pi = pi ##???
        self.x0 = x0 #the first state (which together with random innovation vector determines the whole state vector).
        self.mu0 = mu0 #reference normal dist. for calculating prior likelihoods on Theta. (equation 10, and read paragraph below for p(Theta)).. p.muPrior or muP gets passed in
        self.cov0 = cov0 #reference normal distn param. "" ... p.covPrior or covP gets passed in
        self.gamma = gamma # in main, gamma = torch.tensor(np.zeros((p.T,p.K))+p.eps, dtype = dtype, requires_grad=True). no idea what p.eps is, but it's defined in timeseriesparams.py
        self.geoMean = geoMean #a cbl.Euclidean() object (costbarycentrelib.py). used to calculate the means term in equation2, the wasserstein2 distance between two gaussian distns.
        self.geoCov = geoCov #a cbl.WassersteinPSD() object. (unless GMM, handled in this class.) used to calculate the cov term in equation2, the wass2 dist between two gaussian distns.
        self.A_Beta = A_Beta #H - hyperparam for transition component of beta mixture
        self.B_Beta = B_Beta #H - hyperparam for transition component of beta mixture 
        self.A_Beta0 = A_Beta0 #H - hyperparam for stationary component of beta mixture 
        self.B_Beta0 = B_Beta0 #H - hyperparam for stationary component of beta mixture 
        self.w_Beta = w_Beta #H - weights (for each purestate) determining the mixture amounts of stationary v. transition components in beta mixture model
        self.computeEval = 0
        
        # the folowing function computes the loss function given values for the parameters of interest. We minimise this function in main to get the estimated values for the parameters.
        #arguments in the below fn: reference gaussian parameters: 
        #mu is mu0 but in the right data structure 
        #gonna assume covP is more of less sig0 except that's already one of the class attributes: cov0.
        # useage in main: costFunc.cost(gamma, x0, mu, covP, A_Beta0, B_Beta0, A_Beta, B_Beta, w_Beta)

    # compute equation9 (bar argmin part) given certain values for parameters of interest 
    def cost(self, gamma, x0, mu, covP, A0_beta, B0_beta, A_beta, B_beta, w_Beta):
        # compute time series state given step parameter
        X2 = util.StateEvolutionDynamics(self.p, x0, self.pi, gamma) #gamma is delta in utils py . 
        
        cov = []
        for i in range(self.p.K):
            cov.append(covP[i])
        
        # Update the current mean and variance parameters
        clustMean = []
        clustCov = []
        for i in range(self.p.K):
            clustMean.append(mu[i])
            clustCov.append(cov[i])
        
        # Now compute barycentric distributional parameters and pdf for each observation
        # calculating wasserstein2 dist between empirical gaussians and model predicted gaussians for equation9 term
        obsCost = []
        baryMeans=[]
        baryCovs=[]
        obsEval = []
        for i in range(self.p.T): #for each window...
            # first deal with GMM method of calculating covariance distance (orsomething).. would be used for GMM linear interpolation baseline, not relevant for us 
            if (self.geoCov == 'GMM'): # The gmm models as a mixture
                obsCost.append(util.GaussGmm_WassDist(y[i][0], y[i][1], mu, cov, X2[i])**2)
                # This monte carlo simulation is costly, only update the evaluation when writing debug file
                if (self.computeEval==1):
                    obsEval.append(util.GaussGmm_WassDist_MonteCarlo(y[i][0].detach().numpy(), y[i][1].detach().numpy(), 
                                                                 mu, cov, X2[i].detach().numpy()))
                else:
                    obsEval.append(0)
                
            else:  # all other models output a single gaussian (the method proposed in the paper)
                # first get the mean and covs of the model predicted gaussians i think
                baryMeans.append(self.geoMean.barycenter(torch.stack(clustMean),X2[i]))
                baryCovs.append(self.geoCov.barycenter(torch.stack(clustCov), X2[i]))
                
                # Loss is Wasserstein distance to barcenter distribution
                obsCost.append(self.geoMean.distance(y[i][0], baryMeans[i])**2 + 
                                 cbl.WassersteinPSD.dist(y[i][1], baryCovs[i])**2) #wass2 dist between model predicted and empirical gaussians in equation9.
                obsEval.append(obsCost[i].detach().numpy()) # eval same as cost

        # Compute loss for step parameters(gamma)
        # innovation parameters prior (Gamma prior)
        log_pGamma = util.LogGammaLiklihoodBimodalAB(p, A_Beta0, B_Beta0, A_Beta, B_Beta, w_Beta, gamma) 
        
        # Pure state parameters prior (Theta prior)
        log_pTheta = []
        for i in range(self.p.K):
            log_pTheta.append(-torch.sqrt(2*torch.tensor(np.pi))*self.p.cluster_sig +  -1/(2*self.p.cluster_sig**2)*util.GaussWassDistance(self.mu0, self.cov0, mu[i], cov[i]) )
                
        # EQUATION9. (except for the argmin part). This entire class is based off presumed values for the parameters of interest. optimisation happens in main
        lossFunc = self.p.regObs * (torch.sum(torch.stack(obsCost))) - torch.tensor(self.p.T)*(torch.sum(torch.stack(log_pTheta))) - (torch.sum(log_pGamma)) 
        # lambda * [wasserstein2 distance between observed gaussians and model predicted gaussians under given parameter values] - T*log(p(Theta)p(Gamma))

        if (torch.isnan(lossFunc)):
            print('stop')
            raise Exception("bad value")
        return (lossFunc, X2, torch.stack(obsCost), log_pGamma, torch.stack(log_pTheta), torch.stack(cov), obsEval)
        
    # this function ... is used in PSD_RiemannianOptimization.py
    def evaluate(self, listMuSig):
        n = int(len(listMuSig)/2)
        return self.cost(self.gamma, self.x0, [torch.tensor(x) for x in listMuSig[:n]], [torch.tensor(x) for x in listMuSig[n:]], self.A_Beta0, self.B_Beta0, self.A_Beta, self.B_Beta, self.w_Beta)[0].detach().numpy()
        
if __name__=="__main__":
    time_start = time.time()
    #Initialize Data
    dtype = torch.float
    device = torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataSet", type=str, default=None)
    parser.add_argument("dataFile", type=str, default=None)
    parser.add_argument("debugFolder", type=str, default=None)
    parser.add_argument("geoCov", type=str, default=None)
    parser.add_argument("--ParamTest", dest="ParamTest", default=0, type=int) # Used in ParamTest
    parser.add_argument("--lambda", dest="lam", type=float) # Used in ParamTest
    parser.add_argument("--s", dest="s", type=float) # Used in ParamTest
    ParamTest=0
    
    try:
        print("Reading Arguments from Command Line")
        args = parser.parse_args()
        print(args.dataSet)
        print(args.dataFile)
        print(args.debugFolder)
        print(args.geoCov)
        p = dsp.GetDataParameters(args.dataSet)
        print('parameters updated')
        p.update({'dataFile':args.dataFile, 'debugFolder':args.debugFolder, 'geoCov':args.geoCov, 'paramFile':args.geoCov+'_params.txt'})
        if (args.ParamTest ==1):
            p.update({'regObs':args.lam, 'cluster_sig':args.s})
            if (args.ParamTest==1):
                ParamTest=1
        else:
            print("Not Running ParamTest test")
                    
        print('DataFile: ' + p.dataFile)
        print('DebugFolder: ' + p.debugFolder)
        print('geoCov: ' + p.geoCov)
    except:
        print("Instead using default Params: ", sys.exc_info()[0])
        # Load dataset. Here we need defined y = observationsde
        dataSet = 'MSR_Batch'
        p = dsp.GetDataParameters(dataSet)
    
    datO = scio.loadmat(p.dataFile)[p.dataVariable].astype(float) # often dataVariable is 'Y'.  # check dat0 shape after importing 
    dat = util.WindowData(datO, p.window, p.stride, p.offset)
    (T, dump, dim) = dat.shape

    try: # If the file contains a K value, use it.
        K = np.squeeze(scio.loadmat(p.dataFile)['K'].astype(int))
        p.update({'K':K})
        print('Using K value specified in data file.')
    except:
        print('Using K Value specified in DataSetParameters.py')

    # Update and print parameters
    p.update({'T':T, 'dim':dim})
    try:
        os.mkdir(p.debugFolder)
    except: 
        print("Folder already exists") #all good in output initial run up to here, then traceback somewhere between now til next print stmt
    p.write(f=p.debugFolder + p.paramFile)
    
    # Initialize Model Parameters. 
    print('Initialising model parameters')
    gamma = torch.tensor(np.zeros((p.T,p.K))+p.eps, dtype = dtype, requires_grad=True) #takes barely any time. debug fine up to here on bookshelffftshort
    if (ParamTest==0): #this block happens instantly
        x0 = torch.tensor(np.ones(p.K)/p.K, dtype = dtype, requires_grad=True)
        A_Beta = torch.tensor(np.ones(p.K)*10, dtype = dtype, requires_grad=True)
        B_Beta = torch.tensor(np.ones(p.K)*20, dtype = dtype, requires_grad=True)
        A_Beta0 = torch.tensor(np.ones(p.K)*1.1, dtype = dtype, requires_grad=False)
        B_Beta0 = torch.tensor(np.ones(p.K)*20, dtype = dtype, requires_grad=False)
        w_Beta = torch.tensor(np.ones(p.K)*0.5, dtype=dtype, requires_grad=True)
    else:
        x0 = torch.tensor(np.ones(p.K)/p.K, dtype = dtype, requires_grad=True)
        A_Beta = torch.tensor(np.ones(p.K)*p.alpha, dtype = dtype, requires_grad=True)
        B_Beta = torch.tensor(np.ones(p.K)*p.beta, dtype = dtype, requires_grad=True)
        A_Beta0 = torch.tensor(np.ones(p.K)*1.1, dtype = dtype, requires_grad=False)
        B_Beta0 = torch.tensor(np.ones(p.K)*20, dtype = dtype, requires_grad=False)
        w_Beta = torch.tensor(np.ones(p.K)*0.5, dtype=dtype, requires_grad=True)
        
    meanDat = np.mean(datO, axis=0)
    stdDat = np.std(datO,axis=0) #happens instantly

    # used in optimisation or something. just leave same as the other datasets.
    if (p.initMethod == 'CPD'): # initialize based on K CPD Fit. 
        cpd = util.CPD_WM1(datO, p.window*2)
        (muO, sigO) = util.CPD_Init(datO, cpd, max(cpd)*0.2, p.K) # this is NOT mu0 and sig0 - not the reference gaussian parameters for prior on Theta.
    elif (p.initMethod == 'label'): # Alternately initialize based on labels
        try:
            L = np.squeeze(scio.loadmat(p.dataFile)['L'].astype(int))
        except: 
            np.ones(len(datO))*5 #not gonna be used by us but just chucked it in to proof against errors based on MSR data values
        (muO, sigO) = util.label_Init(datO, L)
        cpd=[]

    time_now = time.time()
    print(f'Time Taken since start: {time_now-time_start:.2f}s')
    print('Finding reference gaussian distribution for prior on Theta (equation 10)')
    # reference gaussian params after equation10. becomes p.muPrior and p.covPrior. gets passed to TimeSeriesCost for caluclating loss.
    # I guess it makes sense for this to be the reference gaussian - prior on Theta is fitted to the data.
    (muP, sigP) = util.FitMuSig(datO, p.K) # Regularize based on distance to (mean of gmm parameters, average eValue of gmm Covariances) # P for Prior
    print('Am I ever reached? finished fitting gaussian mixture')
    covDat = dwl.CovarianceBarycenter(torch.tensor(sigO, dtype=dtype), torch.ones(p.K)/p.K, torch.eye(p.dim), nIter=10).detach().numpy()

    p.muPrior=torch.tensor(muP, dtype=dtype, requires_grad=False)
    p.covPrior=torch.tensor(sigP, dtype=dtype, requires_grad=False)
        
    time_now = time.time()
    print(f'Time Taken since start: {time_now-time_start:.2f}s')
    # Compue empirical means and covariances of the input time series for each window based on windowsize and stride specified in dsp
    print('Computing empirical gaussians for each window of time series (equation 5)')
    y=[]
    for i in range(p.T):
        if (i == p.T//4):
            print('quarter way through all windows')
        if (i == p.T//2):
            print('half way through all windows')
        obsMean = np.mean(dat[i], axis=0)
        obsCov = 1/(p.window-1)*np.matmul((dat[i]-obsMean).T, dat[i]-obsMean)
        minDiag=1e-2
        for j in range(p.dim): #just to make sure we are not singular
            obsCov[j,j]=max(minDiag,obsCov[j,j])
        y.append((torch.tensor(obsMean, dtype=dtype, requires_grad=False), torch.tensor(obsCov, dtype=dtype, requires_grad=False)))
    print('Finished computing empirical gaussians')
    time_now = time.time()
    print(f'Time Taken since start: {time_now-time_start:.2f}s')
    pi = torch.tensor(np.ones(p.K)/p.K, dtype=dtype, requires_grad=False) # this needs to sum to 1

    
    # Setup params for manifold optimization (riemannian line search i think)
    print('Setting up params for optimsation')
    man = []
    mu = torch.tensor(muO, dtype=dtype, requires_grad=True)
    for i in range(p.K): # Euclidean manifold for Means
        man.append(psd.Euclidean())
        geoMean = cbl.Euclidean()
    for i in range(p.K): # WB manifold for cov matrices
        man.append(psd.WassersteinBuresPSDManifold())
    covP = torch.tensor(sigO, dtype=dtype, requires_grad=True)

    if (p.geoCov=='Wass'):
        geoCov = cbl.WassersteinPSD(torch.eye(p.dim)) #instanciate an object with barys0 = identity matrix, able to calculate covariance term in wasserstein2 distance between two gaussians
    elif (p.geoCov=='GMM'):
        geoCov = 'GMM'
           

    # Setup optimization params            
    SManifold = psd.Product(man)
    optimGamma = torch.optim.Adam([gamma, x0], lr=p.lr_Gamma)
    optimAB = torch.optim.Adam([A_Beta, B_Beta, w_Beta], lr=p.lr_Gamma)
    init=0
    swap = 'Gamma' # Coordinate descent
    runningCount=0
    costFunc = TimeSeriesCost(p, y, pi, p.muPrior, p.covPrior, geoMean, geoCov)#################
    history=[]
    evalHistory=[]
    cyclicPoints=[]
    # Start Optimization 
    time_now = time.time()
    print(f'Time Taken since start: {time_now-time_start:.2f}s')
    print('Beginning optimisation')
    for t in range(p.nOptimStep): # 50,000 according to tsp.py
        # Since the monte carlo simulations for GMM evaluation takes a long time, set a flag to indicate we are running debug
        # I should run the code in vscode and actually view this lol
        if (t % p.printInterval*5 == 0): 
            time_now = time.time()
            print(f'Time Taken since start: {time_now-time_start:.2f}s')
            print('Making progress... done 500 steps')
            costFunc.computeEval = 1
            (lossFunc, X2, obsCost, log_pGamma, log_pTheta, cov, obsEval) = costFunc.cost(gamma, x0, mu, covP, A_Beta0, B_Beta0, A_Beta, B_Beta, w_Beta)
            evalHistory.append(np.sum(obsEval))
            costFunc.computeEval = 0
        else:
            (lossFunc, X2, obsCost, log_pGamma, log_pTheta, cov, obsEval) = costFunc.cost(gamma, x0, mu, covP, A_Beta0, B_Beta0, A_Beta, B_Beta, w_Beta)
        
        # Backward pass: compute gradient of the loss with respect to model
        lossFunc.backward()
        history.append(lossFunc.detach().numpy())
        
        # Log results
        print(lossFunc) #nts: this didn't happen initial run.
        if (p.logFile is not None):
            print(lossFunc, file=p.logFile)
             
        # Update for gamma
        if (swap == 'Gamma'):    
            # Update for Gammas
            optimGamma.step()
    
            # Kind of a hack, but we clamp Gamma to [0,1] and apply boundary conditions to x0 outside of the gradient update
            gamma.data = gamma.clamp(min=p.eps, max=1-p.eps)
            x0.data = x0.clamp(min=p.eps)
            x0.data = x0.data/torch.sum(x0)
        elif (swap == 'AB'):
            optimAB.step()
            A_Beta.data = A_Beta.clamp(min=1.1)
            for k in range(p.K):
                B_Beta.data[k] = B_Beta[k].clamp(min=1.1, max=A_Beta[k].detach().numpy()*(1-0.15)/0.15) # mean of Beta distribution >= 0.1
            w_Beta.data = w_Beta.clamp(min=0.01, max=0.99)
            
        else: # Line Search update for means
            xt = [mu[i].detach().numpy() for i in range(p.K)] + [covP[i].detach().numpy() for i in range(p.K)]
            xt_nGrad  = [-mu.grad[i].detach().numpy() for i in range(p.K)] + [-covP.grad[i].detach().numpy() for i in range(p.K)]
            
            if (p.cyclic_MuSig): 
                for i in range(p.K*2): 
                    if (i % p.K != runningCount % (p.K)): #Optimize one mean at a time
                        xt_nGrad[i] = xt_nGrad[i]*0
                    
            riemannianGradient = SManifold.euc_to_riemannian_gradient(xt, xt_nGrad)
            
            optimMean = psd.LineSearchSimple(SManifold, TimeSeriesCost(p, y, pi.detach(), p.muPrior, p.covPrior, geoMean, geoCov, x0=x0.detach(), gamma=gamma.detach(), A_Beta0=A_Beta0.detach(), B_Beta0=B_Beta0.detach(), A_Beta=A_Beta.detach(), B_Beta=B_Beta.detach(), w_Beta=w_Beta.detach()), suff_decrease=1e-10, maxIter=20, init_alpha=p.lr_Cluster)
            update = optimMean.search(xt, [x for x in riemannianGradient])
            mu.data=torch.tensor(update[:p.K], dtype=dtype)
            covP.data=torch.tensor(update[p.K:], dtype=dtype)

        # Reset Gradient Computation        
        x0.grad.data.zero_()
        gamma.grad.data.zero_()
        mu.grad.data.zero_()
        covP.grad.data.zero_()
        A_Beta.grad.data.zero_()
        B_Beta.grad.data.zero_()
        w_Beta.grad.data.zero_()
                 
        # Coordinate Descent criterea
        runningCount = runningCount+1
        if ((runningCount > (p.K*2) and history[-p.K]-history[-1] < p.cyclicThresh*(p.K)) or runningCount > p.cyclicIterMax):
            if (swap == 'Gamma'):
                swap = "AB"
                print("Swapping from Gamma to AB optim after " + str(runningCount) + " steps")
            elif (swap == 'AB'):
                swap = "Cluster"
                print("Swapping from AB to Cluster optim after " + str(runningCount) + " steps")
            else:
                swap = 'Gamma'    
                print("Swapping from Cluster to Gamma optim after " + str(runningCount) + " steps")
            runningCount=0
            cyclicPoints.append(t)
        
        # Save Data
        if (t % p.printInterval == 0 and not torch.isnan(lossFunc)): # Print Interval
            print("Save Debug " + p.debugFolder + p.geoCov +"_"+  p.outputFile )
            scio.savemat(p.debugFolder + p.geoCov +"_"+ p.outputFile, mdict={'t':t, 
                                              'dat':dat,
                                              'pi':pi.detach().numpy(),
                                              'datO':datO,
                                              'meanDat':meanDat,
                                              'covDat':covDat,
                                              'meanP':muP,
                                              'sigP':sigP,
                                              'cpd':cpd,
                                              'cyclicPoints':cyclicPoints,
                                              'log_pGamma':log_pGamma.detach().numpy(),
                                              'log_pTheta':log_pTheta.detach().numpy(),
                                              'X':X2.detach().numpy(),
                                              'obsCost':obsCost.detach().numpy(),
                                              'obsEval':obsEval,
                                              'evalHistory':evalHistory,
                                              'history':history,
                                              'window':p.window,
                                              'stride':p.stride,
                                              'x0':x0.detach().numpy(), 
                                              'gamma':gamma.detach().numpy(), 
                                              'mu':mu.detach().numpy(), 
                                              'covP':covP.detach().numpy(), 
                                              'covv':cov.detach().numpy(), 
                                              'muO':muO,
                                              'covO':sigO,
                                              'A_Beta0':A_Beta0.detach().numpy(),
                                              'B_Beta0':B_Beta0.detach().numpy(),
                                              'A_Beta':A_Beta.detach().numpy(),
                                              'B_Beta':B_Beta.detach().numpy(),
                                              'w_Beta':w_Beta.detach().numpy()})