# -*- coding: utf-8 -*-

import os
import glob
import TimeSeriesParams as TSP


def GetDataParameters(dataSet, sel = None):
    p = TSP.TimeSeriesParams()
    # the following override those in TimeSeriesParams.
    if (dataSet == "BeepTest"):
        params={
            'dataName':'Beep Test' }
        params['dataFile']='../data/BeepTestData/beep_3103_pre_Trunc.mat'
        params['debugFolder'] = "..//debug/BeepTestDebug/"
        params['dataVariable'] = 'Y' # which variable in the matlab file is the time series data
        params['K'] = 2
        params['window'] = 100
        params['stride'] = 25
#        params['stride'] = 50

        params['cluster_sig'] = 1.0 #s, variance on prior for Theta, equation10.

    
        params['alpha'] = 1.1 # only gets used if paramtest != 0 
        params['beta'] =  3 # only gets used if paramtest != 0 
        params['regObs'] = 100 #lambda
        params['initMethod'] = 'CPD' #line 166 in code. used in optimisation or something.

        params['cyclicIterMax'] = 1000 #self explanatory. for convergence of cost function to a local min i think
        params['cyclicThresh'] = 0.1 #Eta in algorithm 1. 
    elif (dataSet == "MSR_Batch"):
        # This data is normalized for its mean and standard deviation.
        params={
            'dataName':'MSR'}
        params['dataFile']='../data/MSR_Data/subj001_1.mat' 
        params['debugFolder'] = "../debug/MSR_Batch/"
        params['dataVariable'] = 'Y'

        params['window'] = 250
        params['stride'] = 125

        params['cluster_sig'] = 1.0 
        
        params['alpha'] = 1.1 # only gets used if paramtest != 0 
        params['beta'] =  3 # only gets used if paramtest != 0 
        params['regObs'] = 100 #10 #lambda
        params['initMethod'] = 'CPD'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.05
    elif (dataSet == "bookshelf_fft"):
        params={
            'dataName':'bookshelf fft'} # not sure what this is for..
        params['dataFile']='../data/bookshelf_concatenated_fft_tests.mat'
        params['debugFolder'] = "../debug/bookshelf_fft/" #??? why do we need this . is it supposed to be a .mat file... no
        params['dataVariable'] = 'Y' # variable containing the time series in the matlab file
# update these.
        params['K'] = 3
        params['window'] = 800
        params['stride'] = 800 # just a test run cause this should produce pretty obvious results i guess

        params['cluster_sig'] = 1.0 # kept the same variance on prior for Theta
        
        params['alpha'] = 1.1 # only gets used if paramtest != 0 so ignoring
        params['beta'] =  3 # only gets used if paramtest != 0 so ignoring
        params['regObs'] = 100 #10 #lambda keeping it the same
        params['initMethod'] = 'CPD'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.05
    elif (dataSet == "bookshelf_fft_short"):
        params={
            'dataName':'bookshelf fft short with healthy and damaged only'} # not sure what this is for..
        params['dataFile']='../data/bookshelf_concatenated_fft_tests_short.mat'
        params['debugFolder'] = "../debug/bookshelf_fft_short/" #??? why do we need this . is it supposed to be a .mat file... no
        params['dataVariable'] = 'Y' # variable containing the time series in the matlab file
# update these.
        params['K'] = 2
        params['window'] = 800
        params['stride'] = 800 # just a test run cause this should produce pretty obvious results i guess

        params['cluster_sig'] = 1.0 # kept the same variance on prior for Theta
        
        params['alpha'] = 1.1 # only gets used if paramtest != 0 so ignoring
        params['beta'] =  3 # only gets used if paramtest != 0 so ignoring
        params['regObs'] = 100 #10 #lambda keeping it the same
        params['initMethod'] = 'CPD'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.1 #trying a bigger value so that it's faster..
    elif (dataSet == "MSR_BatchGT"):
        # This data is normalized for its mean and standard deviation.
        params={
            'dataName':'MSR'}
        params['dataFile']='../data/MSR_Data/subj001_1.mat'
        params['debugFolder'] = "../debug/MSR_Batch/"
        params['dataVariable'] = 'Y'
#        params['K'] = 10
        params['K'] = 4 #7
        params['window'] = 250 
        params['stride'] = 125

        params['cluster_sig'] = 1.0
        
        params['alpha'] = 1.1
        params['beta'] =  3
        params['regObs'] = 100
        params['regObs'] = 10

        params['initMethod'] = 'label'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.05

    elif (dataSet == "ParamTest"):
        # This data is normalized for its mean and standard deviation.
        params={
            'dataName':'paramTest'}
        params['dataFile']='../data/ParamTest/subj001_1.mat'
        params['debugFolder'] = "../data/debug/ParamTest/"
        params['dataVariable'] = 'Y'

        params['window'] = 250 
        params['stride'] = 125

        params['K'] = 2
        params['alpha'] = 1.1
        params['beta'] =  3
        
        params['initMethod'] = 'CPD'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.05
    else:
        print('dataSet not recognised by DataSetParameters')
    p.update(params)
    return p

