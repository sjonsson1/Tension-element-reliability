import numpy as np
import openturns as ot
ot.Log.Show(ot.Log.NONE)
import time


def input_OpenTurns(X, descriptions, myLSF, failure_threshold):
    '''Set up OpenTURNs objects for MCS and FORM analysis.

    Inputs:
    X (list of ot.Distribution): List of distributions for each input variable.
    descriptions (list of str): List of descriptions for each input variable.
    myLSF (function): Limit state function.
    failure_threshold (float): Threshold for failure.

    Returns:
    None (sets global variables for OpenTURNs objects).
    '''


    global R, multinorm_copula, inputDistribution, inputRandomVector
    global myfunction, outputvector, failureevent, optimAlgo 
    global start_pt, start, algo

    R = ot.CorrelationMatrix(len(X))   
    multinorm_copula = ot.NormalCopula(R)

    inputDistribution = ot.ComposedDistribution(X, multinorm_copula)
    inputDistribution.setDescription(descriptions)
    inputRandomVector = ot.RandomVector(inputDistribution)

    myfunction = ot.PythonFunction(len(X), 1, myLSF)

    # Vector obtained by applying limit state function to X1 and X2
    outputvector = ot.CompositeRandomVector(myfunction, inputRandomVector)

    # Define failure event: here when the limit state function takes negative values
    failureevent = ot.ThresholdEvent(outputvector, ot.Less(), failure_threshold)
    failureevent.setName('LSF inferior to 0')

    optimAlgo = ot.Cobyla()
    optimAlgo.setMaximumEvaluationNumber(1000)
    optimAlgo.setMaximumAbsoluteError(1.0e-4)
    optimAlgo.setMaximumRelativeError(1.0e-4)
    optimAlgo.setMaximumResidualError(1.0e-4)
    optimAlgo.setMaximumConstraintError(1.0e-4)


def run_FORM_analysis():
    '''Run FORM analysis using OpenTurns and return key results.

    Returns:
    result (ot.FORMResult): The result of the FORM analysis.
    x_star (ot.Point): The design point in the original space.
    u_star (ot.Point): The design point in the standard normal space.
    pf (float): The probability of failure.
    beta (float): The Hasofer-Lind reliability index.
    '''
    
    start_pt = []

    # Start timer
    start = time.time()
    algo = ot.FORM(optimAlgo,
                   failureevent,
                   inputDistribution.getMean())
    algo.run()
    result = algo.getResult()
    x_star = result.getPhysicalSpaceDesignPoint()
    u_star = result.getStandardSpaceDesignPoint()
    pf = result.getEventProbability()
    beta = result.getHasoferReliabilityIndex()
    
    # End timer
    end = time.time()
    print(f'The FORM analysis took {end-start:.3f} seconds')
    
    print('FORM result, pf = {:.4f}'.format(pf))
    print('FORM result, beta = {:.3f}\n'.format(beta))
    print('The design point in the u space: ', u_star)
    print('The design point in the x space: ', x_star)
    
    return result, x_star, u_star, pf, beta    


def run_MonteCarloSimulation(mc_size):
    '''Run MCS using OpenTurns and return the probability of failure.

    Inputs:
    mc_size (int): Number of samples to generate.

    Returns:
    pf_mc (float): The probability of failure.
    '''


    # Start timer
    start = time.time()
    montecarlosize = mc_size
    outputSample = outputvector.getSample(montecarlosize)

    number_failures = sum(i < 0 for i in np.array(outputSample))[0]
    pf_mc = number_failures/montecarlosize                    

    # End timer and print the time
    end = time.time()
    print(f'The MCS took {end-start:.3f} seconds to '+
          f'evaluate {mc_size} samples.')
    
    print('pf for MCS: ', pf_mc)
    
    return pf_mc



def importance_factors(result):
    """Compute and print the importance factors using FORM results.
    
    Inputs:
    result (ot.FORMResult): The result of the FORM analysis.

    Returns:
    alpha_ot (list): Importance factors from OpenTURNs.
    alpha (list): Importance factors based on the normal vector in U-space.
    sens (list): Sensitivity of the beta to the multivariate distribution.
    """
    print(f'--- FORM Importance Factors (alpha) ---')
    import matplotlib.pyplot as plt
    plt.ion()
    print()
    alpha_ot = result.getImportanceFactors()
    print(f'\nImportance factors, from OpenTURNs:')
    [print(f'  {i:6.3f}') for i in alpha_ot]

    u_star = result.getStandardSpaceDesignPoint()
    inverseTransform = inputDistribution.getInverseIsoProbabilisticTransformation()
    failureBoundaryStandardSpace = ot.ComposedFunction(myfunction, inverseTransform)
    du0 = failureBoundaryStandardSpace.getGradient().gradient(u_star)
    g_grad = np.array(du0).transpose()[0]
    alpha = -g_grad/np.linalg.norm(g_grad)
    print('\nImportance factors, based on normal vector in U-space = ')
    [print(f'  {i:6.3f}') for i in alpha]
    print('Note: this will be different from'
          + ' result.getImportanceFactors()'
          + '\nif there are resistance variables.')

    sens = result.getHasoferReliabilityIndexSensitivity()

    print(f'\nSensitivity of Reliability Index to Multivariate Distribution')
    for i, j in enumerate(result.getHasoferReliabilityIndexSensitivity()):
        print(f'\nDistribution item number: {i}')
        print(f'  Item name: {j.getName()}')
    for k, l in zip(j.getDescription(), j):
        print(f'    {l:+6.3e} for parameter {k}')

    return alpha_ot, alpha, sens