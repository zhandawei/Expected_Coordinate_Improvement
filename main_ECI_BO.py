# this is a BoTorch implementation of the ECI approach
import torch
from botorch.models.transforms import Normalize,Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats.qmc import LatinHypercube
from botorch.test_functions.synthetic import Rosenbrock
from expected_coordinate_improvement import ExpectedCoordinateImprovement
import warnings
warnings.filterwarnings("ignore")

num_vari = 10
obj_fun = Rosenbrock(dim = num_vari,negate=True)
num_initial = 2*num_vari
max_evaluation = 10*num_vari
bounds = obj_fun.bounds
sampler = LatinHypercube(d=num_vari)
train_x = torch.from_numpy(sampler.random(n=num_initial))*(bounds[1,:]-bounds[0,:]) + bounds[0,:]
train_y = obj_fun(train_x).unsqueeze(-1)
fmin = train_y.max() 
xmin = train_x[train_y.argmax(),:]     
iteration = 0
evaluation = train_x.shape[0]
print(f'ECI on {num_vari}-D Rosenbrock problem, iterations: {iteration}, evaluations: {evaluation}, fmin: {-fmin:0.1f}')
while evaluation < max_evaluation:
    model = SingleTaskGP(train_X=train_x,
                        train_Y=train_y,
                        train_Yvar = torch.full_like(train_y, 1e-6),
                        input_transform=Normalize(d=num_vari), 
                        outcome_transform=Standardize(m=1)
                        )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)  
    # find the maximal ECI values for all coordinates
    max_ECI = torch.zeros((1,num_vari))
    max_x = torch.zeros((1,num_vari),dtype=float)
    for i in range(num_vari):
        acq_fun = ExpectedCoordinateImprovement(model=model,best_f=fmin,best_x=xmin,coordinate=i)
        max_x[:,i],max_ECI[:,i] = optimize_acqf(acq_function=acq_fun,bounds = bounds[:,i:i+1],q = 1,num_restarts=5,raw_samples=5)  
    # get the coordinate optimization order based on the ECI values
    sort_dim = torch.argsort(max_ECI,descending=True)
    for i in range(num_vari):
        select_dim = sort_dim[:,i]
        if i == 0:
            sub_x = max_x[:,select_dim]
        else:
            model = SingleTaskGP(train_X=train_x,
                            train_Y=train_y,
                            train_Yvar = torch.full_like(train_y, 1e-6),
                            input_transform=Normalize(d=num_vari), 
                            outcome_transform=Standardize(m=1)
                            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)              
            acq_fun = ExpectedCoordinateImprovement(model=model,best_f=fmin,best_x=xmin,coordinate=select_dim)
            sub_x,max_ECI = optimize_acqf(acq_function=acq_fun,bounds = bounds[:,select_dim:select_dim+1],q = 1,num_restarts=5,raw_samples=5)
        new_x = xmin.repeat(1,1)
        new_x[:,select_dim] = sub_x
        new_y   = obj_fun(new_x).unsqueeze(-1)
        train_x = torch.cat([train_x,new_x])
        train_y = torch.cat([train_y,new_y])
        fmin = train_y.max()    
        xmin = train_x[train_y.argmax(),:].reshape((1,num_vari))        
        evaluation = train_x.shape[0]
        iteration = iteration + 1
        print(f'ECI on {num_vari}-D Rosenbrock problem, iterations: {iteration}, evaluations: {evaluation}, fmin: {-fmin:0.1f}')


