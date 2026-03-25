import torch
from torch import Tensor
from botorch.acquisition.analytic import AnalyticAcquisitionFunction,_scaled_improvement,_ei_helper
from botorch.models.model import Model
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import average_over_ensemble_models,t_batch_mode_transform



class ExpectedCoordinateImprovement(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        best_x: float | Tensor,
        coordinate: int | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("best_x", torch.as_tensor(best_x))
        self.register_buffer("coordinate", torch.as_tensor(coordinate))
        self.maximize = maximize
    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models

    def forward(self, X: Tensor) -> Tensor:
        input_X = self.best_x.repeat(X.shape[0],1,1)
        input_X[:,:,self.coordinate:self.coordinate+1] = X
        mean, sigma = self._mean_and_sigma(input_X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)  
        return (sigma * _ei_helper(u)).squeeze(-1)