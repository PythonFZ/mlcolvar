import torch
from warnings import warn

__all__ = ["KL_TDA_loss"]

class KL_TDA_loss(torch.nn.Module):
    
    def __init__(self,
                 n_states : int,
                 target_centers : list or torch.tensor,
                 target_sigmas : list or torch.tensor):
        """Constructor.

        Parameters
        ----------
        n_states : int
            Number of states. The integer labels are expected to be in between 0
            and ``n_states-1``.
        target_centers : list or torch.Tensor
            Shape ``(n_states, n_cvs)``. Centers of the Gaussian targets.
        target_sigmas : list or torch.Tensor
            Shape ``(n_states, n_cvs)``. Standard deviations of the Gaussian targets.
        """
        super().__init__()
        self.n_states = n_states
        self.target_centers = target_centers
        self.target_sigmas = target_sigmas

    def forward(self,
                H: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """Compute the value of the loss function.

        Parameters
        ----------
        H : torch.Tensor
            Shape ``(n_batches, n_features)``. Output of the NN.
        labels : torch.Tensor
            Shape ``(n_batches,)``. Labels of the dataset.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        loss_centers : torch.Tensor, optional
            Only returned if ``return_loss_terms is True``. The value of the
            loss term associated to the centers of the target Gaussians.
        """
        return kl_tda_loss(H, labels, self.n_states, self.target_centers, self.target_sigmas)

def kl_tda_loss(H : torch.tensor,
            labels : torch.tensor,
            n_states : int,
            target_centers : list or torch.tensor,
            target_sigmas : list or torch.tensor) -> torch.tensor:
    """
    Compute a loss function as the distance from a simple Gaussian target distribution.
    
    Parameters
    ----------
    H : torch.tensor
        Output of the NN
    labels : torch.tensor
        Labels of the dataset
    n_states : int
        Number of states in the target
    target_centers : list or torch.tensor
        Centers of the Gaussian targets
        Shape: (n_states, n_cvs)
    target_sigmas : list or torch.tensor
        Standard deviations of the Gaussian targets
        Shape: (n_states, n_cvs)

    Returns
    -------
    torch.tensor
        Total loss, centers loss, sigmas loss
    """
    if not isinstance(target_centers,torch.Tensor):
        target_centers = torch.Tensor(target_centers)
    if not isinstance(target_sigmas,torch.Tensor):
        target_sigmas = torch.Tensor(target_sigmas)
    
    device = H.device
    target_centers = target_centers.to(device)
    target_sigmas = target_sigmas.to(device) 

    # min_max = torch.zeros_like(target_centers).unsqueeze(0)
    # min_max = torch.vstack((min_max, min_max))
    # n_sigma = target_centers / target_sigmas
    # min_max[:, 0] = target_centers - n_sigma*target_sigmas
    # min_max[:, 1] = target_centers + n_sigma*target_sigmas      

    target_distributions = torch.zeros((n_states, 100), device = device)
    for i in range(n_states): # TODO fix for more dimensions
        x = torch.linspace(float((target_centers[i]-5*target_sigmas[i]).cpu().numpy()), float((target_centers[i]+5*target_sigmas[i]).cpu().numpy()), 100, device=device)
        target_distributions[i] = sym_func(x, target_centers[i], target_sigmas[i])

    loss = torch.zeros(n_states)
    for i in range(n_states):
        # check which elements belong to class i
        if not torch.nonzero(labels == i).any():
            raise ValueError(f'State {i} was not represented in this batch! Either use bigger batch_size or a more equilibrated dataset composition!')
        else:
            H_red = H[torch.nonzero(labels == i, as_tuple=True)]
            mu = torch.mean(H_red, 0)
            sigma = torch.std(H_red, 0)
        
            # x = torch.linspace((mu-4*target_sigmas[i]).numpy(), (mu+4*target_sigmas[i]).numpy(), 100)
            # target_distributions[i] = sym_func(x, mu, target_sigmas[i])
            out = easy_KDE(H_red, [float((target_centers[i]-5*target_sigmas[i]).detach().cpu().numpy()), float((target_centers[i]+5*target_sigmas[i]).detach().cpu().numpy())], 100)
            #out = easy_KDE(H_red, [float((mu-10*target_sigmas[i]).detach().numpy()), float((mu+10*target_sigmas[i]).detach().numpy())], 100)
            out = out / torch.sum(out)
            out = out / torch.max(out) * torch.max(target_distributions[i])
            loss[i] = torch.sum(torch.pow(torch.kl_div(torch.log(out), target_distributions[i]) , 2)) + torch.pow((mu-target_centers[i]),2) + torch.pow((sigma-target_sigmas[i]),2) 
    loss = torch.sum(loss)
    return loss


def sym_func(x, centers, sigma):
    return torch.exp(- torch.div(torch.pow(x-centers, 2), 2*torch.pow(sigma,2) ))

def easy_KDE(x, min_max, n):
    centers = torch.linspace(min_max[0], min_max[1], n, device=x.device)
    sigma = (centers[1] - centers[0]) * 5
    out = torch.sum(sym_func(x, centers, sigma), dim=0)
    return out



