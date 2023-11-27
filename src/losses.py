import torch


def KL_div_N01(
        z_mu: torch.Tensor, z_log_sigma_sq: torch.Tensor
        ) -> torch.Tensor:
    """Calculate the KL divergence of a Gaussian distribution from N(0,I)

    Args:
        z_mu (torch.Tensor): _description_  #TODO: describe the shape
        z_log_sigma_sq (torch.Tensor): _description_  #TODO: describe the shape

    Returns:
        torch.Tensor: _description_  #TODO: describe the shape
    """

    term_1 = z_log_sigma_sq.sum(axis=1)
    term_2 = z_mu.shape[1]
    term_3 = z_log_sigma_sq.exp().sum(axis=1)
    term_4 = (z_mu * z_mu).sum(axis=1)
    kl = 0.5 * (-term_1 - term_2 + term_3 + term_4)

    return kl[0]


if __name__ == "__main__":
    mu = torch.tensor([[0]*5])
    sigma = torch.tensor([[0]*5])
    kl = KL_div_N01(mu, sigma)

    print(f"mu: {mu}, {type(mu)}")
    print(f"sigma: {sigma}, {type(sigma)}")
    print(f"KL: {kl}, {type(kl)}")
