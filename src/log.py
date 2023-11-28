import wandb


def initialize_wandb(args):
    wandb.init(
        project = "Implant GAN",
        entity = "yehyun-suh",
        name = args.experiment_name
    )


def log_result(total_loss, mse_loss, kl_loss, train_loader_length):
    wandb.log({
        'Total Loss': total_loss/train_loader_length,
        'MSE Loss': mse_loss/train_loader_length,
        'KL Loss': kl_loss/train_loader_length
    })