import wandb


def initialize_wandb(args):
    wandb.init(
        project = "Implant GAN",
        entity = "yehyun-suh",
        name = args.experiment_name
    )


def log_unpaired_result(total_loss, mse_loss, kl_loss, train_loader_length):
    wandb.log({
        'Total Loss': total_loss/train_loader_length,
        'MSE Loss': mse_loss/train_loader_length,
        'KL Loss': kl_loss/train_loader_length
    })


def log_paired_result(total_loss, mse_loss, kl_loss, synth_loss, train_loader_length):
    wandb.log({
        'Total Loss': total_loss/train_loader_length,
        'MSE Loss': mse_loss/train_loader_length,
        'KL Loss': kl_loss/train_loader_length,
        'Synthetic Loss': synth_loss/train_loader_length
    })