def get_trainer(option):
    if option['uncer_method'].lower() == 'gan':
        from trainer.trainer_gan import train_one_epoch
    elif option['uncer_method'].lower() == 'vae':
        from trainer.trainer_vae import train_one_epoch
    elif option['uncer_method'].lower() == 'abp':
        from trainer.trainer_abp import train_one_epoch
    elif option['uncer_method'].lower() == 'ebm':
        from trainer.trainer_ebm import train_one_epoch
    elif option['uncer_method'].lower() == 'basic':
        from trainer.trainer_basic import train_one_epoch
    elif option['uncer_method'].lower() == 'ganabp':
        from trainer.trainer_ganabp import train_one_epoch

    return train_one_epoch
