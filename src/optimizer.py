from mindspore import nn
import mindspore as ms


def get_optim(cfg,net):
    optim_cfg = cfg.SOLVER
    params = get_param_groups(net)

    if optim_cfg.OPTIMIZER != 'adam_onecycle':
        model_params = get_param_groups(net)

    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = nn.Adam(params, learning_rate=optim_cfg.BASE_LR, weight_decay=ms.Tensor(optim_cfg.WEIGHT_DECAY),
                            beta1=0.9, beta2=0.99)

    elif optim_cfg.OPTIMIZER == 'adamw':
        optimizer = nn.AdamWeightDecay(params, learning_rate=ms.Tensor(optim_cfg.BASE_LR),
                                       weight_decay=optim_cfg.WEIGHT_DECAY,
                                       beta1=0.9, beta2=0.99)

    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = nn.SGD(
            params, learning_rate=ms.Tensor(optim_cfg.BASE_LR), weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )

    else:
        raise NotImplementedError

    return optimizer


def get_param_groups(network):
    """ get param groups """
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith(".weight"):
            # Dense or Conv's weight using weight decay
            decay_params.append(x)
        else:
            # all bias not using weight decay
            # bn weight bias not using weight decay, be carefully for now x not include LN
            no_decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]