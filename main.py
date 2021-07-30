from Cocodataset.build_dataset import Build_Dataset
from collections import OrderedDict
import torch.nn as nn
import torch
import os
from fasterrcnn.config import cfg
from fasterrcnn.utils.visualization import *
import warnings

warnings.filterwarnings('ignore')
from fasterrcnn.FASTERRCNN import FasterRCNN


def dataparallel(model, device_ids=None):
    if device_ids is None:
        device_ids = [0]
    Model = nn.DataParallel(model, device_ids=device_ids)
    return Model


def build_data(cfg):
    root = cfg['data_cfg']['data_root']
    train_pipeline = cfg['data_cfg']['train_pipeline']
    kwargs_loader = cfg['data_cfg']['data_loader']
    BD = Build_Dataset()
    dataset = BD.build_train_data(root, train_pipeline)
    dataloader = BD.build_dataloader(dataset, **kwargs_loader)
    return dataloader


def build_data_img(img_root, cfg):
    # 测试一张图方法集成
    test_pipeline = cfg['data_cfg']['test_pipeline']
    BD = Build_Dataset()
    dataset_img = BD.build_img_data(test_pipeline)
    result_info = dataset_img.data_test(img_root)
    return result_info


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError('{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()
    return loss, log_vars


def upgrade_lr(optimizer, cur_iters, cur_epoch, **kwargs):
    type = kwargs.get('type')  # 优化策略

    warmup = kwargs.get('warmup', 'linear')  # 初始的学习率增加的策略，linear为线性增加
    warmup_iters = kwargs.get('warmup_iters', 500)  # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio = kwargs.get('warmup_ratio', 1.0 / 3)  # 起始的学习率

    # 获得学习率
    lr = optimizer.param_groups[0]['lr']

    if cur_iters <= warmup_iters:
        if cur_iters < warmup_iters:
            if warmup == 'constant':
                warmup_lr = lr * warmup_ratio
            elif warmup == 'linear':
                k = (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
                warmup_lr = lr * (1 - k)
            elif warmup == 'exp':
                k = warmup_ratio ** (1 - cur_iters / warmup_iters)
                warmup_lr = lr * k
        elif cur_iters == warmup_iters:
            warmup_lr = optimizer.param_groups[0]['init_lr']
            # 修改学习率
        for b in optimizer.param_groups: b.setdefault('lr', warmup_lr)
    else:
        if type == 'Step':
            step = kwargs.get('step', [8, 11])  # 在第8和11个epoch时降低学习率
            gamma = kwargs.get('gamma', 0.1)
            if isinstance(step, int):
                epoch_lr = lr * (gamma ** (cur_epoch // step))
            else:
                exp = len(step)
                for i, s in enumerate(step):  # 依次递减
                    if cur_epoch < s:
                        exp = i
                        break
                epoch_lr = lr * gamma ** exp
            for b in optimizer.param_groups:
                b.setdefault('lr', epoch_lr)

    return optimizer


def save_checkpoint(model, out_dir, epoch,
                    optimizer=None, save_optimizer=True, meta=None):
    filepath = os.path.join(out_dir, 'epoch_' + str(epoch + 1) + '.pth')

    optimizer = optimizer if save_optimizer else None

    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(
            type(meta)))

    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    torch.save(checkpoint, filepath)


def weights_to_cpu(state_dict):
    """
    Copy a model state_dict to cpu.
    Args: state_dict (OrderedDict): Model weights on GPU.
    Returns: OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def load_checkpoint(model, filename, map_location=None, strict=False):
    checkpoint = torch.load(filename, map_location=map_location)
    state_dict = checkpoint['state_dict']
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict)
    else:
        load_state_dict(model, state_dict, strict)
    return checkpoint


def load_state_dict(module, state_dict, strict=False):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []  # 保存checkpoint不在module中的key
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        try:
            own_state[name].copy_(param)  # 试图赋值给模型
        except Exception:
            raise RuntimeError(
                'While copying the parameter named {}, '
                'whose dimensions in the model are {} not equal '
                'whose dimensions in the checkpoint are {}.'.format(
                    name, own_state[name].size(), param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        else:
            print(err_msg)


def train(cfg, mode='train'):
    # 训练参数
    total_epochs = cfg['total_epochs']
    work_dir = cfg['work_dir']
    check_point = cfg['check_point']

    dataloader = build_data(cfg)
    model = dataparallel(FasterRCNN(cfg))
    optimizer_cfg = cfg['optimizer_info']['optimizer']
    lr_strategy_cfg = cfg['optimizer_info']['lr_strategy']
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_cfg)  # 构建优化器
    lr = optimizer.param_groups[0]['lr']
    for b in optimizer.param_groups: b.setdefault('init_lr', lr)
    total_iters = total_epochs * len(dataloader)
    epoch_per_iters = len(dataloader)

    cur_iters = 0
    cur_epoch = 0
    # kwargs = cfg
    for epoch in range(total_epochs):
        cur_epoch += 1
        for i, data_batch in enumerate(dataloader):
            data, img_metas = data_batch
            try:
                optimizer.zero_grad()
                losses = model(data, img_metas, mode=mode)
                losses, log_vars = parse_losses(losses)
                losses.backward()
                optimizer.step()
                # print(log_vars)
                print('info : epoch[{}]-{}:{}-loss:{}'.format(cur_epoch, i, epoch_per_iters, log_vars))
            except:
                print('error')
                continue
            # save_checkpoint(model, work_dir, epoch, optimizer=optimizer, meta=None)
            cur_iters += 1

            optimizer = upgrade_lr(optimizer, cur_iters, cur_epoch, **lr_strategy_cfg)
        if cur_epoch % check_point['interval'] == 0:
            save_checkpoint(model, work_dir, epoch, optimizer=optimizer, meta=None)


def init_model(checkpoint_root, config):
    model = FasterRCNN(config)
    checkpoint = load_checkpoint(model, checkpoint_root, map_location='cpu')  # 已经载入权重

    model = dataparallel(model, device_ids=[0])

    return model


def init_data_test(cfg):
    # 为构建单张数据及进行实列化
    test_pipeline = cfg['data_cfg']['test_pipeline']
    BD = Build_Dataset()
    dataset_img = BD.build_img_data(test_pipeline)
    return dataset_img


def test_single(model, data_cls, img_root, mode='test'):
    data_img_info = data_cls.data_test(img_root)
    imgs, img_metas = data_img_info['imgs'], [data_img_info['img_metas']]
    with torch.no_grad():  # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        model.eval()
        results = model(imgs, img_metas, mode=mode)
    return results


def test(model, data_cls, img_root):
    results = test_single(model, data_cls, img_root)
    img_name = img_root.split('\\')[-1] if '\\' in img_root else img_root.split('/')[-1]
    res_df = result_convert(results, img_name)
    cat_lst, box_lst, score_lst = res_df['cat'].values, res_df['bbox'].values, res_df['score'].values
    img = cv2.imread(img_root)
    res_img = draw_bbox(img, cat_lst, box_lst, score_lst)
    return res_img, res_df


if __name__ == '__main__':
    # mode = 'train'
    mode = 'test'

    assert mode in ['train', 'test'], 'mode must be in train or test'
    if mode == 'train':
        train(cfg)
    elif mode == 'test':
        data_root = r'C:\Users\51102\Desktop\VOC2007\data_verity\person'
        checkpoint_root = r'C:\Users\51102\Desktop\VOC2007\work_dir\epoch_130.pth'
        model = init_model(checkpoint_root, cfg)
        data_cls = init_data_test(cfg)
        names = os.listdir(data_root)
        for name in names:
            if name[-4:] == '.jpg':
                img_root = os.path.join(data_root, name)
                res_img, res_df = test(model, data_cls, img_root)
                cv2.imwrite('C:/Users/51102/Desktop/VOC2007/test_dir/' + name, res_img)
                show_img(res_img)
                print(res_df)


