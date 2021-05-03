# from __future__ import print_function
import os
from data.dataset import Dataset
from torch.utils import data
from models.metrics import *
from utilss.visualizer import Visualizer
# from utils.view_model import view_model
import torch
import numpy as np
import time
from config.config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from models.focal_loss import FocalLoss
from models.resnet import resnet_face18
from rec_test import rec_test
from models.IRLoss import InterReguProduct
import cv2

max_acc = 0


def test_swjtu(model):
    model.eval()
    acc, num = rec_test(model, opt.gallery_swjtu, opt.probe_swjtu, opt.swjtu_root, opt.test_batch_size)
    print("epoch:{}, acc:{}".format(i, acc))


def test_curtin(model):
    model.eval()
    f = open("res_curtin.txt", "a")

    # model, gallery_path, probe_path, root_path, batch_size
    acc_ie, ie = rec_test(model, opt.gallery_c, opt.c_probe_IE, opt.test_root, opt.test_batch_size)
    acc_oc, oc = rec_test(model, opt.gallery_c, opt.c_probe_OC, opt.test_root, opt.test_batch_size)
    acc_pe, pe = rec_test(model, opt.gallery_c, opt.c_probe_PE, opt.test_root, opt.test_batch_size)

    acc_avg = (acc_ie * ie + acc_oc * oc + acc_pe * pe) / (oc + ie + pe)
    print(
        "epoch:{}, avg:{}, ie: {}, pe:{}, oc:{}".format(i, acc_avg, acc_ie, acc_pe, acc_oc)+'\n')
    f.write(
        "epoch:{}, avg:{}, ie: {}, pe:{}, oc:{}".format(i, acc_avg, acc_ie, acc_pe, acc_oc)+'\n')
    f.close()

    global max_acc
    if acc_avg > max_acc:
        max_acc = acc_avg
        save_model(model, opt.checkpoints_path, opt.backbone, "best")
    # save_model(model, opt.checkpoints_path, opt.backbone, "new")
    time_str = time.asctime(time.localtime(time.time()))
    print("time:{}, epoch:{}, test acc:{}, best acc:{}".format(time_str, i, acc_avg, max_acc))
    # if i % opt.save_interval == 0 or i == opt.max_epoch:
    #     save_model(model, opt.checkpoints_path, opt.backbone, i)


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + iter_cnt + '.pth')
    # save_name = os.path.join(save_path, name + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def test(model):
    model.eval()
    # acc_oc, oc = ver_test(model, opt.test_list_oc, opt.test_root, opt.test_batch_size)
    # acc_fe, fe = ver_test(model, opt.test_list_fe, opt.test_root, opt.test_batch_size)
    # acc_ps, ps = ver_test(model, opt.test_list_ps, opt.test_root, opt.test_batch_size)
    # acc_tm, tm = ver_test(model, opt.test_list_tm, opt.test_root, opt.test_batch_size)
    # acc_avg = (ps + oc + fe + tm) / (1285 + 1002 + 1012 + 1360)
    # f = open("test_res.txt", "a")
    # f.write(
    #     "epoch:{}, avg:{}, acc_fe:{}, acc_oc:{}, acc_ps:{}, acc_tm:{}, ".format(i, acc_avg, acc_fe, acc_oc,
    #                                                                                                acc_ps,
    #                                                                                                acc_tm) + "\n")
    # model, gallery_path, probe_path, root_path, batch_size
    acc_oc, oc = rec_test(model, opt.test_gallery, opt.probe_oc, opt.test_root, opt.test_batch_size)
    acc_fe, fe = rec_test(model, opt.test_gallery, opt.probe_fe, opt.test_root, opt.test_batch_size)
    acc_ps, ps = rec_test(model, opt.test_gallery, opt.probe_ps, opt.test_root, opt.test_batch_size)
    acc_tm, tm = rec_test(model, opt.test_gallery, opt.probe_tm, opt.test_root, opt.test_batch_size)
    acc_nu, nu = rec_test(model, opt.test_gallery, opt.probe_nu, opt.test_root, opt.test_batch_size)
    acc_avg = (acc_ps*ps + acc_oc*oc + acc_fe*fe + acc_tm*tm + acc_nu*nu) / (oc + fe + ps + tm + nu)
    print(
        "epoch:{}, avg:{}, nu: {}, fe:{}, oc:{}, ps:{}, tm:{}, ".format(i, acc_avg, acc_nu, acc_fe, acc_oc,
                                                                                acc_ps,
                                                                                acc_tm) + "\n")
    # f.close()
    global max_acc
    if acc_avg > max_acc:
        max_acc = acc_avg
        save_model(model, opt.checkpoints_path, opt.backbone, "best")
    # save_model(model, opt.checkpoints_path, opt.backbone, "new")
    time_str = time.asctime(time.localtime(time.time()))
    print("time:{}, epoch:{}, test acc:{}, best acc:{}".format(time_str, i, acc_avg, max_acc))
    # if i % opt.save_interval == 0 or i == opt.max_epoch:
    #     save_model(model, opt.checkpoints_path, opt.backbone, i)


if __name__ == '__main__':
    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    # identity_list = get_lfw_list(opt.lfw_test_list)
    # img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model = resnet_face18(pretrained=False)

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.0)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.3, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    elif opt.metric == 'ir_loss':
        metric_fc = InterReguProduct(opt.num_classes, opt.num_classes, s=64, m=0.55, easy_margin=opt.easy_margin)
    else:
        metric_fc = SoftMaxProduct(512, opt.num_classes)
        # metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    # print(model)
    model.to(device)
    metric_fc.to(device)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()

    for i in range(opt.max_epoch):
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feats = model(data_input)
            feats = metric_fc(feats, label)
            loss = criterion(feats, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters = i * len(trainloader) + ii
            if iters % opt.print_freq == 0:
                output = feats.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} epoch:{} iter:{} {}:iters/s loss:{}  '
                      'acc:{} '.format(time_str, i, ii, round(speed, 3), round(loss.item(), 3),
                                                acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        # test_curtin(model)
        # test_swjtu(model)
        save_model(model, opt.checkpoints_path, opt.backbone, "new")
        scheduler.step()

        if opt.display:

            visualizer.display_current_results(iters, max_acc, name='test_acc')
