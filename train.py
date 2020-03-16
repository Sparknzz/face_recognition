import argparse
import os
from dataloaders import make_data_loader
from models import mobilefacenet
from models.metric import ArcFace
from models.loss import FocalLoss
import torch.optim as optim
import numpy as np
# from visualizer import *
from config import *
import time
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def save_model(model, save_path, name, iter_cnt, acc):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '-' + str(acc) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)

        # Define Network
        net = mobilefacenet.FaceMobileNet(embedding_size=512).cuda()
        
        # Define Metric
        metric = ArcFace(embedding_size=512, class_num=10576).cuda()

        # Define Optimizer
        train_params = [{'params': net.parameters()}, {'params': metric.parameters()}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum, lr=args.lr,
                                    weight_decay=args.weight_decay)

        # Define Criterion
        self.criterion = FocalLoss(gamma=2)

        # Define Evaluator
        # self.evaluator = Evaluator(self.nclass)

        # Define lr scheduler
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_steps, gamma=0.1)

        self.model, self.optimizer, self.metric = torch.nn.DataParallel(net), optimizer, metric

        # self.visualizer = Visualizer()

    def training(self, epoch_no):
        train_loss = 0.0
        self.model.train()

        start = time.time()
        # tbar = tqdm(self.train_loader)
        for i, (_img, _label) in enumerate(self.train_loader):

            if self.args.cuda:
                image, label = _img.cuda(), _label.cuda()

            self.optimizer.zero_grad()
            embedding = self.model(image)            
            thetas = self.metric(embedding, label)
            loss = self.criterion(thetas, label)

            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()

            iters = epoch_no * len(self.train_loader) + i

            if iters % self.args.frequent == 0:
                output = thetas.data.cpu().numpy()
                output = np.argmax(output, axis=1)

                # print(output)
                # print(_label)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                
                speed = self.args.frequent / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} loss {} acc {}'.format(time_str, epoch_no, i, loss.item(), acc))
                
                if False:
                    self.visualizer.display_current_results(iters, loss.item(), name='train_loss') # works for only batch loss not overall loss
                    self.visualizer.display_current_results(iters, acc, name='train_acc') # works for only batch acc not overall acc

            
            if iters!=0 and (iters % self.args.save_interval == 0 or epoch_no == self.args.end_epoch):
                save_model(self.model.module, self.args.prefix, self.args.loss, iters, acc)
        

    def validation(self, epoch):
        self.model.eval()
        test_loss = 0.0

        acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')


        valid_loss = np.zeros(2, np.float32)
        valid_num = np.zeros(2, np.float32)
        for i, (_img, _label) in enumerate(self.val_loader):
            
            _img, _label = _img.cuda(), _label.cuda()

            with torch.no_grad():
                embedding = self.model(_img)
                self.metric(embedding, _label)

            loss, _ = self.criterion(output, truth_mask)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            logit = output.data.cpu().numpy()  # N C H W

            truth_mask = truth_mask.cpu().numpy()
            pred = np.argmax(logit, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(truth_mask, pred)

            l = np.array([*tn, *tp])
            n = np.array([*num_neg, *num_pos])
            valid_loss += l
            valid_num += n

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        valid_loss = valid_loss / valid_num
        print(valid_loss)

#         new_pred = mIoU
#         if new_pred > self.best_pred:
#             self.best_pred = new_pred
        torch.save(self.model.module.state_dict(),
                   '/data1/ningyupeng/model_epoch_{}_mIOU_{}.pth'.format(epoch, mIoU))

def parse_args():

    parser = argparse.ArgumentParser(description="Train Face Network")

    parser.add_argument('--cuda', default=True, help='use cuda')
    parser.add_argument('--network', default=default.network, help='network config')
    parser.add_argument('--loss', default=default.loss, help='loss config')

    parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
    
    #parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
    #parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')

    parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')

    parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
    parser.add_argument('--weight-decay', type=float, default=default.wd, help='weight decay')
    parser.add_argument('--momentum', type=float, default=default.mom, help='momentum')
    parser.add_argument('--frequent', type=int, default=default.frequent, help='')
    parser.add_argument('--batch-size', type=int, default=default.batch_size, help='batch size in each context')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--image-shape', type=int, default=default.image_shape, metavar='N', help='dataloader threads')
    parser.add_argument('--save-interval', type=int, default=10000, metavar='N', help='dataloader threads')

    # training hyper params
    parser.add_argument('--end_epoch', type=int, default=default.end_epoch, metavar='N',
                        help='number of epochs to train (default: auto)')

    # # optimizer params
    # parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
    #                     help='learning rate (default: auto)')

    # parser.add_argument('--lr-scheduler', type=str, default='poly',
    #                     choices=['poly', 'step', 'cos'],
    #                     help='lr scheduler mode: (default: poly)')

    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     metavar='M', help='momentum (default: 0.9)')

    # parser.add_argument('--weight-decay', type=float, default=5e-4,
    #                     metavar='M', help='w-decay (default: 5e-4)')

    # parser.add_argument('--nesterov', action='store_true', default=False,
    #                     help='whether use nesterov (default: False)')

    # # cuda, seed and logging
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--gpu-ids', type=str, default="0,1",
    #                     help='use which gpu to train, must be a \
    #                     comma-separated list of integers only (default=0)')

    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')

    # parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')

    # # evaluation option
    # parser.add_argument('--eval-interval', type=int, default=1,
    #                     help='evaluuation interval (default: 1)')

    # parser.add_argument('--no-val', action='store_true', default=False,
    #                     help='skip validation during training')

    # args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # if args.cuda:
    #     try:
    #         args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    #     except ValueError:
    #         raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # print(args)
    # torch.manual_seed(args.seed)
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    prefix = os.path.join(args.models_root, '%s-%s'%(args.network, args.loss), 'model')
    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)

    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    
    # if config.count_flops:
    #     all_layers = sym.get_internals()
    #     _sym = all_layers['fc1_output']
    #     FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
    #     _str = flops_counter.flops_str(FLOPs)
    #     print('Network FLOPs: %s'%_str)

    args.batch_size = 4 * default.batch_size

    trainer = Trainer(args)
    print('num_classes', config.num_classes)

    for epoch in range(args.end_epoch):
        trainer.training(epoch)
        # if epoch % args.eval_interval == (args.eval_interval - 1):
        # trainer.validation(epoch)

if __name__ == "__main__":
    main()
