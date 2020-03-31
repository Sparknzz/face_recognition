import argparse
import os
from dataloaders import make_data_loader
# from models import mobilefacenet
from models import model
from models.metric import ArcFace
from models.loss import FocalLoss
import torch.optim as optim
import numpy as np
# from visualizer import *
from config import *
import time
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)

        # Define Network
        # net = mobilefacenet.FaceMobileNet(embedding_size=512).cuda()
        net = model.MobileFaceNet().cuda()
        # paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

        # Define Metric
        metric = ArcFace(embedding_size=512, class_num=args.num_classes).cuda()

        # Define Optimizer
        train_params = [{'params': net.parameters()}, {'params': metric.parameters()}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum, lr=args.lr,
                                    weight_decay=args.weight_decay)

        # Define Criterion
        # self.criterion = FocalLoss(gamma=2)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define Evaluator
        # self.evaluator = Evaluator(self.nclass)

        # Define lr scheduler
        # self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_steps, gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epoch,)
        
        # torch.nn.DataParallel(net)
        self.model, self.optimizer, self.metric = torch.nn.DataParallel(net), optimizer, metric

        print(self.optimizer)

        # resume pretrained model
        if args.resume is not None:
            # load backbone weights
            if not os.path.isfile(args.resume+'.pth'):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume+'.pth'))

            backbone_path = args.resume.replace(args.loss, args.network)
            state_dict = torch.load(backbone_path+'.pth')
            self.model.module.load_state_dict(state_dict)

            splits = args.resume.split('/')[-1].split('_')
            # /root/face_recognition/checkpoints/mobilefacenet/arcface_269_240000_0.419921875
            print("=> loaded checkpoint backbone '{}' (epoch {} iters {})".format(args.resume, splits[1], splits[2]))


            # # load arcface weights
            # if not os.path.isfile(args.resume+'.pth'):
            #     raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume+'.pth'))

            # state_dict = torch.load(args.resume+'.pth')
            # self.metric.load_state_dict(state_dict)

            # splits = args.resume.split('/')[-1].split('_')
            # print("=> loaded checkpoint arcface '{}' (epoch {} iters {})".format(args.resume, splits[1], splits[2]))

        # self.visualizer = Visualizer()

    def get_learning_rate(self, ):
        lr=[]

        for param_group in self.optimizer.param_groups:
            lr +=[ param_group['lr'] ]
        
        return lr[0]

    def save_model(self, model, save_path, name, epoch_no, iter_cnt, acc):
        save_name = os.path.join(save_path, name + '_' + str(epoch_no) + '_' + str(iter_cnt) + '_' + str(acc) + '.pth')
        torch.save(model.state_dict(), save_name)
        return save_name
        
    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
        
    
    def training(self, epoch_no):
        self.model.train()
        train_loss = 0.0

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

                lr_rate = self.get_learning_rate()

                print('{} train epoch {} iter {} lr {} loss {:.5f} acc {:.2f}'.format(time_str, epoch_no, i, lr_rate,loss.item(), acc))
                
                if False:
                    self.visualizer.display_current_results(iters, loss.item(), name='train_loss') # works for only batch loss not overall loss
                    self.visualizer.display_current_results(iters, acc, name='train_acc') # works for only batch acc not overall acc

            
            if iters!=0 and (iters % self.args.save_interval == 0 or epoch_no == self.args.end_epoch):
                self.save_model(self.model.module, prefix_dir, self.args.network, epoch_no, iters, acc)
                self.save_model(self.metric, prefix_dir, self.args.loss, epoch_no, iters, acc)
        
        self.scheduler.step()


#     def validation(self, epoch):
#         self.model.eval()
#         test_loss = 0.0

#         acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
#         if opt.display:
#             visualizer.display_current_results(iters, acc, name='test_acc')


#         valid_loss = np.zeros(2, np.float32)
#         valid_num = np.zeros(2, np.float32)
#         for i, (_img, _label) in enumerate(self.val_loader):
            
#             _img, _label = _img.cuda(), _label.cuda()

#             with torch.no_grad():
#                 embedding = self.model(_img)
#                 self.metric(embedding, _label)

#             loss, _ = self.criterion(output, truth_mask)
#             test_loss += loss.item()
#             tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
#             logit = output.data.cpu().numpy()  # N C H W

#             truth_mask = truth_mask.cpu().numpy()
#             pred = np.argmax(logit, axis=1)
#             # Add batch sample into evaluator
#             self.evaluator.add_batch(truth_mask, pred)

#             l = np.array([*tn, *tp])
#             n = np.array([*num_neg, *num_pos])
#             valid_loss += l
#             valid_num += n

#         # Fast test during the training
#         Acc = self.evaluator.Pixel_Accuracy()
#         Acc_class = self.evaluator.Pixel_Accuracy_Class()
#         mIoU = self.evaluator.Mean_Intersection_over_Union()
#         FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

#         print('Validation:')
#         print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
#         print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
#         print('Loss: %.3f' % test_loss)

#         valid_loss = valid_loss / valid_num
#         print(valid_loss)

# #         new_pred = mIoU
# #         if new_pred > self.best_pred:
# #             self.best_pred = new_pred
#         torch.save(self.model.module.state_dict(),
#                    '/data1/ningyupeng/model_epoch_{}_mIOU_{}.pth'.format(epoch, mIoU))

def parse_args():

    parser = argparse.ArgumentParser(description="Train Face Network")

    parser.add_argument('--cuda', default=True, help='use cuda')
    parser.add_argument('--network', default=default.network, help='network config')
    parser.add_argument('--loss', default=default.loss, help='loss config')

    parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
    
    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')

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
    parser.add_argument('--save-interval', type=int, default=1000, metavar='N', help='dataloader threads')

    # training hyper params
    parser.add_argument('--end_epoch', type=int, default=default.end_epoch, metavar='N',
                        help='number of epochs to train (default: auto)')


    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')

    # # evaluation option
    # parser.add_argument('--eval-interval', type=int, default=1,
    #                     help='evaluuation interval (default: 1)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']  # the root to log your train/val status


    BACKBONE_NAME = cfg['BACKBONE_NAME'] 
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES']  # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    writer = SummaryWriter(LOG_ROOT)

    # ======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE),
                     'ResNet_101': ResNet_101(INPUT_SIZE),
                     'ResNet_152': ResNet_152(INPUT_SIZE),
                     'IR_50': IR_50(INPUT_SIZE),
                     'IR_101': IR_101(INPUT_SIZE),
                     'IR_152': IR_152(INPUT_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE),
                     'IR_SE_101': IR_SE_101(INPUT_SIZE),
                     'IR_SE_152': IR_SE_152(INPUT_SIZE)}

    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'SphereFace': SphereFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'Am_softmax': Am_softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID)}

    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)


    LOSS_DICT = {'Focal': FocalLoss(),
                 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        # separate batch_norm parameters from others; 
        # do not do weight decay for batch_norm parameters to improve the generalizability
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE)  
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(
            BACKBONE)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)

    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                           {'params': backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)


    # ======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 100  # frequency to display training loss & acc

    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    batch = 0  # batch index


    for epoch in range(NUM_EPOCH):  # start training process

        if epoch == STAGES[0]:
            # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        BACKBONE.train()  # set to training mode
        HEAD.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        for inputs, labels in tqdm(iter(train_loader)):

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                    batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            loss = LOSS(outputs, labels)

             # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss=losses, top1=top1, top5=top5))
                print("=" * 60)

            batch += 1  # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, NUM_EPOCH, loss=losses, top1=top1, top5=top5))
        print("=" * 60)

        # save checkpoints per epoch
        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT,
                                                                  "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                                      BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT,
                                                       "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                           HEAD_NAME, epoch + 1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT,
                                                           "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                               BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT,
                                                       "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                           HEAD_NAME, epoch + 1, batch, get_time())))




    # args = parse_args()
    # prefix_dir = os.path.join(args.models_root, '%s'%(args.network))

    # if not os.path.exists(prefix_dir):
    #     os.makedirs(prefix_dir)

    # args.batch_size = 2 * default.batch_size
    # # args.resume = os.path.join(prefix_dir, 'arcface_393_350000_0.611328125')
    # args.num_classes = config.num_classes
    # trainer = Trainer(args)
    # print('num_classes', config.num_classes)


    # for epoch in range(1, args.end_epoch):
    #     trainer.training(epoch)
    #     # if epoch % args.eval_interval == (args.eval_interval - 1):
    #     # trainer.validation(epoch)