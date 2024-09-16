import torch
import os
import time
from utils.func import IOStream
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.segmentors import EISNet
from torch.optim import AdamW
from utils.scheduler import PolynomialLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.metrics import MetricsSemseg
from utils.labels import dataset_info


class Trainer():

    def __init__(self, cfg):

        self.cfg = cfg

        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=self.cfg['TRAIN']['log_dir'])

        # Initialize log printer
        self.printer = IOStream(self.cfg['TRAIN']['log_dir'] + '/run.log')
        print(str(self.cfg))
        self.printer.cprint(str(self.cfg))

        # Dataset and loader
        semseg_ignore_label, semseg_class_names, _ = dataset_info(semseg_num_classes=self.cfg['DATASET']['classes'])
        if self.cfg['DATASET']['name'] == 'DDD17Event':
            from datasets.ddd17_dataset import DDD17Event
            self.training_dataset = DDD17Event(root=self.cfg['DATASET']['path'], split='train',
                                               event_representation=self.cfg['DATASET']['event_representation'],
                                               nr_events_data=1, delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                               nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                               require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                               augmentation=True, fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                               random_crop=True)
            self.validation_dataset = DDD17Event(root=self.cfg['DATASET']['path'], split='test',
                                                 event_representation=self.cfg['DATASET']['event_representation'],
                                                 nr_events_data=1, delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                                 nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                                 require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                                 augmentation=False, fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                                 random_crop=False)
            self.training_loader = DataLoader(self.training_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                              batch_size=self.cfg['TRAIN']['batch_size'], shuffle=True)
            self.validation_loader = DataLoader(self.validation_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                                batch_size=self.cfg['TRAIN']['batch_size'])
        elif self.cfg['DATASET']['name'] == 'DSECEvent':
            from datasets.dsec_dataset import DSECEvent
            self.training_dataset = DSECEvent(self.cfg['DATASET']['path'], nr_events_data=1,
                                              nr_events_window=self.cfg['DATASET']['nr_events'], augmentation=True,
                                              mode='train', event_representation=self.cfg['DATASET']['event_representation'],
                                              nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                              require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                              semseg_num_classes=self.cfg['DATASET']['classes'],
                                              fixed_duration=self.cfg['DATASET']['fixed_duration'], random_crop=True)
            self.validation_dataset = DSECEvent(self.cfg['DATASET']['path'], nr_events_data=1,
                                                nr_events_window=self.cfg['DATASET']['nr_events'], augmentation=False,
                                                mode='val', event_representation=self.cfg['DATASET']['event_representation'],
                                                nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                                require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                                semseg_num_classes=self.cfg['DATASET']['classes'],
                                                fixed_duration=self.cfg['DATASET']['fixed_duration'], random_crop=False)
            self.training_loader = DataLoader(self.training_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                              batch_size=self.cfg['TRAIN']['batch_size'], shuffle=True)
            self.validation_loader = DataLoader(self.validation_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                                batch_size=self.cfg['TRAIN']['batch_size'])

        # Model
        self.model = EISNet(ver_ev=self.cfg['MODEL']['version_ev'], ver_img=self.cfg['MODEL']['version_img'] ,
                            num_classes=self.cfg['DATASET']['classes'], aet_rep=self.cfg['MODEL']['aet_rep'],
                            num_channels_ev=self.cfg['DATASET']['nr_bins'], num_channels_img=self.cfg['DATASET']['img_chnls'],
                            pretrained_ev=self.cfg['MODEL']['pretrained_ev'], pretrained_img=self.cfg['MODEL']['pretrained_img'],
                            weight_path=self.cfg['MODEL']['pretrained_path'])

        # Optimizer, learning rate scheduler and loss function
        self.opt = AdamW(params=self.model.parameters(), lr=self.cfg['TRAIN']['lr_init'])
        total_iters = int(self.cfg['TRAIN']['num_epochs'] * len(self.training_loader))
        self.scheduler = PolynomialLR(optimizer=self.opt, total_iters=total_iters)
        self.criterion = CrossEntropyLoss(ignore_index=semseg_ignore_label)

        # Put the model into the computing device
        self.model.to(self.cfg['DEVICE'])

        # Evaluation metrics
        self.metrics = MetricsSemseg(self.cfg['DATASET']['classes'], semseg_ignore_label, semseg_class_names)

    def train(self):

        self.backup()

        best_mIOU = 0.0

        start = time.time()
        for epoch_id in range(self.cfg['TRAIN']['num_epochs']):
            print("Training step [{:3d}/{:3d}]".format(epoch_id + 1, self.cfg['TRAIN']['num_epochs']))
            self.train_epoch(epoch_id)
            if (epoch_id + 1) >= 1:
                print("Testing step [{:3d}/{:3d}]".format(epoch_id + 1, self.cfg['TRAIN']['num_epochs']))
                best_mIOU = self.eval(epoch_id, best_mIOU)
        self.writer.close()
        end = time.gmtime(time.time() - start)
        print('Total training time is:', time.strftime("%H:%M:%S", end))

    def train_epoch(self, epoch_id):

        training_loss = 0.0
        count = 0

        self.model.train()
        print('Current learning rate: %e'%(self.opt.state_dict()['param_groups'][0]['lr']))
        for ev_rep, img, label in tqdm(self.training_loader):
            ev_rep, img, label = ev_rep.type(torch.FloatTensor).to(self.cfg['DEVICE']),\
                img.type(torch.FloatTensor).to(self.cfg['DEVICE']), label.to(self.cfg['DEVICE'])
            # Forward propagation and back propagation
            self.opt.zero_grad()
            pred = self.model(ev_rep, img)
            pred_label = pred.argmax(dim=1)
            loss = self.criterion(pred, label)
            loss.backward()
            self.opt.step()
            self.scheduler.step()
            # Statistics
            count += self.cfg['TRAIN']['batch_size']
            training_loss += loss.item() * self.cfg['TRAIN']['batch_size']
            self.metrics.update_batch(pred_label, label)
        # Logger
        scores = self.metrics.get_metrics_summary()
        print("Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(training_loss * 1.0 / count, scores['mean_iou'],
                                                                    scores['acc']))
        log_str = "[Train]  Epoch: {:d}, Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(epoch_id + 1,
                                                                                              training_loss * 1.0 / count,
                                                                                              scores['mean_iou'],
                                                                                              scores['acc'])
        self.printer.cprint(log_str)
        self.writer.add_scalar(self.cfg['TRAIN']['log_dir'] + 'train_loss', training_loss * 1.0 / count, epoch_id + 1)
        self.writer.add_scalar(self.cfg['TRAIN']['log_dir'] + 'train_mIOU', scores['mean_iou'], epoch_id + 1)
        self.writer.add_scalar(self.cfg['TRAIN']['log_dir'] + 'train_acc', scores['acc'], epoch_id + 1)
        # Save the model
        if (epoch_id + 1) % self.cfg['TRAIN']['save_every_n_epochs'] == 0:
            torch.save({'epoch': epoch_id + 1, 'state_dict': self.model.state_dict()},
                       os.path.join(self.cfg['TRAIN']['log_dir'], 'checkpoint_epoch_' + str((epoch_id + 1)) + '.pth'))
            print("Save the model at {}".format(os.path.join(self.cfg['TRAIN']['log_dir'], 'checkpoint_epoch_' + str((epoch_id + 1)) + '.pth')))
        # Reset the evaluation metrics
        self.metrics.reset()

    def eval(self, epoch_id, best_mIOU):

        testing_loss = 0.0
        count = 0

        self.model.eval()

        with torch.no_grad():
            for ev_rep, img, label in tqdm(self.validation_loader):
                ev_rep, img, label = ev_rep.type(torch.FloatTensor).to(self.cfg['DEVICE']),\
                    img.type(torch.FloatTensor).to(self.cfg['DEVICE']), label.to(self.cfg['DEVICE'])
                pred = self.model(ev_rep, img)
                pred_label = pred.argmax(dim=1)
                loss = self.criterion(pred, label)
                # Statistics
                count += self.cfg['TRAIN']['batch_size']
                testing_loss += loss.item() * self.cfg['TRAIN']['batch_size']
                self.metrics.update_batch(pred_label, label)
        # Logger
        scores = self.metrics.get_metrics_summary()
        print("Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(testing_loss * 1.0 / count,
                                                                    scores['mean_iou'], scores['acc']))
        log_str = "[Test]   Epoch: {:d}, Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(epoch_id + 1,
                                                                                              testing_loss * 1.0 / count,
                                                                                              scores['mean_iou'],
                                                                                              scores['acc'])
        self.printer.cprint(log_str)
        self.writer.add_scalar(self.cfg['TRAIN']['log_dir'] + 'test_loss', testing_loss * 1.0 / count, epoch_id + 1)
        self.writer.add_scalar(self.cfg['TRAIN']['log_dir'] + 'test_mIOU', scores['mean_iou'], epoch_id + 1)
        self.writer.add_scalar(self.cfg['TRAIN']['log_dir'] + 'test_acc', scores['acc'], epoch_id + 1)
        # Save the model
        if scores['mean_iou'] >= best_mIOU:
            best_mIOU = scores['mean_iou']
            torch.save({'epoch': epoch_id + 1, 'state_dict': self.model.state_dict()},
                       os.path.join(self.cfg['TRAIN']['log_dir'], 'best_model' + '.pth'))
            print("Save the best model at {}".format(os.path.join(self.cfg['TRAIN']['log_dir'], 'best_model' + '.pth')))
            print('New best mIOU is %.4f'%best_mIOU)
            self.printer.cprint('New best mIOU is %.4f'%best_mIOU)
        # Reset the evaluation metrics
        self.metrics.reset()
        return best_mIOU

    def backup(self):
        root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        if not os.path.exists(self.cfg['TRAIN']['log_dir'] + '/' + 'Backup'):
            os.makedirs(self.cfg['TRAIN']['log_dir'] + '/' + 'Backup')
        model_saving_dir = os.path.join(self.cfg['TRAIN']['log_dir'], 'Backup')
        os.system('cp %s '%(os.path.join(root, 'configs/DDD17.yaml')) + model_saving_dir + '/' + 'DDD17.yaml.backup')
        os.system('cp %s '% (os.path.join(root, 'configs/DSEC_Semantic.yaml')) + model_saving_dir + '/' + 'DSEC_Semantic.yaml.backup')
        os.system('cp %s '%(os.path.join(root, 'utils/trainer.py')) + model_saving_dir + '/' + 'trainer.py.backup')
        os.system('cp %s '%(os.path.join(root, 'models/modules.py')) + model_saving_dir + '/' + 'modules.py.backup')
        os.system('cp %s '%(os.path.join(root, 'models/segmentors.py')) + model_saving_dir + '/' + 'segmentors.py.backup')
        os.system('cp %s '%(os.path.join(root, 'datasets/data_util.py')) + model_saving_dir + '/' + 'data_util.py.backup')
        os.system('cp %s '%(os.path.join(root, 'datasets/ddd17_dataset.py')) + model_saving_dir + '/' + 'ddd17_dataset.py.backup')
        os.system('cp %s '%(os.path.join(root, 'datasets/extract_data_tools/DSEC/sequence.py')) + model_saving_dir + '/' + 'sequence.py.backup')
