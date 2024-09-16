import torch
from torch.utils.data import DataLoader
from models.segmentors import EISNet
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.metrics import MetricsSemseg
from utils.labels import dataset_info


class Evaluator():

    def __init__(self, cfg):

        self.cfg = cfg
        print(str(self.cfg))

        # Dataset and loader
        semseg_ignore_label, semseg_class_names, _ = dataset_info(semseg_num_classes=self.cfg['DATASET']['classes'])
        if self.cfg['DATASET']['name'] == 'DDD17Event':
            from datasets.ddd17_dataset import DDD17Event
            self.testing_dataset = DDD17Event(root=self.cfg['DATASET']['path'], split='test',
                                              event_representation=self.cfg['DATASET']['event_representation'],
                                              nr_events_data=1, delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                              nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                              require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                              augmentation=False, fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                              random_crop=False)
            self.testing_loader = DataLoader(self.testing_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                             batch_size=self.cfg['EVAL']['batch_size'])
        elif self.cfg['DATASET']['name'] == 'DSECEvent':
            from datasets.dsec_dataset import DSECEvent
            self.testing_dataset = DSECEvent(self.cfg['DATASET']['path'], nr_events_data=1,
                                             nr_events_window=self.cfg['DATASET']['nr_events'], augmentation=False,
                                             mode='val', event_representation=self.cfg['DATASET']['event_representation'],
                                             nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                             require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                             semseg_num_classes=self.cfg['DATASET']['classes'],
                                             fixed_duration=self.cfg['DATASET']['fixed_duration'], random_crop=False)
            self.testing_loader = DataLoader(self.testing_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                             batch_size=self.cfg['EVAL']['batch_size'])

        # Model
        self.model = EISNet(ver_ev=self.cfg['MODEL']['version_ev'], ver_img=self.cfg['MODEL']['version_img'],
                            num_classes=self.cfg['DATASET']['classes'], aet_rep=self.cfg['MODEL']['aet_rep'],
                            num_channels_ev=self.cfg['DATASET']['nr_bins'], num_channels_img=self.cfg['DATASET']['img_chnls'],
                            pretrained_ev=False, pretrained_img=False, weight_path=self.cfg['MODEL']['pretrained_path'])

        # Loss function
        self.criterion = CrossEntropyLoss(ignore_index=semseg_ignore_label)

        # Put the model into the computing device
        print('Load weights from the checkpoint:', self.cfg['EVAL']['weight_path'])
        checkpoint = torch.load(self.cfg['EVAL']['weight_path'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.cfg['DEVICE'])

        # Evaluation metrics
        self.metrics = MetricsSemseg(self.cfg['DATASET']['classes'], semseg_ignore_label, semseg_class_names)

    def eval(self):

        testing_loss = 0.0
        count = 0

        self.model.eval()

        with torch.no_grad():
            for ev_rep, img, label in tqdm(self.testing_loader):
                ev_rep, img, label = ev_rep.type(torch.FloatTensor).to(self.cfg['DEVICE']),\
                    img.type(torch.FloatTensor).to(self.cfg['DEVICE']), label.to(self.cfg['DEVICE'])
                pred = self.model(ev_rep, img)
                pred_label = pred.argmax(dim=1)
                loss = self.criterion(pred, label)
                # Statistics
                count += self.cfg['EVAL']['batch_size']
                testing_loss += loss.item() * self.cfg['EVAL']['batch_size']
                self.metrics.update_batch(pred_label, label)
        # Logger
        scores = self.metrics.get_metrics_summary()
        print("Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(testing_loss * 1.0 / count,
                                                                    scores['mean_iou'], scores['acc']))
        # Reset the evaluation metrics
        self.metrics.reset()
