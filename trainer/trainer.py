import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from os import path
from librosa.feature import inverse
from librosa import core
from scipy.io import wavfile


class SpecVaeTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(SpecVaeTrainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.logger.info(f"Training set size: {len(self.data_loader)}")
        if self.do_validation:
            self.logger.info(f"Validation set size: {len(self.valid_data_loader)}")

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _reshape(self, x):
        n_batch, n_chunk, n_freqBand, n_contextWin = x.size()
        return x.view(-1, n_freqBand, n_contextWin)

    def _forward_and_computeLoss(self, x, target):
        x_recon, zdist = self.model(x)
        loss = self.loss(x_recon, target, zdist)
        return loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_log_loss = self.loss.get_loss_dict()
        # total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data_idx, label, data) in enumerate(self.data_loader):
            x = data.type('torch.FloatTensor').to(self.device)
            x = self._reshape(x)

            self.optimizer.zero_grad()
            loss = self._forward_and_computeLoss(x, x)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_log_loss += self.loss.log_loss
            # total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # TODO: visualize input/reconstructed spectrograms in TensorBoard
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            k: v / len(self.data_loader) for k, v in total_log_loss.items()
        }
        # 'metrics': (total_metrics / len(self.data_loader)).tolist()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_log_loss = self.loss.get_loss_dict()
        # total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data_idx, label, data) in enumerate(self.valid_data_loader):
                x = data.type('torch.FloatTensor').to(self.device)
                x = self._reshape(x)
                loss = self._forward_and_computeLoss(x, x)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_log_loss += self.loss.log_loss
                # total_val_metrics += self._eval_metrics(output, target)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = {
            "val_" + k: v / len(self.valid_data_loader) for k, v in total_val_log_loss.items()
        }
        return log

    def train(self):
        super().train()
        self.save_sample()

    def save_sample(self, n=5):
        for idxs, labels, data in self.data_loader:
            labels = labels[:n]
            data = data[:n]
            break
        x = data.type('torch.FloatTensor').to(self.device)
        o_size = x.size()
        x = self._reshape(x)
        x_recon, *_ = self.model(x)
        # Revert the shape
        x = x.view(*o_size).detach().cpu().numpy()
        x_recon = x_recon.view(*o_size).detach().cpu().numpy()
        # Join chunks
        x = x.transpose(0, 2, 1, 3).reshape(o_size[0], o_size[2], -1)
        x_recon = x_recon.transpose(0, 2, 1, 3).reshape(o_size[0], o_size[2], -1)
        for i in range(x.shape[0]):
            idx = idxs[i]
            fpath = self.data_loader.dataset.path_to_data[idx]
            fname = path.splitext(path.basename(fpath))[0]
            # save spectrogram
            np.save(path.join(self.config.sample_dir, fname + ".npy"), x[i])
            np.save(path.join(self.config.sample_dir, fname + "_recon.npy"), x_recon[i])

            try:
                # save waveform
                au = inverse.mel_to_audio(core.db_to_power(x[i]), sr=22050, n_fft=2048, hop_length=735)
                recon_au = inverse.mel_to_audio(core.db_to_power(x_recon[i]), sr=22050, n_fft=2048, hop_length=735)
                rate = len(au) // 5
                wavfile.write(path.join(self.config.sample_dir, fname + ".wav"), rate, au)
                wavfile.write(path.join(self.config.sample_dir, fname + "_recon.wav"), rate, recon_au)
            except Exception as e:
                self.logger.debug(str(e))
                self.logger.info("Cannot save from mel to audio")
