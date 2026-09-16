"""Trainer for RhythmFormer Few-Shot Adaption at Inference"""
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from evaluation.post_process import *
from evaluation.metrics import calculate_metrics
from evaluation.few_shot_metrics import calculate_few_shot_metrics, save_few_shot_plots, save_few_shot_reports
from neural_methods.model.RhythmFormer import RhythmFormer
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.RythmFormerLossComputer import RhythmFormer_Loss
from neural_methods.few_shot.utils import collect_subject_sequences, adapt_model_on_support
from neural_methods.noise.utils import is_noise_target, apply_support_noise

class RhythmFormerTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = RhythmFormer().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.num_train_batches = len(data_loader["train"])
            self.criterion = RhythmFormer_Loss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = RhythmFormer().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.criterion = RhythmFormer_Loss()
        else:
            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()

            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                pred_ppg = (pred_ppg-torch.mean(pred_ppg, axis=-1).view(-1, 1))/torch.std(pred_ppg, axis=-1).view(-1, 1)    # normalize

                loss = 0.0
                for ib in range(N):
                    loss = loss + self.criterion(pred_ppg[ib], labels[ib], epoch , self.config.TRAIN.DATA.FS , self.diff_flag)
                loss = loss / N
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)


    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                pred_ppg_valid = self.model(data_valid)
                pred_ppg_valid = (pred_ppg_valid-torch.mean(pred_ppg_valid, axis=-1).view(-1, 1))/torch.std(pred_ppg_valid, axis=-1).view(-1, 1)    # normalize
                for ib in range(N):
                    loss = self.criterion(pred_ppg_valid[ib], labels_valid[ib], self.config.TRAIN.EPOCHS , self.config.VALID.DATA.FS , self.diff_flag)
                    valid_loss.append(loss.item())
                    valid_step += 1
                    vbar.set_postfix(loss=loss.item())
        return np.mean(valid_loss)
    
    def test_few_shot(self, data_loader):
        """Few-shot adaptation testing on support/query splits for each subject."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
            raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

        print('')
        print("===Few-Shot Testing (RhythmFormer)===" )

        self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Loaded baseline pretrained model for few-shot adaptation.")
        if self.config.INFERENCE.FEW_SHOT.NOISE.ENABLE:
            print(
                "Few-shot noise config: "
                f"subjects={list(self.config.INFERENCE.FEW_SHOT.NOISE.SUBJECT_IDS)}, "
                f"std={self.config.INFERENCE.FEW_SHOT.NOISE.STD}, "
                f"seed={self.config.INFERENCE.FEW_SHOT.NOISE.SEED}"
            )

        diff_flag = self.config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized"
        support_frames = int(self.config.INFERENCE.FEW_SHOT.SUPPORT_FRAMES)
        adapt_steps = int(self.config.INFERENCE.FEW_SHOT.ADAPT_STEPS)
        window_frames = int(self.config.INFERENCE.FEW_SHOT.WINDOW_FRAMES)
        subject_sequences = collect_subject_sequences(data_loader['test'])
        subject_ids = sorted(subject_sequences.keys())

        gt_hr_all = []
        baseline_hr_all = []
        adapted_hr_all = []
        baseline_snr_all = []
        adapted_snr_all = []
        noisy_subject_ids = []
        subject_query_window_counts = dict()
        per_subject_metrics = dict()
        baseline_predictions = dict()
        adapted_predictions = dict()
        all_labels = dict()

        for subj_index in tqdm(subject_ids, ncols=80):

            subject_data = subject_sequences[subj_index]["data"]
            subject_labels = subject_sequences[subj_index]["labels"]
            if subject_data.shape[0] <= support_frames:
                print(f"Warning: Subject {subj_index} has only {subject_data.shape[0]} frames and will be skipped.")
                continue

            support_data = subject_data[:support_frames]
            support_labels = subject_labels[:support_frames]
            query_data = subject_data[support_frames:]
            query_labels = subject_labels[support_frames:]
            query_window_count = query_data.shape[0] // window_frames
            if query_window_count < 1:
                print(f"Warning: Subject {subj_index} has no full query windows and will be skipped.")
                continue

            subject_gt_hr = []
            subject_baseline_hr = []
            subject_adapted_hr = []
            subject_baseline_snr = []
            subject_adapted_snr = []
            subject_baseline_preds = []
            subject_adapted_preds = []
            subject_label_windows = []

            # Few-Shot Baseline Model Query Window Eval
            for window_idx in range(query_window_count):
                start = window_idx * window_frames
                end = start + window_frames
                query_window_data = query_data[start:end]
                query_window_label = query_labels[start:end].numpy()

                with torch.no_grad():
                    input_data = query_window_data.unsqueeze(0).to(self.config.DEVICE)
                    pred_ppg = self.model(input_data)
                    pred_ppg = (pred_ppg - torch.mean(pred_ppg, axis=-1).view(-1, 1)) / (torch.std(pred_ppg, axis=-1).view(-1, 1) + 1e-8)
                    baseline_pred_window = pred_ppg.squeeze(0).detach().cpu().numpy()
                    gt_hr, baseline_hr, baseline_snr, _ = calculate_metric_per_video(baseline_pred_window, query_window_label, diff_flag=diff_flag, fs=self.config.TEST.DATA.FS, hr_method="FFT")

                gt_hr_all.append(gt_hr)
                baseline_hr_all.append(baseline_hr)
                baseline_snr_all.append(baseline_snr)
                subject_gt_hr.append(gt_hr)
                subject_baseline_hr.append(baseline_hr)
                subject_baseline_snr.append(baseline_snr)
                subject_baseline_preds.append(baseline_pred_window)
                subject_label_windows.append(query_window_label)

            # Exp_2 & Exp_4: Applying Noise to Worse MAE Improved Subjects
            should_corrupt_support = is_noise_target(self.config, subj_index)
            if should_corrupt_support:
                support_data_for_adaptation = apply_support_noise(self.config, support_data, subj_index)
                noisy_subject_ids.append(subj_index)
            else:
                support_data_for_adaptation = support_data
                
            adapted_model = adapt_model_on_support(copy.deepcopy(self.model), support_data_for_adaptation, support_labels, adapt_steps, diff_flag, window_frames, self.config, self.criterion)

            # Few-Shot Adapted Model Query Window Eval
            for window_idx in range(query_window_count):
                start = window_idx * window_frames
                end = start + window_frames
                query_window_data = query_data[start:end]
                query_window_label = query_labels[start:end].numpy()

                with torch.no_grad():
                    input_data = query_window_data.unsqueeze(0).to(self.config.DEVICE)
                    pred_ppg = adapted_model(input_data)
                    pred_ppg = (pred_ppg - torch.mean(pred_ppg, axis=-1).view(-1, 1)) / (torch.std(pred_ppg, axis=-1).view(-1, 1) + 1e-8)
                    adapted_pred_window = pred_ppg.squeeze(0).detach().cpu().numpy()
                    _, adapted_hr, adapted_snr, _ = calculate_metric_per_video(adapted_pred_window, query_window_label, diff_flag=diff_flag, fs=self.config.TEST.DATA.FS, hr_method="FFT")

                adapted_hr_all.append(adapted_hr)
                adapted_snr_all.append(adapted_snr)
                subject_adapted_hr.append(adapted_hr)
                subject_adapted_snr.append(adapted_snr)
                subject_adapted_preds.append(adapted_pred_window)

            subject_query_window_counts[subj_index] = int(query_window_count)
            baseline_predictions[subj_index] = np.concatenate(subject_baseline_preds, axis=0)
            adapted_predictions[subj_index] = np.concatenate(subject_adapted_preds, axis=0)
            all_labels[subj_index] = np.concatenate(subject_label_windows, axis=0)

            subject_baseline_metrics, subject_adapted_metrics = calculate_few_shot_metrics(subject_gt_hr, subject_baseline_hr, subject_adapted_hr, subject_baseline_snr, subject_adapted_snr)
            per_subject_metrics[subj_index] = {
                "query_window_count": int(query_window_count),
                "baseline": subject_baseline_metrics,
                "adapted": subject_adapted_metrics,
                "noise_targeted": bool(should_corrupt_support),
            }

        baseline_metrics, adapted_metrics = calculate_few_shot_metrics(gt_hr_all, baseline_hr_all, adapted_hr_all, baseline_snr_all, adapted_snr_all)
        save_few_shot_plots(gt_hr_all, baseline_hr_all, adapted_hr_all, self.config)

        results = {
            "support_frames": support_frames,
            "adapt_steps": adapt_steps,
            "window_frames": window_frames,
            "num_subjects": len(subject_query_window_counts),
            "baseline": baseline_metrics,
            "adapted": adapted_metrics,
            "subject_query_window_counts": subject_query_window_counts,
            "per_subject_metrics": per_subject_metrics,
            "noisy_subject_ids": noisy_subject_ids,
            "noise_config": {
                "enable": bool(self.config.INFERENCE.FEW_SHOT.NOISE.ENABLE),
                "subject_ids": list(self.config.INFERENCE.FEW_SHOT.NOISE.SUBJECT_IDS),
                "std": float(self.config.INFERENCE.FEW_SHOT.NOISE.STD),
                "seed": int(self.config.INFERENCE.FEW_SHOT.NOISE.SEED),
            },
        }
        save_few_shot_reports(results, baseline_predictions, adapted_predictions, all_labels, self.config)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")

        # Change chunk length to be test chunk length
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            predictions = dict()
            labels = dict()
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                chunk_len = self.chunk_len
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.model(data_test)
                pred_ppg_test = (pred_ppg_test-torch.mean(pred_ppg_test, axis=-1).view(-1, 1))/torch.std(pred_ppg_test, axis=-1).view(-1, 1)    # normalize
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = pred_ppg_test.view( -1 , 1)
                for ib in range(batch_size):
                    subj_index = test_batch[2][ib]
                    sort_index = int(test_batch[3][ib])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[ib * chunk_len:(ib + 1) * chunk_len]
                    labels[subj_index][sort_index] = labels_test[ib * chunk_len:(ib + 1) * chunk_len]
            print(' ')
            calculate_metrics(predictions, labels, self.config)
            self.save_predictions_readable(predictions, labels, os.path.join(self.config.TEST.OUTPUT_SAVE_DIR, 'readable'))
            if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
                self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
