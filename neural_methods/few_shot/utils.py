import torch
from tqdm import tqdm
import torch.optim as optim

def collect_subject_sequences(test_loader):
    subject_chunks = {}
    print("Collecting subject clips for few-shot support/query split...")
    for test_batch in tqdm(test_loader, ncols=80):
        for batch_index in range(test_batch[0].shape[0]):
            subject = test_batch[2][batch_index]
            if torch.is_tensor(subject):
                subject = subject.item()
            subject = str(subject)
            chunk_index = test_batch[3][batch_index]
            if torch.is_tensor(chunk_index):
                chunk_index = int(chunk_index.item())
            else:
                chunk_index = int(chunk_index)
            subject_chunks.setdefault(subject, {})[chunk_index] = (
                test_batch[0][batch_index].float().cpu(),
                test_batch[1][batch_index].float().cpu(),
            )
    output = {
        subject: {
            "data": torch.cat(
                [pair[0] for _, pair in sorted(chunks.items())],
                dim=0
            ),
            "labels": torch.cat(
                [pair[1] for _, pair in sorted(chunks.items())],
                dim=0
            ),
        }
        for subject, chunks in subject_chunks.items()
    }

    return output

def build_support_windows(support_data, support_labels, window_frames):
    if support_data.shape[0] < window_frames:
        raise ValueError(
            f"Support region has {support_data.shape[0]} frames, which is shorter than required window size {window_frames}."
        )
    window_starts = list(range(0, support_data.shape[0] - window_frames + 1, window_frames))
    final_start = support_data.shape[0] - window_frames
    if window_starts[-1] != final_start:
        window_starts.append(final_start)
    return [
        (support_data[start:start + window_frames], support_labels[start:start + window_frames])
        for start in window_starts
    ]

def adapt_model_on_support(model, support_data, support_labels, adapt_steps, diff_flag, window_frames, config, criterion):
    if adapt_steps <= 0:
        model.eval()
        return model

    optimizer = optim.AdamW(model.parameters(), lr=config.INFERENCE.FEW_SHOT.LR, weight_decay=config.INFERENCE.FEW_SHOT.WEIGHT_DECAY)
    support_windows = build_support_windows(support_data, support_labels, window_frames)

    model.train()
    for step in range(adapt_steps):
        optimizer.zero_grad()
        total_loss = 0.0
        for support_window_data, support_window_labels in support_windows:
            support_input = support_window_data.unsqueeze(0).to(config.DEVICE)
            support_target = support_window_labels.to(config.DEVICE)
            pred_ppg = model(support_input)
            pred_ppg = (pred_ppg - torch.mean(pred_ppg, axis=-1).view(-1, 1)) / (torch.std(pred_ppg, axis=-1).view(-1, 1) + 1e-8)
            total_loss = total_loss + criterion(pred_ppg[0], support_target, step, config.TEST.DATA.FS, diff_flag)

        loss = total_loss / len(support_windows)
        loss.backward()
        optimizer.step()

    model.eval()
    return model