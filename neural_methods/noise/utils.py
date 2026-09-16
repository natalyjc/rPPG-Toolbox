import numpy as np
import torch


def subject_seed(subject_id, base_seed):
    try:
        subject_offset = int(subject_id)
    except (TypeError, ValueError):
        subject_offset = sum(ord(char) for char in str(subject_id))
    return int(base_seed) + int(subject_offset)


def is_noise_target(config, subject_id):
    noise_config = config.INFERENCE.FEW_SHOT.NOISE
    if not noise_config.ENABLE:
        return False
    if str(subject_id) in {str(subject) for subject in noise_config.SUBJECT_IDS}:
        return True
    return False


def apply_support_noise(config, support_data, subject_id):
    noise_config = config.INFERENCE.FEW_SHOT.NOISE
    if not noise_config.ENABLE or float(noise_config.STD) <= 0.0:
        return support_data
    generator = torch.Generator()
    generator.manual_seed(subject_seed(subject_id, noise_config.SEED))
    noise = torch.randn(
        support_data.shape,
        generator=generator,
        device=support_data.device,
        dtype=support_data.dtype,
    ) * float(noise_config.STD)
    return support_data + noise