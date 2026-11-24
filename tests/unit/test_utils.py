import pytest
import torch

from megatron.utils import get_ltor_masks_and_position_ids


@pytest.mark.cpu
def test_get_ltor_masks_eod_mask_loss_without_eod_token():
    """eod_mask_loss with no end-of-document token must fail loudly instead
    of silently masking nothing (GPT-NeoX #1498)."""
    data = torch.tensor([[10, 11, 12, 13]])
    with pytest.raises(ValueError, match="no end-of-document token"):
        get_ltor_masks_and_position_ids(data, eod_token=None, eod_mask_loss=True)
