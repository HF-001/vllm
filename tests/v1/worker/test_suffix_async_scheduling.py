# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for suffix speculative decoding + async scheduling helpers
in GPUModelRunner._convert_draft_list_to_gpu_tensor.

The triton-kernel-based methods (_suffix_rejection_sample,
_do_suffix_drafting_async, _suffix_handle_not_fits) are tested via
the e2e spec decode tests (test_spec_decode.py, test_async_spec_decode.py).
"""

from unittest.mock import MagicMock

import torch


# ---------------------------------------------------------------------------
# Lightweight fake that mimics just enough of the real GPUModelRunner
# ---------------------------------------------------------------------------

class FakeRunner:
    """Minimal object that has the attributes used by
    _convert_draft_list_to_gpu_tensor."""

    def __init__(self, num_reqs: int, num_spec_tokens: int = 3,
                 device: str = "cpu"):
        self.input_batch = MagicMock()
        self.input_batch.req_ids = [f"req_{i}" for i in range(num_reqs)]
        self.num_spec_tokens = num_spec_tokens
        self.device = torch.device(device)
        self._draft_token_ids = None

        # Pinned CPU buffer for list→GPU conversion
        self.cpu_proposer_draft_pinned = torch.zeros(
            (num_reqs, num_spec_tokens), dtype=torch.int32,
        )

    # Import the real method under test
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    _convert_draft_list_to_gpu_tensor = (
        GPUModelRunner._convert_draft_list_to_gpu_tensor
    )


# ---------------------------------------------------------------------------
# Tests for _convert_draft_list_to_gpu_tensor
# ---------------------------------------------------------------------------

class TestConvertDraftListToGpuTensor:

    def test_list_draft_tokens_converted_to_tensor(self):
        """List-based draft tokens should become a padded tensor."""
        runner = FakeRunner(num_reqs=3, num_spec_tokens=3)
        runner._draft_token_ids = [
            [100, 101, 102],  # full 3 drafts
            [200],            # only 1 draft
            [],               # no drafts
        ]

        runner._convert_draft_list_to_gpu_tensor()

        assert isinstance(runner._draft_token_ids, torch.Tensor)
        assert runner._draft_token_ids.shape == (3, 3)
        assert runner._draft_token_ids.dtype == torch.int32
        # Row 0: [100, 101, 102]
        assert runner._draft_token_ids[0].tolist() == [100, 101, 102]
        # Row 1: [200, 0, 0] (padded)
        assert runner._draft_token_ids[1].tolist() == [200, 0, 0]
        # Row 2: [0, 0, 0] (all padding)
        assert runner._draft_token_ids[2].tolist() == [0, 0, 0]

    def test_tensor_draft_tokens_unchanged(self):
        """If draft tokens are already a tensor, leave them as-is."""
        runner = FakeRunner(num_reqs=2, num_spec_tokens=3)
        original = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        runner._draft_token_ids = original

        runner._convert_draft_list_to_gpu_tensor()

        assert runner._draft_token_ids is original  # same object

    def test_draft_tokens_longer_than_num_spec_tokens_truncated(self):
        """Draft tokens exceeding num_spec_tokens should be truncated."""
        runner = FakeRunner(num_reqs=1, num_spec_tokens=2)
        runner._draft_token_ids = [[100, 101, 102, 103]]  # 4 tokens, limit 2

        runner._convert_draft_list_to_gpu_tensor()

        assert runner._draft_token_ids.shape == (1, 2)
        assert runner._draft_token_ids[0].tolist() == [100, 101]

    def test_none_draft_tokens_unchanged(self):
        """If draft tokens are None, should not crash."""
        runner = FakeRunner(num_reqs=1, num_spec_tokens=2)
        runner._draft_token_ids = None

        runner._convert_draft_list_to_gpu_tensor()

        assert runner._draft_token_ids is None
