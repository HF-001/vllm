# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for suffix speculative decoding + async scheduling fixes
in GPUModelRunner._sync_for_cpu_proposer and _finalize_cpu_proposer_async.

These tests mock the GPUModelRunner to isolate the logic of the two new
helper methods without requiring a full GPU or model setup.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Lightweight fakes that mimic just enough of the real objects
# ---------------------------------------------------------------------------

@dataclass
class FakeCachedRequestState:
    req_id: str
    output_token_ids: list[int] = field(default_factory=list)
    prev_num_draft_len: int = 0


class FakeInputBatch:
    """Minimal stand-in for InputBatch used by the methods under test."""

    def __init__(self, num_reqs: int, max_model_len: int, vocab_size: int = 32000):
        self.req_ids: list[str] = [f"req_{i}" for i in range(num_reqs)]
        self.token_ids_cpu = np.full(
            (num_reqs, max_model_len), -1, dtype=np.int64
        )
        self.is_token_ids = np.zeros(
            (num_reqs, max_model_len), dtype=bool
        )
        self.num_tokens_no_spec = np.zeros(num_reqs, dtype=np.int64)
        self.vocab_size = vocab_size
        self.prev_sampled_token_ids = None
        self.prev_req_id_to_index: dict[str, int] | None = None


class FakeRunner:
    """
    Minimal object that has the same attributes / methods used by
    _sync_for_cpu_proposer and _finalize_cpu_proposer_async.
    """

    def __init__(self, num_reqs: int, max_model_len: int = 512,
                 num_spec_tokens: int = 3, device: str = "cpu"):
        self.input_batch = FakeInputBatch(num_reqs, max_model_len)
        self.requests: dict[str, FakeCachedRequestState] = {}
        for rid in self.input_batch.req_ids:
            self.requests[rid] = FakeCachedRequestState(req_id=rid)
        self.num_spec_tokens = num_spec_tokens
        self.device = torch.device(device)
        self._draft_token_ids = None

        # Pre-allocated CPU buffers (mirrors gpu_model_runner init)
        self.valid_sampled_token_count_cpu = torch.zeros(
            num_reqs, dtype=torch.int64
        )
        self.valid_sampled_token_count_event = MagicMock()
        self.valid_sampled_token_count_event.record = MagicMock()

        # Pinned CPU buffer for _to_list
        self.sampled_token_ids_pinned_cpu = torch.zeros(
            (num_reqs, num_spec_tokens + 1), dtype=torch.int32
        )
        self.transfer_event = MagicMock()
        self.transfer_event.record = MagicMock()
        self.transfer_event.synchronize = MagicMock()

    # Import the real methods under test
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    _sync_for_cpu_proposer = GPUModelRunner._sync_for_cpu_proposer
    _finalize_cpu_proposer_async = GPUModelRunner._finalize_cpu_proposer_async
    _to_list = GPUModelRunner._to_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_runner_with_tokens(
    num_reqs: int,
    prompt_len: int,
    output_tokens: list[list[int]],
    num_spec_tokens: int = 3,
) -> FakeRunner:
    """Create a FakeRunner pre-populated as if _bookkeeping_sync already ran
    in async mode (wrote -1 placeholders)."""
    runner = FakeRunner(num_reqs, num_spec_tokens=num_spec_tokens)
    for i in range(num_reqs):
        # Fill prompt tokens
        runner.input_batch.token_ids_cpu[i, :prompt_len] = list(
            range(100, 100 + prompt_len)
        )
        runner.input_batch.is_token_ids[i, :prompt_len] = True
        # Simulate prior real output tokens
        pos = prompt_len
        for tok in output_tokens[i]:
            runner.input_batch.token_ids_cpu[i, pos] = tok
            runner.input_batch.is_token_ids[i, pos] = True
            pos += 1
            runner.requests[f"req_{i}"].output_token_ids.append(tok)
        # Simulate _bookkeeping_sync async: one -1 placeholder at pos
        runner.input_batch.token_ids_cpu[i, pos] = -1
        runner.input_batch.is_token_ids[i, pos] = True
        runner.input_batch.num_tokens_no_spec[i] = pos + 1
        runner.requests[f"req_{i}"].output_token_ids.append(-1)
    return runner


# ---------------------------------------------------------------------------
# Tests for _sync_for_cpu_proposer
# ---------------------------------------------------------------------------

class TestSyncForCpuProposer:

    def test_single_token_overwrites_placeholder(self):
        """First decode step: shape [N, 1], one real token per request."""
        runner = _setup_runner_with_tokens(
            num_reqs=2, prompt_len=10, output_tokens=[[], []]
        )
        # sampled_token_ids shape [2, 1]
        sampled = torch.tensor([[42], [99]], dtype=torch.int32)

        result = runner._sync_for_cpu_proposer(sampled, invalid_req_indices=[])

        assert result == [[42], [99]]
        # token_ids_cpu should have real token (not -1) at position 10
        assert runner.input_batch.token_ids_cpu[0, 10] == 42
        assert runner.input_batch.token_ids_cpu[1, 10] == 99
        # num_tokens_no_spec should be 11 (prompt_len + 1)
        assert runner.input_batch.num_tokens_no_spec[0] == 11
        assert runner.input_batch.num_tokens_no_spec[1] == 11
        # output_token_ids should have the base token only (not -1)
        assert runner.requests["req_0"].output_token_ids == [42]
        assert runner.requests["req_1"].output_token_ids == [99]

    def test_single_token_with_invalid_req(self):
        """Requests in invalid_req_indices should be cleared."""
        runner = _setup_runner_with_tokens(
            num_reqs=3, prompt_len=5, output_tokens=[[], [], []]
        )
        sampled = torch.tensor([[10], [20], [30]], dtype=torch.int32)

        result = runner._sync_for_cpu_proposer(sampled, invalid_req_indices=[1])

        assert result[0] == [10]
        assert result[1] == []  # cleared
        assert result[2] == [30]
        # token_ids_cpu for invalid req should still have -1
        assert runner.input_batch.token_ids_cpu[1, 5] == -1

    def test_multi_token_spec_decode_accepted(self):
        """Second+ decode step: spec decode tokens accepted, shape [N, K+1]."""
        runner = _setup_runner_with_tokens(
            num_reqs=2, prompt_len=10,
            output_tokens=[[50], [60]],  # one prior output token each
            num_spec_tokens=3,
        )
        # Simulate sampled_token_ids from rejection sampler
        # vocab_size=32000, PLACEHOLDER_TOKEN_ID is a large value
        PLACEHOLDER = 2**31 - 1  # same as rejection_sampler.py
        sampled = torch.tensor([
            [70, 71, PLACEHOLDER, PLACEHOLDER],  # req0: 2 accepted (base + 1 draft)
            [80, 81, 82, PLACEHOLDER],            # req1: 3 accepted (base + 2 drafts)
        ], dtype=torch.int64)

        with patch(
            "vllm.v1.worker.gpu_model_runner.RejectionSampler.parse_output"
        ) as mock_parse:
            mock_parse.return_value = ([[70, 71], [80, 81, 82]], None)
            result = runner._sync_for_cpu_proposer(sampled, invalid_req_indices=[])

        assert result == [[70, 71], [80, 81, 82]]

        # token_ids_cpu: req0 should have [70, 71] starting at position 11
        # (prompt=10, prior_output=1, placeholder was at 11, overwrite from 11)
        assert runner.input_batch.token_ids_cpu[0, 11] == 70
        assert runner.input_batch.token_ids_cpu[0, 12] == 71
        assert runner.input_batch.num_tokens_no_spec[0] == 13  # 11 + 2

        # token_ids_cpu: req1 should have [80, 81, 82] starting at position 11
        assert runner.input_batch.token_ids_cpu[1, 11] == 80
        assert runner.input_batch.token_ids_cpu[1, 12] == 81
        assert runner.input_batch.token_ids_cpu[1, 13] == 82
        assert runner.input_batch.num_tokens_no_spec[1] == 14  # 11 + 3

        # output_token_ids should only have base token (not drafts)
        # Prior: [50, -1], after fix: [50, 70] (only base token replaced)
        assert runner.requests["req_0"].output_token_ids == [50, 70]
        assert runner.requests["req_1"].output_token_ids == [60, 80]

    def test_empty_sampled_ids_skipped(self):
        """Requests with empty sampled_ids (all invalid) are skipped."""
        runner = _setup_runner_with_tokens(
            num_reqs=2, prompt_len=5, output_tokens=[[], []]
        )
        sampled = torch.tensor([[10], [20]], dtype=torch.int32)

        # All requests are invalid
        result = runner._sync_for_cpu_proposer(sampled, invalid_req_indices=[0, 1])

        assert result[0] == []
        assert result[1] == []
        # Placeholders remain
        assert runner.input_batch.token_ids_cpu[0, 5] == -1
        assert runner.input_batch.token_ids_cpu[1, 5] == -1


# ---------------------------------------------------------------------------
# Tests for _finalize_cpu_proposer_async
# ---------------------------------------------------------------------------

class TestFinalizeCpuProposerAsync:

    def test_list_draft_tokens_converted_to_tensor(self):
        """List-based draft tokens should become a padded GPU tensor."""
        runner = FakeRunner(num_reqs=3, num_spec_tokens=3)
        runner._draft_token_ids = [
            [100, 101, 102],  # full 3 drafts
            [200],            # only 1 draft
            [],               # no drafts
        ]
        valid = [[10], [20], [30]]

        runner._finalize_cpu_proposer_async(valid)

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
        valid = [[10], [20]]

        runner._finalize_cpu_proposer_async(valid)

        assert runner._draft_token_ids is original  # same object

    def test_prev_sampled_token_ids_set_correctly(self):
        """prev_sampled_token_ids should be shape [N, 1] with last accepted token."""
        runner = FakeRunner(num_reqs=3, num_spec_tokens=2)
        runner._draft_token_ids = [[], [], []]
        valid = [[10, 11], [20], []]  # multi-token, single, empty

        runner._finalize_cpu_proposer_async(valid)

        prev = runner.input_batch.prev_sampled_token_ids
        assert prev.shape == (3, 1)
        assert prev[0, 0].item() == 11  # last of [10, 11]
        assert prev[1, 0].item() == 20  # last of [20]
        assert prev[2, 0].item() == 0   # empty → 0

    def test_valid_sampled_token_count_cpu(self):
        """valid_sampled_token_count_cpu should hold token counts per request."""
        runner = FakeRunner(num_reqs=3, num_spec_tokens=2)
        runner._draft_token_ids = [[], [], []]
        valid = [[10, 11, 12], [20], []]

        runner._finalize_cpu_proposer_async(valid)

        counts = runner.valid_sampled_token_count_cpu
        assert counts[0].item() == 3
        assert counts[1].item() == 1
        assert counts[2].item() == 0
        runner.valid_sampled_token_count_event.record.assert_called_once()

    def test_draft_tokens_longer_than_num_spec_tokens_truncated(self):
        """Draft tokens exceeding num_spec_tokens should be truncated."""
        runner = FakeRunner(num_reqs=1, num_spec_tokens=2)
        runner._draft_token_ids = [[100, 101, 102, 103]]  # 4 tokens, limit is 2
        valid = [[10]]

        runner._finalize_cpu_proposer_async(valid)

        assert runner._draft_token_ids.shape == (1, 2)
        assert runner._draft_token_ids[0].tolist() == [100, 101]


# ---------------------------------------------------------------------------
# Tests for the assertion fix (temp prev_sampled_token_ids before bookkeeping)
# ---------------------------------------------------------------------------

class TestBookkeepingAssertionFix:

    def test_temp_prev_sampled_token_ids_prevents_assertion(self):
        """When sampled_token_ids has spec decode columns (shape [N, K]),
        setting prev_sampled_token_ids[:, :1] should prevent the
        shape[-1]==1 assertion in _bookkeeping_sync."""
        # Simulate the sampled tensor with spec decode columns
        sampled = torch.tensor([
            [10, 11, 12],  # base + 2 spec decode tokens
            [20, 21, 22],
        ], dtype=torch.int32)

        # Extract first column (as our fix does)
        temp = sampled[:, :1]

        assert temp.shape == (2, 1)
        assert temp.shape[-1] == 1  # This is the assertion check
        assert temp[0, 0].item() == 10
        assert temp[1, 0].item() == 20


# ---------------------------------------------------------------------------
# Integration-like test: full flow
# ---------------------------------------------------------------------------

class TestFullFlow:

    def test_single_token_flow(self):
        """Test the complete flow for first decode step (no prior drafts)."""
        runner = _setup_runner_with_tokens(
            num_reqs=2, prompt_len=10, output_tokens=[[], []],
            num_spec_tokens=3,
        )
        sampled = torch.tensor([[42], [99]], dtype=torch.int32)

        # Step 1: _sync_for_cpu_proposer
        valid = runner._sync_for_cpu_proposer(sampled, invalid_req_indices=[])
        assert valid == [[42], [99]]

        # Step 2: simulate proposer producing draft tokens
        runner._draft_token_ids = [[200, 201, 202], [300, 301, 302]]

        # Step 3: _finalize_cpu_proposer_async
        runner._finalize_cpu_proposer_async(valid)

        # Verify final state
        assert isinstance(runner._draft_token_ids, torch.Tensor)
        assert runner._draft_token_ids.shape == (2, 3)
        assert runner.input_batch.prev_sampled_token_ids.shape == (2, 1)
        assert runner.input_batch.prev_sampled_token_ids[0, 0].item() == 42
        assert runner.input_batch.prev_sampled_token_ids[1, 0].item() == 99
        assert runner.valid_sampled_token_count_cpu[0].item() == 1
        assert runner.valid_sampled_token_count_cpu[1].item() == 1

    def test_multi_token_flow(self):
        """Test the complete flow for decode step with accepted drafts."""
        runner = _setup_runner_with_tokens(
            num_reqs=2, prompt_len=10,
            output_tokens=[[50], [60]],
            num_spec_tokens=3,
        )
        # Simulated rejection sampler output
        with patch(
            "vllm.v1.worker.gpu_model_runner.RejectionSampler.parse_output"
        ) as mock_parse:
            mock_parse.return_value = ([[70, 71], [80, 81, 82]], None)
            sampled = torch.tensor([
                [70, 71, 0, 0],
                [80, 81, 82, 0],
            ], dtype=torch.int64)
            valid = runner._sync_for_cpu_proposer(sampled, invalid_req_indices=[])

        assert valid == [[70, 71], [80, 81, 82]]

        # Simulate proposer
        runner._draft_token_ids = [[400, 401, 402], [500, 501]]

        # Finalize
        runner._finalize_cpu_proposer_async(valid)

        # prev_sampled_token_ids should be last accepted token
        assert runner.input_batch.prev_sampled_token_ids[0, 0].item() == 71
        assert runner.input_batch.prev_sampled_token_ids[1, 0].item() == 82

        # Counts
        assert runner.valid_sampled_token_count_cpu[0].item() == 2
        assert runner.valid_sampled_token_count_cpu[1].item() == 3

        # token_ids_cpu has all accepted tokens
        assert runner.input_batch.token_ids_cpu[0, 11] == 70
        assert runner.input_batch.token_ids_cpu[0, 12] == 71
        assert runner.input_batch.token_ids_cpu[1, 11] == 80
        assert runner.input_batch.token_ids_cpu[1, 12] == 81
        assert runner.input_batch.token_ids_cpu[1, 13] == 82

        # output_token_ids has only base token (not draft tokens)
        assert runner.requests["req_0"].output_token_ids == [50, 70]
        assert runner.requests["req_1"].output_token_ids == [60, 80]

        # Draft tokens are a tensor
        assert isinstance(runner._draft_token_ids, torch.Tensor)
        assert runner._draft_token_ids[0].tolist() == [400, 401, 402]
        assert runner._draft_token_ids[1].tolist() == [500, 501, 0]
