# Copyright (c) 2026, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from megatron.model.transformer import _validate_qk_layernorm_sharing


@pytest.mark.parametrize(
    ("q_norm_size", "k_norm_size", "qk_layernorm_separate"),
    [(64, 64, False), (64, 64, True), (128, 64, True)],
)
def test_qk_layernorm_sharing_accepts_compatible_configuration(
    q_norm_size, k_norm_size, qk_layernorm_separate
):
    _validate_qk_layernorm_sharing(q_norm_size, k_norm_size, qk_layernorm_separate)


def test_qk_layernorm_sharing_rejects_different_sizes():
    with pytest.raises(ValueError) as exc_info:
        _validate_qk_layernorm_sharing(128, 64, False)

    message = str(exc_info.value)
    assert "normalized sizes differ (query=128, key=64)" in message
    assert "qk_layernorm_separate=True" in message
    assert "qk_layernorm_type='per_head'" in message
    assert "num_kv_heads" in message
