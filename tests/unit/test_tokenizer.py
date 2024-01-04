import pytest
from megatron.tokenizer import train_tokenizer


@pytest.mark.cpu
def test_train_tokenizer():
    input_args = [
        "--json_input_dir",
        "./tests/data/enwik8_first100.txt",
        "--tokenizer_output_path",
        "",
    ]
    args = train_tokenizer.parse_args(input_args)
    train_tokenizer.main(args)
