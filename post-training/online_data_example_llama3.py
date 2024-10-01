import socket
import threading
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import requests
import pickle
from collections import defaultdict
import time


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def http_bot(url, pload):
    for i in range(10):
        try:
            headers = {"User-Agent": "vLLM Client"}
            response = requests.post(url, headers=headers, json=pload, stream=True)
            data = response.json()
            return data
        except Exception as e:
            # give it a few seconds to recover
            time.sleep(5)
            print(e)
            continue
    raise Exception("Failed to connect to server")


def threaded_data_gatherer(
    prefix,
    max_completion_len,
    tokenizer,
    model_name,
    num_completions,
    i,
    dp_idx,
    data_to_send,
    rm_pipeline,
):
    pload = {
        "temperature": 1.0,
        "max_tokens": 0,
        "stop": "<|eot_id|>",
        "stream": False,
        "model": model_name,
        "prompt": "",
        "n": num_completions,
    }
    # Grab tokens...
    prefix_tokens = tokenizer.encode(prefix)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Please write a mildly negative movie review starting with "
                + prefix,
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = tokenizer.encode(prompt)
    pload["max_tokens"] = max_completion_len - len(prefix_tokens)
    pload["prompt"] = prompt + prefix
    completions = http_bot(f"http://localhost:{8000+dp_idx}/v1/completions", pload)
    completions = [completion["text"].strip() for completion in completions["choices"]]

    def reward_fn(samples, **kwargs):
        sentiments = list(map(get_positive_score, rm_pipeline(samples)))
        return sentiments

    rewards = reward_fn([prefix + " " + completion for completion in completions])
    if i == 0 and dp_idx == 0:
        print(completions)
    completions = [
        tokenizer.encode(completion + "<|eot_id|>") for completion in completions
    ]
    data_to_send.append(
        {"prefix": prompt_tokens, "completions": completions, "rewards": rewards}
    )


def data_generator(
    bs_per_dp,
    dataset,
    tokenizer,
    model_name,
    max_prefix_len,
    max_completion_len,
    num_completions,
    dp_idx,
    dp_size,
    tp_size,
    rm_pipeline,
):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(
        ("localhost", 10000 + dp_idx)
    )  # only one data loader per data parallel group
    split_counter = defaultdict(lambda: dp_idx)
    while True:
        server.listen(1)
        conn, addr = server.accept()
        split = conn.recv(4096).decode()
        if split == "valid":
            split = "unsupervised"
        data_to_send = list()
        threads = list()
        for i in range(bs_per_dp):
            prefix = " ".join(
                dataset[split][split_counter[split]]["text"].split()[:5]
            )  # grab a few words to prompt it...
            split_counter[split] = (split_counter[split] + dp_size) % len(
                dataset[split]
            )
            threads.append(
                threading.Thread(
                    target=threaded_data_gatherer,
                    args=(
                        prefix,
                        max_completion_len,
                        tokenizer,
                        model_name,
                        num_completions,
                        i,
                        dp_idx,
                        data_to_send,
                        rm_pipeline,
                    ),
                )
            )
            threads[-1].start()
        for thread in threads:
            thread.join()
        conn.send(pickle.dumps(data_to_send))
        conn.close()
        print(
            f"Sent data to {dp_idx} for {split} split at iter {split_counter[split]}..."
        )


if __name__ == "__main__":
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device="cpu",
    )
    dataset = datasets.load_dataset("imdb")
    threads = list()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    for i in range(2):
        threads.append(
            threading.Thread(
                target=data_generator,
                args=(
                    64,  # bs_per_dp
                    dataset,  # dataset
                    tokenizer,  # tokenizer
                    "meta-llama/Meta-Llama-3-8B-Instruct",  # model_name
                    128,  # max_prefix_len
                    256,  # max_completion_len
                    4,  # num_completions
                    i,  # dp_idx
                    2,  # dp_size
                    4,  # tp_size
                    sentiment_fn,  # rm_pipeline
                ),
            )
        )
        threads[-1].start()
    for thread in threads:
        thread.join()
