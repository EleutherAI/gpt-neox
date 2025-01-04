from typing import Union, List

import torch
import socket
import pickle


def send_tensor(state_dict_key, data, sock, end: bool):
    storage = data.storage()
    (
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()
    sock.send(
        pickle.dumps(
            {
                "state_dict_key": state_dict_key,
                "dtype": data.dtype,
                "tensor_size": data.shape,
                "tensor_stride": data.stride(),
                "tensor_offset": data.storage_offset(),  # !Not sure about this one.
                "storage_cls": type(storage),
                "storage_device": storage_device,
                "storage_handle": storage_handle,
                "storage_size_bytes": storage_size_bytes,
                "storage_offset_bytes": storage_offset_bytes,
                "requires_grad": False,
                "ref_counter_handle": ref_counter_handle,
                "ref_counter_offset": ref_counter_offset,
                "event_handle": event_handle,
                "event_sync_required": event_sync_required,
                "end": end,
            }
        )
    )


def send_state_dict(state_dict, sock):
    for i, key in enumerate(state_dict.keys()):
        print(key)
        end = i == len(state_dict.keys()) - 1
        send_tensor(key, state_dict[key], sock, end)
        sock.recv(4096)


def start_server(model, ports: Union[int, List[int]] = 6000):
    global_rank = torch.distributed.get_rank()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if type(ports) == int:
        port = ports + global_rank
    else:
        port = ports[global_rank]
    s.bind(("localhost", port))
    s.listen(1)
    conn, addr = s.accept()
    state_dict = model.state_dict()
    send_state_dict(state_dict, conn)
    conn.close()
