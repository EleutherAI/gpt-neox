str1 = """
GPU 0: A100-PCIE-40GB (UUID: GPU-f41a1a93-42ec-1dc1-e936-03c617b58fd4)
         Link 0: Data Tx: 8299479 KiB
         Link 0: Data Rx: 7492431 KiB
         Link 1: Data Tx: 8299472 KiB
         Link 1: Data Rx: 7493112 KiB
         Link 2: Data Tx: 8301835 KiB
         Link 2: Data Rx: 7493042 KiB
         Link 3: Data Tx: 8301982 KiB
         Link 3: Data Rx: 7492261 KiB
         Link 4: Data Tx: 8302504 KiB
         Link 4: Data Rx: 7490381 KiB
         Link 5: Data Tx: 8303175 KiB
         Link 5: Data Rx: 7490403 KiB
         Link 6: Data Tx: 8299483 KiB
         Link 6: Data Rx: 7492488 KiB
         Link 7: Data Tx: 8298793 KiB
         Link 7: Data Rx: 7493947 KiB
         Link 8: Data Tx: 8301854 KiB
         Link 8: Data Rx: 7492317 KiB
         Link 9: Data Tx: 8301947 KiB
         Link 9: Data Rx: 7492933 KiB
         Link 10: Data Tx: 8303158 KiB
         Link 10: Data Rx: 7490425 KiB
         Link 11: Data Tx: 8301807 KiB
         Link 11: Data Rx: 7489712 KiB
GPU 1: A100-PCIE-40GB (UUID: GPU-83994dd1-c701-4866-c130-02d446989f65)
         Link 0: Data Tx: 46450143 KiB
         Link 0: Data Rx: 34406764 KiB
         Link 1: Data Tx: 46450189 KiB
         Link 1: Data Rx: 34405962 KiB
         Link 2: Data Tx: 46469945 KiB
         Link 2: Data Rx: 34386201 KiB
         Link 3: Data Tx: 46469381 KiB
         Link 3: Data Rx: 34391181 KiB
         Link 4: Data Tx: 46462312 KiB
         Link 4: Data Rx: 34392277 KiB
         Link 5: Data Tx: 46462186 KiB
         Link 5: Data Rx: 34397858 KiB
         Link 6: Data Tx: 46450226 KiB
         Link 6: Data Rx: 34406089 KiB
         Link 7: Data Tx: 46457005 KiB
         Link 7: Data Rx: 34408501 KiB
         Link 8: Data Tx: 46470642 KiB
         Link 8: Data Rx: 34386874 KiB
         Link 9: Data Tx: 46469388 KiB
         Link 9: Data Rx: 34392357 KiB
         Link 10: Data Tx: 46462230 KiB
         Link 10: Data Rx: 34392281 KiB
         Link 11: Data Tx: 46455935 KiB
         Link 11: Data Rx: 34396604 KiB
GPU 2: A100-PCIE-40GB (UUID: GPU-cfe64a16-bb4b-1b93-cf89-53ab87027f81)
         Link 0: Data Tx: 34406764 KiB
         Link 0: Data Rx: 46450143 KiB
         Link 1: Data Tx: 34405962 KiB
         Link 1: Data Rx: 46450189 KiB
         Link 2: Data Tx: 34386201 KiB
         Link 2: Data Rx: 46469945 KiB
         Link 3: Data Tx: 34391181 KiB
         Link 3: Data Rx: 46469381 KiB
         Link 4: Data Tx: 34392277 KiB
         Link 4: Data Rx: 46462312 KiB
         Link 5: Data Tx: 34397858 KiB
         Link 5: Data Rx: 46462186 KiB
         Link 6: Data Tx: 34406089 KiB
         Link 6: Data Rx: 46450226 KiB
         Link 7: Data Tx: 34408501 KiB
         Link 7: Data Rx: 46457005 KiB
         Link 8: Data Tx: 34386874 KiB
         Link 8: Data Rx: 46470642 KiB
         Link 9: Data Tx: 34392357 KiB
         Link 9: Data Rx: 46469388 KiB
         Link 10: Data Tx: 34392281 KiB
         Link 10: Data Rx: 46462230 KiB
         Link 11: Data Tx: 34396604 KiB
         Link 11: Data Rx: 46455935 KiB
GPU 3: A100-PCIE-40GB (UUID: GPU-a3c7f094-cdd0-63b0-4cfa-5051b963ddb9)
         Link 0: Data Tx: 7492431 KiB
         Link 0: Data Rx: 8299479 KiB
         Link 1: Data Tx: 7493112 KiB
         Link 1: Data Rx: 8299472 KiB
         Link 2: Data Tx: 7493042 KiB
         Link 2: Data Rx: 8301835 KiB
         Link 3: Data Tx: 7492261 KiB
         Link 3: Data Rx: 8301982 KiB
         Link 4: Data Tx: 7490381 KiB
         Link 4: Data Rx: 8302504 KiB
         Link 5: Data Tx: 7490403 KiB
         Link 5: Data Rx: 8303175 KiB
         Link 6: Data Tx: 7492488 KiB
         Link 6: Data Rx: 8299483 KiB
         Link 7: Data Tx: 7493947 KiB
         Link 7: Data Rx: 8298793 KiB
         Link 8: Data Tx: 7492317 KiB
         Link 8: Data Rx: 8301854 KiB
         Link 9: Data Tx: 7492933 KiB
         Link 9: Data Rx: 8301947 KiB
         Link 10: Data Tx: 7490425 KiB
         Link 10: Data Rx: 8303158 KiB
         Link 11: Data Tx: 7489712 KiB
         Link 11: Data Rx: 8301807 KiB
GPU 4: A100-PCIE-40GB (UUID: GPU-4bb80da9-f1b7-63e4-7806-f3123645c4d9)
         Link 0: Data Tx: 449258837 KiB
         Link 0: Data Rx: 438164379 KiB
         Link 1: Data Tx: 449215786 KiB
         Link 1: Data Rx: 438171691 KiB
         Link 2: Data Tx: 449196222 KiB
         Link 2: Data Rx: 438854670 KiB
         Link 3: Data Tx: 449223347 KiB
         Link 3: Data Rx: 438889674 KiB
         Link 4: Data Tx: 448475201 KiB
         Link 4: Data Rx: 438882171 KiB
         Link 5: Data Tx: 448480794 KiB
         Link 5: Data Rx: 438854607 KiB
         Link 6: Data Tx: 449255737 KiB
         Link 6: Data Rx: 438181569 KiB
         Link 7: Data Tx: 449249604 KiB
         Link 7: Data Rx: 438179191 KiB
         Link 8: Data Tx: 449211006 KiB
         Link 8: Data Rx: 438866197 KiB
         Link 9: Data Tx: 449223200 KiB
         Link 9: Data Rx: 438876487 KiB
         Link 10: Data Tx: 448477424 KiB
         Link 10: Data Rx: 438878723 KiB
         Link 11: Data Tx: 448477420 KiB
         Link 11: Data Rx: 438872993 KiB
GPU 5: A100-PCIE-40GB (UUID: GPU-a5e3ac3a-6a5a-8a72-32ce-4c3ad2c74506)
         Link 0: Data Tx: 438164379 KiB
         Link 0: Data Rx: 449258837 KiB
         Link 1: Data Tx: 438171691 KiB
         Link 1: Data Rx: 449215786 KiB
         Link 2: Data Tx: 438854670 KiB
         Link 2: Data Rx: 449196222 KiB
         Link 3: Data Tx: 438889674 KiB
         Link 3: Data Rx: 449223347 KiB
         Link 4: Data Tx: 438882171 KiB
         Link 4: Data Rx: 448475201 KiB
         Link 5: Data Tx: 438854607 KiB
         Link 5: Data Rx: 448480794 KiB
         Link 6: Data Tx: 438181569 KiB
         Link 6: Data Rx: 449255737 KiB
         Link 7: Data Tx: 438179191 KiB
         Link 7: Data Rx: 449249604 KiB
         Link 8: Data Tx: 438866197 KiB
         Link 8: Data Rx: 449211006 KiB
         Link 9: Data Tx: 438876487 KiB
         Link 9: Data Rx: 449223200 KiB
         Link 10: Data Tx: 438878723 KiB
         Link 10: Data Rx: 448477424 KiB
         Link 11: Data Tx: 438872993 KiB
         Link 11: Data Rx: 448477420 KiB
"""

str2 = """
GPU 0: A100-PCIE-40GB (UUID: GPU-f41a1a93-42ec-1dc1-e936-03c617b58fd4)
         Link 0: Data Tx: 8992818 KiB
         Link 0: Data Rx: 8179345 KiB
         Link 1: Data Tx: 8992811 KiB
         Link 1: Data Rx: 8180090 KiB
         Link 2: Data Tx: 8995364 KiB
         Link 2: Data Rx: 8180019 KiB
         Link 3: Data Tx: 8995511 KiB
         Link 3: Data Rx: 8179174 KiB
         Link 4: Data Tx: 8996100 KiB
         Link 4: Data Rx: 8177104 KiB
         Link 5: Data Tx: 8996837 KiB
         Link 5: Data Rx: 8177129 KiB
         Link 6: Data Tx: 8992823 KiB
         Link 6: Data Rx: 8179402 KiB
         Link 7: Data Tx: 8992068 KiB
         Link 7: Data Rx: 8180989 KiB
         Link 8: Data Tx: 8995382 KiB
         Link 8: Data Rx: 8179231 KiB
         Link 9: Data Tx: 8995475 KiB
         Link 9: Data Rx: 8179909 KiB
         Link 10: Data Tx: 8996819 KiB
         Link 10: Data Rx: 8177152 KiB
         Link 11: Data Tx: 8995339 KiB
         Link 11: Data Rx: 8176372 KiB
GPU 1: A100-PCIE-40GB (UUID: GPU-83994dd1-c701-4866-c130-02d446989f65)
         Link 0: Data Tx: 47150279 KiB
         Link 0: Data Rx: 35093497 KiB
         Link 1: Data Tx: 47150325 KiB
         Link 1: Data Rx: 35092631 KiB
         Link 2: Data Tx: 47170140 KiB
         Link 2: Data Rx: 35072858 KiB
         Link 3: Data Tx: 47169511 KiB
         Link 3: Data Rx: 35077837 KiB
         Link 4: Data Tx: 47162311 KiB
         Link 4: Data Rx: 35079503 KiB
         Link 5: Data Tx: 47162186 KiB
         Link 5: Data Rx: 35085148 KiB
         Link 6: Data Tx: 47150362 KiB
         Link 6: Data Rx: 35092759 KiB
         Link 7: Data Tx: 47157206 KiB
         Link 7: Data Rx: 35095300 KiB
         Link 8: Data Tx: 47170901 KiB
         Link 8: Data Rx: 35073594 KiB
         Link 9: Data Tx: 47169518 KiB
         Link 9: Data Rx: 35079014 KiB
         Link 10: Data Tx: 47162100 KiB
         Link 10: Data Rx: 35079508 KiB
         Link 11: Data Tx: 47155805 KiB
         Link 11: Data Rx: 35083765 KiB
GPU 2: A100-PCIE-40GB (UUID: GPU-cfe64a16-bb4b-1b93-cf89-53ab87027f81)
         Link 0: Data Tx: 35093497 KiB
         Link 0: Data Rx: 47150279 KiB
         Link 1: Data Tx: 35092631 KiB
         Link 1: Data Rx: 47150325 KiB
         Link 2: Data Tx: 35072858 KiB
         Link 2: Data Rx: 47170140 KiB
         Link 3: Data Tx: 35077837 KiB
         Link 3: Data Rx: 47169511 KiB
         Link 4: Data Tx: 35079503 KiB
         Link 4: Data Rx: 47162311 KiB
         Link 5: Data Tx: 35085148 KiB
         Link 5: Data Rx: 47162186 KiB
         Link 6: Data Tx: 35092759 KiB
         Link 6: Data Rx: 47150362 KiB
         Link 7: Data Tx: 35095300 KiB
         Link 7: Data Rx: 47157206 KiB
         Link 8: Data Tx: 35073594 KiB
         Link 8: Data Rx: 47170901 KiB
         Link 9: Data Tx: 35079014 KiB
         Link 9: Data Rx: 47169518 KiB
         Link 10: Data Tx: 35079508 KiB
         Link 10: Data Rx: 47162100 KiB
         Link 11: Data Tx: 35083765 KiB
         Link 11: Data Rx: 47155805 KiB
GPU 3: A100-PCIE-40GB (UUID: GPU-a3c7f094-cdd0-63b0-4cfa-5051b963ddb9)
         Link 0: Data Tx: 8179345 KiB
         Link 0: Data Rx: 8992818 KiB
         Link 1: Data Tx: 8180090 KiB
         Link 1: Data Rx: 8992811 KiB
         Link 2: Data Tx: 8180019 KiB
         Link 2: Data Rx: 8995364 KiB
         Link 3: Data Tx: 8179174 KiB
         Link 3: Data Rx: 8995511 KiB
         Link 4: Data Tx: 8177104 KiB
         Link 4: Data Rx: 8996100 KiB
         Link 5: Data Tx: 8177129 KiB
         Link 5: Data Rx: 8996837 KiB
         Link 6: Data Tx: 8179402 KiB
         Link 6: Data Rx: 8992823 KiB
         Link 7: Data Tx: 8180989 KiB
         Link 7: Data Rx: 8992068 KiB
         Link 8: Data Tx: 8179231 KiB
         Link 8: Data Rx: 8995382 KiB
         Link 9: Data Tx: 8179909 KiB
         Link 9: Data Rx: 8995475 KiB
         Link 10: Data Tx: 8177152 KiB
         Link 10: Data Rx: 8996819 KiB
         Link 11: Data Tx: 8176372 KiB
         Link 11: Data Rx: 8995339 KiB
GPU 4: A100-PCIE-40GB (UUID: GPU-4bb80da9-f1b7-63e4-7806-f3123645c4d9)
         Link 0: Data Tx: 449959027 KiB
         Link 0: Data Rx: 438851546 KiB
         Link 1: Data Tx: 449915911 KiB
         Link 1: Data Rx: 438858858 KiB
         Link 2: Data Tx: 449896102 KiB
         Link 2: Data Rx: 439541333 KiB
         Link 3: Data Tx: 449923227 KiB
         Link 3: Data Rx: 439576337 KiB
         Link 4: Data Tx: 449175396 KiB
         Link 4: Data Rx: 439568957 KiB
         Link 5: Data Tx: 449181118 KiB
         Link 5: Data Rx: 439541459 KiB
         Link 6: Data Tx: 449955862 KiB
         Link 6: Data Rx: 438868799 KiB
         Link 7: Data Tx: 449949794 KiB
         Link 7: Data Rx: 438866421 KiB
         Link 8: Data Tx: 449910886 KiB
         Link 8: Data Rx: 439552861 KiB
         Link 9: Data Tx: 449923080 KiB
         Link 9: Data Rx: 439563150 KiB
         Link 10: Data Tx: 449177620 KiB
         Link 10: Data Rx: 439565380 KiB
         Link 11: Data Tx: 449177616 KiB
         Link 11: Data Rx: 439559716 KiB
GPU 5: A100-PCIE-40GB (UUID: GPU-a5e3ac3a-6a5a-8a72-32ce-4c3ad2c74506)
         Link 0: Data Tx: 438851546 KiB
         Link 0: Data Rx: 449959027 KiB
         Link 1: Data Tx: 438858858 KiB
         Link 1: Data Rx: 449915911 KiB
         Link 2: Data Tx: 439541333 KiB
         Link 2: Data Rx: 449896102 KiB
         Link 3: Data Tx: 439576337 KiB
         Link 3: Data Rx: 449923227 KiB
         Link 4: Data Tx: 439568957 KiB
         Link 4: Data Rx: 449175396 KiB
         Link 5: Data Tx: 439541459 KiB
         Link 5: Data Rx: 449181118 KiB
         Link 6: Data Tx: 438868799 KiB
         Link 6: Data Rx: 449955862 KiB
         Link 7: Data Tx: 438866421 KiB
         Link 7: Data Rx: 449949794 KiB
         Link 8: Data Tx: 439552861 KiB
         Link 8: Data Rx: 449910886 KiB
         Link 9: Data Tx: 439563150 KiB
         Link 9: Data Rx: 449923080 KiB
         Link 10: Data Tx: 439565380 KiB
         Link 10: Data Rx: 449177620 KiB
         Link 11: Data Tx: 439559716 KiB
         Link 11: Data Rx: 449177616 KiB
"""

import re
import time


def split_by_gpu(string):
    return [i for i in re.split('GPU [\d]+', string) if i != '']


def get_link_id(string):
    return int(re.search(r'^Link ([\d]+)', string).group(1))


def is_rx(string):
    return 'Rx' in string


def is_tx(string):
    return 'Tx' in string


def split_by_link(gpu_string):
    return [i.strip() for i in re.split('\n', gpu_string) if re.search('Link [\d]+', i)]


def kib_to_b(kib):
    return kib * 1024


def get_tput(string):
    return kib_to_b(int(re.search(r': ([\d]+) KiB$', string).group(1)))


def human_readable(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class NVLinkMonitor:

    def __init__(self):
        self.cumulative_data = {}
        self.tput_data = {}
        self.timer = time.time()

    @staticmethod
    def parse(string):
        string = string.strip()
        _gpu_data = {}
        for counter, gpu_string in enumerate(split_by_gpu(string)):
            every_link = split_by_link(string)
            rx = {}
            tx = {}
            for link in every_link:
                if is_rx(link):
                    rx[get_link_id(link)] = get_tput(link)
                elif is_tx(link):
                    tx[get_link_id(link)] = get_tput(link)
            _gpu_data[f'GPU_{counter}'] = {'tx': tx, 'rx': rx,
                                           'rx_mean': sum(rx.values()) / len(rx.values()),
                                           'tx_mean': sum(tx.values()) / len(tx.values())}
        return _gpu_data

    def update(self, string):
        new_data = self.parse(string)
        if self.cumulative_data != {}:
            self.tput_data = self._diff(self.cumulative_data, new_data)
            self.display(self.tput_data)
        else:
            self.tput_data = self.cumulative_data
        self.cumulative_data = new_data
        time_since_last_update = time.time() - self.timer
        self.timer = time.time()

    def display(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self.display(v)
            else:
                print(f"{k}: {human_readable(v)}/s")

    def per_s(self, d, secs):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = self.per_s(v, secs)
            else:
                ret[k] = v / secs
        return ret

    def _diff(self, gpu_data1, gpu_data2):
        diff_dict = {}
        for gpu_no, data in gpu_data1.items():
            if gpu_no in gpu_data2:
                diff_dict[gpu_no] = {}
                for tput_type, tput_data in data.items():
                    diff_dict[gpu_no][tput_type] = {}
                    if 'mean' not in tput_type:
                        for link_id, tput in tput_data.items():
                            diff_dict[gpu_no][tput_type][link_id] = gpu_data2[gpu_no][tput_type][link_id] - tput
                    else:
                        diff_dict[gpu_no][tput_type] = gpu_data2[gpu_no][tput_type] - tput_data
        return diff_dict


# nv = NVLinkMonitor()
# data_1 = nv.parse(str1)
# data_2 = nv.parse(str2)
# nv._diff(data_1, data_2)
# start = time.time()
# time.sleep(10)
# t = time.time() - start
# nv.display(nv.per_s(data_1, t))

from pprint import pprint
from subprocess import check_output
import re

CONNECTION_TYPES = ["X", "SYS", "NODE", "PHB", "PXB", "PIX", "NV[\d]+"]

def get_topology_str():
    return check_output(["nvidia-smi", "topo", "-m"]).decode()

def contains_nvlinks(topology):
    any([item])
def is_nvlink(connection_type):
    return re.search(CONNECTION_TYPES[-1], connection_type)


def get_nvlink_pairs(topology):
    """
    takes a topology matrix and outputs a list of pairs bridged by nvlink
    """
    out = set()
    for device_idx1, item1 in enumerate(topology):
        for device_idx2, item2 in enumerate(item1):
            if is_nvlink(item2):
                if (device_idx2, device_idx1) not in out:
                    out.add((device_idx1, device_idx2))
    return out


def get_cuda_visible_device_mapping(nvlink_pairs):
    return_string = ''
    for item in sorted(list(nvlink_pairs)):
        return_string += f"{item[0]},{item[1]},"
    return_string = return_string.strip(',')
    return return_string


def topology_from_string(string):
    output_per_gpu = string.strip().split('Legend:')[0].strip().split('\n')
    headers = output_per_gpu.pop(0)
    headers = headers.strip().split()
    headers = [i for i in headers if re.search('GPU[\d]+', i)]
    num_gpus = len(headers)

    topology = []
    for output in output_per_gpu:
        output = output.strip().split()
        gpu_id = output.pop(0)
        output = output[:num_gpus]
        if 'GPU' in gpu_id:
            links = []
            for idx, i in enumerate(output):
                if idx >= num_gpus:
                    break
                links.append(i.strip())
            topology.append(links)

    # checks for consistency
    assert all([len(i) == len(topology) for i in topology])
    pprint(topology)
    return topology

if __name__ == "__main__":
    string = """
    	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	mlx5_0	CPU Affinity	NUMA Affinity
GPU0	 X 	NODE	NODE	NV12	SYS	SYS	NODE	0-27,56-83	0
GPU1	NODE	 X 	NV12	SYS	SYS	SYS	NODE	0-27,56-83	0
GPU2	NODE	NV12	 X 	SYS	SYS	SYS	NODE	0-27,56-83	0
GPU3	NV12	SYS	SYS	 X 	NODE	NODE	SYS	28-55,84-111	1
GPU4	SYS	SYS	SYS	NODE	 X 	NV12	SYS	28-55,84-111	1
GPU5	SYS	SYS	SYS	NODE	NV12	 X 	SYS	28-55,84-111	1
mlx5_0	NODE	NODE	NODE	SYS	SYS	SYS	 X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

    """
    try:
        string = get_topology_str()
    except:
        pass
    topology = topology_from_string(string)
    pairs = get_nvlink_pairs(topology)
    print(get_cuda_visible_device_mapping(pairs))
