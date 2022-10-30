import os
import re
import time

# get slurm job ID
jobId = os.popen('sbatch --comment neox jupyter.sbatch').read()
print(jobId)
host = os.popen('curl http://169.254.169.254/latest/meta-data/public-ipv4').read()
jobId = [int(s) for s in jobId.split() if s.isdigit()][0]

# wait for the output file to appear
while not os.path.exists(f'jupyter_{jobId}.out'):
    time.sleep(1)

# wait until notebook server is started
content=''
while not "http://127.0.0.1" in content:
   with open(f'jupyter_{jobId}.out') as fh:
      content = fh.read()
   time.sleep(1)

with open(f'jupyter_{jobId}.out') as fh:
   fstring = fh.readlines()

host = fstring[0]
print(host)
# extract compute node IP
pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
jIP = ''
for line in fstring:
   if pattern.search(line) is not None:
      jIP = pattern.search(line)[0]
      if jIP.startswith('172.31.'):
         break

# extract jupyter notebook token
token = ''
for line in fstring:
   token = line
   if token.startswith('     or http://127.0.0.1:8888/'):
      break
token = token.split('=')[-1]

username = os.getlogin()

print ("connect with:")
print (f"ssh -L 8888:{jIP}:8888 {username}@{host}")
print()
print("then browse:")
print (f"http://127.0.0.1:8888/?token={token}")
print()
print("when done, close the job:")
print(f"run: scancel {jobId}")

os.system(f"ssh -L 8888:{jIP}:8888 {username}@{host}")
