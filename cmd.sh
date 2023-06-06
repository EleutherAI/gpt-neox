export VV=${1:-"00"}
export FILE=docker.build.v${VV}

set -x

echo $FILE

sleep 5

date  > $FILE;
docker build . -t sliuxl_stable_ckpt_v${VV} >> $FILE;
date >> $FILE;
