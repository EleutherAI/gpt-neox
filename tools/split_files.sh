DATA_FILES=$(find /fsx/multi-lingual-6b -path "*/gpt-neox/data/*/part-00000")
for file in $DATA_FILES
do
    echo ${file}
    split -C 209715200 -a 4 -d --additional-suffix .jsonl ${file} ${file}-
done