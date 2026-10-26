#!/bin/bash
# E10: raw code (codeparrot-clean, 3 files) and PubMed abstract chunks (MedRAG/pubmed, 60 files), then tokenize.
D=/home/soso/advsci/data/kov; PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/prep_domains.py"
mkdir -p $D/raw_code $D/raw_pubmed $D/code $D/pubmed
for k in 1 2 3; do f=$(printf "file-%012d.json.gz" $k)
  [ -s $D/raw_code/$f ] || curl -sSL --retry 5 -o $D/raw_code/$f "https://huggingface.co/datasets/codeparrot/codeparrot-clean/resolve/main/$f"; done
for k in $(seq 1 60); do f=$(printf "pubmed23n%04d.jsonl" $k)
  [ -s $D/raw_pubmed/$f ] || curl -sSL --retry 5 -o $D/raw_pubmed/$f "https://huggingface.co/datasets/MedRAG/pubmed/resolve/main/chunk/$f"; done
du -sh $D/raw_code $D/raw_pubmed
$PY code $D/raw_code $D/code 4e8
$PY pubmed $D/raw_pubmed $D/pubmed 4e8
echo ALLDONE
