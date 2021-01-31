#!/usr/bin/env bash

pdsh -f 1024 -R ssh -w ^/job/hosts 'pkill -f "python -u train*"; pkill -f "python3 -u train*"; pkill -f "python -u deep*"; pkill -f "python3 -u deep*";'
