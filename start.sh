#!/bin/bash
# 헤이 은석! 서버 시작 스크립트

export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
cd /app
python server.py
