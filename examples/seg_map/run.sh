#!/usr/bin/env sh

rm -rf log

TOOLS=../../build/tools

GLOG_logtostderr=1 

"../../bin/fcn.exe" cat.jpg
