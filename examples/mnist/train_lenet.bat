copy ..\\..\\bin\\MainCaller.exe ..\\..\\bin\\train_net.exe
SET GLOG_logtostderr=1
"../../bin/train_net.exe" lenet_solver.prototxt
pause
