#!/bin/bash
cd ${WORKING_PATH}
#export -n HADOOP_HDFS_HOME=/usr/local/hadoop-2.7.2
#export -n HADOOP_CLASSPATH=/usr/local/hadoop-2.7.2/lib/classpath_hdfs.jar
#export -n HADOOP_OPTS=-Djava.library.path=/usr/local/hadoop-2.7.2/lib/native
#export -n HADOOP_COMMOM_LIB_NATIVE_DIR=/usr/local/hadoop-2.7.2/lib/native
env
echo "hadoop"
export -n JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"
hadoop fs -get hdfs://hobot-bigdata/user/zhengkai.jiang/common/dataset/coco/* ../
export JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"
CMD="python experiments/retinanet/retinanet_train.py"
echo Running ${CMD}
${CMD}
CMD="python experiments/retinanet/retinanet_test.py"
echo Running ${CMD}
${CMD}
