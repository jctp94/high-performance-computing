for HOST in 10.43.97.141 10.43.97.135 10.43.97.136 10.43.97.148 10.43.97.146 10.43.97.145
do
  echo "Exporting Spark environment on $HOST..."

  ssh estudiante@$HOST "grep -q SPARK_HOME ~/.bashrc || echo '
export SPARK_HOME=/nfs/condor/apps/Spark
export SPARK_CONF_DIR=\$SPARK_HOME/conf
export PATH=\$SPARK_HOME/bin:\$PATH
export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3
' >> ~/.bashrc"

  echo "Spark environment exported on $HOST"
done