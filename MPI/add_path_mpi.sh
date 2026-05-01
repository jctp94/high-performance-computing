for HOST in 10.43.97.141 10.43.97.135 10.43.97.136 10.43.97.148 10.43.97.146
do
  echo "Adding MPI paths to .bashrc on $HOST..."
  ssh estudiante@$HOST "echo 'export PATH=/nfs/condor/apps/openmpi-4.1.6/bin:\$PATH' >> ~/.bashrc"
  ssh estudiante@$HOST "echo 'export LD_LIBRARY_PATH=/nfs/condor/apps/openmpi-4.1.6/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc"
done