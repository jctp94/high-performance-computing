for HOST in 10.43.97.141 10.43.97.135 10.43.97.136 10.43.97.148 10.43.97.146
do
  echo "Installing $HOST..."
  ssh estudiante@$HOST "sudo dnf -y install openmpi openmpi-devel gtk2-devel"
done