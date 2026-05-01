for HOST in 10.43.97.141 10.43.97.135 10.43.97.136 10.43.97.148 10.43.97.146 10.43.97.145
do
  echo "Installing $HOST..."
  ssh estudiante@$HOST "sudo dnf install -y java-1.8.0-openjdk-devel"
done