for HOST in 10.43.97.146 10.43.97.148 10.43.97.145 10.43.97.136 10.43.97.135 10.43.97.141 10.43.97.149
do
  scp delete_users.sh estudiante@$HOST:/tmp/
  ssh estudiante@$HOST "sudo bash /tmp/delete_users.sh && sudo rm /tmp/delete_users.sh"
done
