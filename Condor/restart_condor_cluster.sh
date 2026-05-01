#!/bin/bash

USER="estudiante"

NODES=(
  "10.43.97.146"  # cadhead02
  "10.43.97.141"  # cad02-w000
  "10.43.97.136"  # cad02-w002
  "10.43.97.148"  # cad02-w003
  "10.43.97.135"  # cad02-w001
  "10.43.97.145"  # cadcliente02
)

for NODE in "${NODES[@]}"; do
  echo "======================================"
  echo "Connecting to $NODE"
  echo "======================================"

  ssh -o StrictHostKeyChecking=no ${USER}@${NODE} << 'EOF'
    echo "On host: $(hostname)"
    sudo systemctl restart condor
    echo "Condor restarted on $(hostname)"
EOF

done

echo "✅ All nodes processed"
