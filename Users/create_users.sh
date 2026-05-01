#!/bin/bash

PASSWORD="ucad02"

USERS=("torrespjc" "jesus-romero" "fdcordova" "jonatan.gallo" "josman.ramirez" "bravos.js" "jo.jaime")
IDS=(1003 1004 1005 1006 1007 1008 1009)
CLIENT_HOST="10.43.97.145"
CURRENT_IP=$(ip route get 1 | awk '{print $7; exit}')

echo "Starting user creation with specific UIDs..."

for i in "${!USERS[@]}"; do

    USER="${USERS[$i]}"
    UID_VALUE="${IDS[$i]}"

    if id "$USER" &>/dev/null; then
        echo "User $USER already exists. Skipping..."
        continue
    fi

    echo "Creating $USER with UID $UID_VALUE"

    useradd -u "$UID_VALUE" -m -s /bin/bash "$USER"

    if [ "$CURRENT_IP" == "$CLIENT_HOST" ]; then
        echo "$USER:$PASSWORD" | chpasswd
        echo "Password set for $USER"
    else
        passwd -l "$USER"
        echo "Password locked for $USER"
    fi


    echo "Created $USER with UID $(id -u $USER)"

done

echo "Finished."
