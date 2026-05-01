#!/bin/bash

USERS=("torrespjc" "jesus-romero" "fdcordova" "jonatan.gallo" "josman.ramirez" "bravos.js" "jo.jaime")

echo "Starting user deletion..."

for USER in "${USERS[@]}"; do

    if id "$USER" &>/dev/null; then
        echo "Deleting user: $USER"

        # -r elimina home directory
        userdel -r "$USER"

        echo "User $USER deleted."
    else
        echo "User $USER does not exist. Skipping..."
    fi

done

echo "Finished."
