#!/bin/bash
exec /mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o IdentitiesOnly=yes -i 'D:\Work\ssh\tongdat_rsa.prv' -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$@"
