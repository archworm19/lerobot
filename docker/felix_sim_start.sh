#!/usr/bin/env bash
# The felix-sim image's start (docker/Dockerfile.felix_sim): RunPod's public key into root's authorized_keys (RunPod
# passes it as $PUBLIC_KEY), sshd, then the container stays up for runs over ssh.
set -euo pipefail

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A
/usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password
echo "felix-sim $(cat /opt/felix/image_tag): sshd up"
exec sleep infinity
