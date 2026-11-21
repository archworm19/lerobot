#!/usr/bin/env bash
# The felix-sim image's start (docker/Dockerfile.felix_sim): RunPod's public key into root's authorized_keys (RunPod
# passes it as $PUBLIC_KEY), the pod's lifetime (`POD_TTL`), sshd, then the container stays up for runs over ssh.
#
# POD_TTL (e.g. 30m, 4h, 1d; the template's env, or the deploy's): the pod stops itself that long after it starts
# (`runpodctl pod stop`, with the pod-scoped $RUNPOD_API_KEY RunPod gives every pod). Stopped, not deleted: the GPU's
# billing ends, the disk's does not until the pod is deleted. When: /opt/felix/stop_at; what happened: /var/log/pod_ttl.log.
set -euo pipefail

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

seconds() {  # 90, 90s, 30m, 4h, 1d -> seconds; nothing for anything else
    case "$1" in
        *[!0-9smhd]* | "" | [smhd]* | *[smhd]?*) return 1 ;;  # (one unit, last)
        *d) echo $(( ${1%d} * 86400 )) ;;
        *h) echo $(( ${1%h} * 3600 )) ;;
        *m) echo $(( ${1%m} * 60 )) ;;
        *s) echo "${1%s}" ;;
        *) echo "$1" ;;
    esac
}

if [[ -n "${POD_TTL:-}" ]]; then
    if ! ttl=$(seconds "${POD_TTL}"); then
        echo "POD_TTL=${POD_TTL}: not a duration (30m, 4h, 1d): no timer" | tee -a /var/log/pod_ttl.log
    elif [[ -z "${RUNPOD_POD_ID:-}" ]]; then
        echo "POD_TTL=${POD_TTL} but no RUNPOD_POD_ID (not a RunPod pod): no timer" | tee -a /var/log/pod_ttl.log
    else
        date -u -d "@$(( $(date +%s) + ttl ))" +%Y-%m-%dT%H:%M:%SZ > /opt/felix/stop_at
        echo "POD_TTL=${POD_TTL}: the pod stops itself at $(cat /opt/felix/stop_at)" | tee -a /var/log/pod_ttl.log
        (
            sleep "${ttl}"
            echo "$(date -u +%FT%TZ) POD_TTL reached: runpodctl pod stop ${RUNPOD_POD_ID}"
            runpodctl pod stop "${RUNPOD_POD_ID}" || echo "$(date -u +%FT%TZ) the stop failed: stop the pod by hand"
        ) >> /var/log/pod_ttl.log 2>&1 &
    fi
fi

ssh-keygen -A
/usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password
echo "felix-sim $(cat /opt/felix/image_tag): sshd up"
exec sleep infinity
