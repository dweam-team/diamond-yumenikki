#!/usr/bin/env bash

set -ex

ISO_URL="https://cdimages.ubuntu.com/xubuntu/releases/24.04/release/xubuntu-24.04.1-desktop-amd64.iso"
ISO_FILE="xubuntu-24.04.1-desktop-amd64.iso"
DISK_SIZE="80G"
VM_NAME="yume-nikki-vm"
SSH_PORT=33022

headless=0

use_installer=$()
if ! test -e "${VM_NAME}.qcow2"; then
  use_installer=1
  qemu-img create -f qcow2 ${VM_NAME}.qcow2 ${DISK_SIZE}
else
  use_installer=
  echo "${VM_NAME}.qcow2 already exists"
fi

#unix_socket="$PWD/unix-vm-control-socket"
#
#if test -e "$unix_socket"; then
#  unlink "$unix_socket"
#fi
#
#{
#  for i in {1..5}; do
#    if test -e "$unix_socket"; then
#      break
#    fi
#    sleep 1
#  done
#  if test -e "$unix_socket"; then
#    echo "help" | socat - "UNIX-CONNECT:$unix_socket"
#    sleep 1
#    echo "sendkey ret" | socat - "UNIX-CONNECT:$unix_socket"
#  fi
#  #virsh send-key yume-nikki-vm --codeset linux --holdtime 100 KEY_ENTER
#} &
#
#./run_xfce_send_keys.sh &
  #-monitor stdio \
  #-monitor "unix:$unix_socket,server,nowait" \

if ! test -z $use_installer; then
    echo "running installer"
    qemu-system-x86_64 \
      -name "$VM_NAME" \
      -qmp unix:qmp.sock,server=on,wait=off \
      -boot d \
      -cdrom "$ISO_FILE" \
      -m 4096 \
      -enable-kvm \
      -cpu host \
      -smp 4 \
      -drive file=${VM_NAME}.qcow2,format=qcow2 \
      -vga virtio \
      -display gtk,grab-on-hover=on \
      -net nic -net user
else
    echo "running system"
    {
        sleep 10
        sshpass -p dweam ssh -p "$SSH_PORT" dweam@localhost -o PreferredAuthentications=password -o StrictHostKeyChecking=no <<EOF
set -ex

sudo dpkg --add-architecture i386
if test -e /etc/apt/sources.list.d/winehq-noble.sources; then
    echo "winehq sources already present"
else
    # https://gitlab.winehq.org/wine/wine/-/wikis/Debian-Ubuntu
    sudo mkdir -pm755 /etc/apt/keyrings
    wget -O - https://dl.winehq.org/wine-builds/winehq.key \
      | sudo gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key -
    sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources
fi
sudo add-apt-repository ppa:kisak/kisak-mesa -y
sudo apt-get update
sudo apt-get install -y curl wget dosbox python3-{dev,pip,xlib} python-is-python3 rsync vim ncdu htop dfc xdotool mesa-utils mesa-vulkan-drivers libvulkan1 vulkan-tools x11-utils ffmpeg tree
sudo apt install --install-recommends winehq-staging -y
pip install --user numpy opencv-python bpython pillow scipy mss openai --break-system-packages
EOF
        rsync -auPz --rsh="sshpass -p dweam ssh -p $SSH_PORT -o PreferredAuthentications=password -o StrictHostKeyChecking=no -l dweam" \
            "/home/admin/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Yume Nikki/yumenikki" \
            dweam@localhost:"/home/dweam/"
        rsync -auPz --rsh="sshpass -p dweam ssh -p $SSH_PORT -o PreferredAuthentications=password -o StrictHostKeyChecking=no -l dweam" "./capture_data_guest.py" dweam@localhost:"/home/dweam/capture_data.py"
        rsync -auPz --rsh="sshpass -p dweam ssh -p $SSH_PORT -o PreferredAuthentications=password -o StrictHostKeyChecking=no -l dweam" dweam@localhost:"/home/dweam/capture_data_host.py" "./"
        rsync -av -f"+ */" -f"- *" --rsh="sshpass -p dweam ssh -p $SSH_PORT -o PreferredAuthentications=password -o StrictHostKeyChecking=no -l dweam" ./data/ dweam@localhost:"/home/dweam/data/"
        extra_xfce_args=
        if test "$headless" -eq 1; then
            extra_xfce_args=<<EOF
EOF
        fi
        sshpass -p dweam ssh -p "$SSH_PORT" dweam@localhost -o PreferredAuthentications=password -o StrictHostKeyChecking=no -R 21090:localhost:21090 &
        sshport_pid=$!
        sshpass -p dweam ssh -X -p "$SSH_PORT" dweam@localhost -o PreferredAuthentications=password -o StrictHostKeyChecking=no <<EOF
nohup xfce4-terminal --maximize --display=:0 &
EOF
        # periodically pull data
        watch -n 150 "rsync -auPz --remove-source-files --rsh='sshpass -p dweam ssh -p $SSH_PORT -o PreferredAuthentications=password -o StrictHostKeyChecking=no -l dweam' dweam@localhost:'/home/dweam/data/' './data/'"
    } &
    control_pid=$!
    qemu_args="
      -name "$VM_NAME" \
      -qmp unix:qmp.sock,server=on,wait=off \
      -boot d \
      -m 4096 \
      -enable-kvm \
      -cpu host \
      -smp 4 \
      -drive file=${VM_NAME}.qcow2,format=qcow2 \
      -vga virtio \
      -net nic -net user,hostfwd=tcp::33022-:22
"
    if test "$headless" -eq 0; then
        qemu_args="$qemu_args -display gtk,grab-on-hover=on"
    else
        qemu_args="$qemu_args -display None"
    fi
    qemu-system-x86_64 $(echo "$qemu_args");kill "$sshport_pid"; "$control_pid"
fi
