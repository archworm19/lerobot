
mac m1
/dev/tty.usbmodem5AE60837301 right_arm

/dev/tty.usbmodem5AE60864731 left_arm


raspberry pi server
uv run python -m lerobot.felix_jr.setup.find_ports
Uninstalled 1 package in 0.70ms
Installed 1 package in 1ms
Unplug the USB cable of left_arm (only that one), then press Enter.
left_arm: /dev/ttyACM1 (serial:5AE6086473). Plug it back in.
Unplug the USB cable of right_arm (only that one), then press Enter.
right_arm: /dev/ttyACM0 (serial:5AE6083730). Plug it back in.

Saved to /home/zcecere/.cache/huggingface/lerobot/calibration/felix_jr/ports.json:
  /dev/ttyACM1 left_arm
  /dev/ttyACM0 right_arm