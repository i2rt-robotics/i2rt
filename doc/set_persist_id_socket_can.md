# Socket can is easy to use, but the default devie name is can{idx} and will vary depend on the order the device got connected to the computer. Below is the precedure of setting up persist ID for those socket CAN devices. 
 
## For canable devices, goto https://canable.io/updater/ to flush its firmware to candlelight to use socketcan, YAM comes with pre-flushed candlelight firmware.


## Step1: find sysfd paths for can devices

```shell
$ ls -l /sys/class/net/can*
```

this should  give you something liks
```shell
lrwxrwxrwx 1 root root 0 Jul 15 14:35 /sys/class/net/can0 -> ../../devices/platform/soc/your_can_device/can0
lrwxrwxrwx 1 root root 0 Jul 15 14:35 /sys/class/net/can1 -> ../../devices/platform/soc/your_can_device/can1
lrwxrwxrwx 1 root root 0 Jul 15 14:35 /sys/class/net/can2 -> ../../devices/platform/soc/your_can_device/can2
```

## Step 2: Use udevadm to Gather Attributes
```shell
udevadm info -a -p /sys/class/net/can0 | grep -i serial
```

## Step 3: Create udev Rules
edit `/etc/udev/rules.d/90-can.rules`
```shell
sudo vim /etc/udev/rules.d/90-can.rules
```
add
```
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="004E00275548501220373234", NAME="can_follow_l"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="0031005F5548501220373234", NAME="can_follow_r"
```

IMPORTANT!!!: Name should start with can (for USB-CAN adapter) or en/eth (for EtherCAT-CAN adapter). And the maximum length limit for a CAN interface name is 13 characters.

## Step 4: Reload udev Rules
```shell
sudo udevadm control --reload-rules && sudo systemctl restart systemd-udevd && sudo udevadm trigger
```

If needed, plug unplug the candevice to make sure the change is effective. 

run the following command to set up the can device, and you need to run this command after every reboot.
```
sudo ip link set up can_right type can bitrate 1000000
```

## Step 5: Verify the can device
```shell
$ ip link show
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: enp5s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
    link/ether d8:43:ae:b7:43:0b brd ff:ff:ff:ff:ff:ff
5: tailscale0: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1280 qdisc fq_codel state UNKNOWN mode DEFAULT group default qlen 500
    link/none 
6: can_right: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can 
7: can_left: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can 
```

You can see that this can device got it's name can_right/can_left

