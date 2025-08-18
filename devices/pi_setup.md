## Create image from SD card
sudo dd if=/dev/sdb of=pi_system.img bs=4M status=progress
sync

## Flash image into new SD card
sudo wipefs -a /dev/sdc
sudo dd if=pi_system.img of=/dev/sdc bs=4M status=progress
sync

## Eject SD card
eject /dev/sdc

## Reduce image size
sudo ./pishrink.sh pi_system.img pi_system_shrunk.img
