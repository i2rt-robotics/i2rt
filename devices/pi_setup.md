## 🔹 Backup and Restore Raspberry Pi SD Card with PiShrink

### 1. Find the SD card mounting location
```bash
lsblk
```

### 2. Create image from SD card

```bash
sudo dd if=/dev/sdX of=pi_system.img bs=4M status=progress
sync
```
*(replace `/dev/sdX` with your SD card device, e.g. `/dev/sdc`)*

### 3. Flash image into new SD card

```bash
sudo wipefs -a /dev/sdX
sudo dd if=pi_system.img of=/dev/sdX bs=4M status=progress
sync
```

### 4. Eject SD card

```bash
eject /dev/sdX
```

### 5. Reduce image size with PiShrink (optional)

```bash
sudo ./pishrink.sh pi_system.img pi_system_shrunk.img
```
