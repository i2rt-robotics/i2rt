# Setting Up SteamVR for a Wired Vive Tracker (No Headset)

A Vive tracker is tracked by SteamVR, and i2rt reads its pose as an OpenVR client (`i2rt/yumi/vive_tracker.py`).
The tracker is **wired**: it connects to the host with a USB cable, so it needs no wireless dongle and no pairing
-- SteamVR picks it up as soon as it is plugged in. No Vive headset (HMD) is needed either, but SteamVR expects one
by default, so it has to be told otherwise. Below is the procedure for installing SteamVR on a Linux host and
checking that the tracker is tracked.

**Requirements:** an **x86-64** Linux host (the OpenVR client library bundled with the `openvr` Python package is
x86-64 only), the tracker plugged into the host by USB, and at least one base station powered and in view of the
tracker.

## Step 1: Install Steam

```shell
wget https://cdn.cloudflare.steamstatic.com/client/installer/steam.deb
sudo apt install ./steam.deb     # apt, not dpkg -i, so its dependencies are pulled in too
```

The package (`steam-launcher`) already ships the udev rules that let SteamVR open the tracker's USB device without
root. **Do not** also install `steam-devices`: it conflicts with `steam-launcher`, and apt resolves the conflict by
removing Steam.

## Step 2: Install SteamVR

Log in to Steam, go to **Store**, search for **SteamVR** and install it.

## Step 3: Remove the Headset Requirement

Without an HMD, SteamVR refuses to start. Edit:

```shell
~/.local/share/Steam/steamapps/common/SteamVR/resources/settings/default.vrsettings
```

and change

```
"requireHmd": true,
```

to

```
"requireHmd": false,
```

## Step 4: Restart SteamVR

Click the **VR** button at the top right of the Steam window. SteamVR should come up without asking for a headset.

## Step 5: Plug In the Tracker

Connect the tracker to the host with its USB cable. There is nothing to pair: SteamVR lists the tracker as soon as
it enumerates on USB, and its LED turns solid green once a base station sees it.

## Step 6: Verify the Tracker

```shell
python i2rt/yumi/vive_tracker.py
```

This lists every device SteamVR reports (class, serial, model), then redraws their live 4x4 poses. Tracker serials
look like `LHR-XXXXXXXX`; move the tracker and check that its matrix changes. The serial is what i2rt code uses to
pick a tracker (e.g. `Yumi(tracker_serial=...)`).

## Troubleshooting

- **The tracker is not listed:** check the USB cable and port (`lsusb` should show the tracker), and that SteamVR is
  running. Unplug and replug the tracker, then restart SteamVR if it still does not appear.
- **Listed, but the pose is not valid (LED not solid green):** check line of sight. At least one base station must
  see the tracker's sensors, within the play area; bodies, clothing and furniture all block it. The cable itself can
  shadow sensors, so route it away from the tracker's face.
- **SteamVR asks for a headset or will not start:** check that `"requireHmd"` is still `false` (Step 3); a SteamVR
  update may overwrite that file.
- **`vive_tracker.py` says openvr is unavailable:** the host is not x86-64, or the `openvr` package is missing
  (`uv sync`).
