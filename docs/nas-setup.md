# NAS Setup for Ollama Hosts

Test suite images are stored on the QNAP NAS (snorlax) and mounted on each Ollama GPU host so models can read images directly.

## NAS Location

```
snorlax:/share/transfer/vlm-geometry-bench/testsuite/
```

92 PNG images + CSV ground truth files + manifest.yaml (~1 MB total).

## Syncing Testsuite to NAS (from Windows dev machine)

```bash
rsync -avz /d/python/imagegen/testsuite/ snorlax:/share/transfer/vlm-geometry-bench/testsuite/
```

## Mounting on an Ollama Host (Ubuntu)

### 1. Install CIFS utilities (if not already installed)

```bash
sudo apt install cifs-utils
```

### 2. Create mount point

```bash
sudo mkdir -p /mnt/nas/vlm-geometry-bench
```

### 3. Create credentials file

```bash
sudo bash -c 'echo "username=mike" > /etc/smbcredentials-snorlax && read -sp "Password: " pw && echo "password=$pw" >> /etc/smbcredentials-snorlax'
sudo chmod 600 /etc/smbcredentials-snorlax
```

### 4. Add to fstab (permanent mount)

```bash
echo '//snorlax/transfer/vlm-geometry-bench /mnt/nas/vlm-geometry-bench cifs credentials=/etc/smbcredentials-snorlax,uid=1000,iocharset=utf8 0 0' | sudo tee -a /etc/fstab
sudo systemctl daemon-reload
```

### 5. Mount and verify

```bash
sudo mount /mnt/nas/vlm-geometry-bench
ls /mnt/nas/vlm-geometry-bench/testsuite/ | head -3
```

### 6. Test with Ollama

```bash
ollama run llava:7b "How many spots are in this image? /mnt/nas/vlm-geometry-bench/testsuite/CTRL_single_wb.png"
```

## Current Hosts

| Host    | IP              | GPU                | Mount Status |
|---------|-----------------|--------------------|--------------|
| jolteon | 192.168.25.x    | NVIDIA Jetson ARM64| Mounted      |
| luxray  | 192.168.25.x    | NVIDIA GPU x86_64  | Mounted      |

## Using with manual-test.cmd

From the Ollama host directly:

```bash
ollama run llava:7b
# Then paste prompts using /mnt/nas/vlm-geometry-bench/testsuite/ paths
```

From Windows (images sent over API from local disk):

```cmd
manual-test.cmd llava:7b
```

From Windows pointing at NAS-mounted host (set OLLAMA_HOST first):

```cmd
set OLLAMA_HOST=http://<host-ip>:11434
manual-test.cmd llava:7b
```
