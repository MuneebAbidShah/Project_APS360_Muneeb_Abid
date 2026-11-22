# Live handwriting demo (CRNN + CTC), Windows

This folder runs the trained CRNN on your laptop's CPU and reads a word you hold up to the camera.
Your phone works as the webcam through **DroidCam**.

## 1. One-time setup

1. **Python**: install Python 3.12 from <https://www.python.org/downloads/windows/>
   ("Windows installer (64-bit)"). On the first installer screen, **tick "Add python.exe to PATH"**,
   then click *Install Now*.
2. **Unzip this folder**: right-click `live_demo.zip` → *Extract All…* → e.g. `Documents\live_demo`.
3. **Open a terminal in the folder**: open the extracted `live_demo` folder in File Explorer, click the
   address bar, type `cmd` and press Enter.
4. **Install the packages** (about 5 minutes, ~250 MB):

       python -m pip install -r requirements.txt

## 2. DroidCam (phone = webcam)

1. Phone: install **DroidCam** (Play Store / App Store). Open it and allow camera access. It shows a
   **WiFi IP** (e.g. `192.168.1.23`) and a port (`4747`).
2. Laptop: install the **DroidCam Client** for Windows from <https://droidcam.app> and open it.
3. Phone and laptop must be on the **same Wi-Fi**. Connect the client to the phone:
   - *Classic client*: choose **WiFi**, type the phone's IP into *Device IP*, keep port 4747, tick
     *Video*, click **Start**.
   - *New client (droidcam.app)*: click in the middle (or right-click → *Add* → *DroidCam*), pick your
     phone or **Use WiFi IP** and type the phone's IP (port 4747), then start/activate it.
   
   **The phone's live picture must appear inside the DroidCam Client window.** If the client is not
   connected, every program (including this demo) only gets DroidCam's *placeholder*: a fixed logo
   screen or a plain blue / orange / green screen.
4. Leave the DroidCam app open on the phone (screen on) and the client running.

## 2b. Easiest: read the phone directly over Wi-Fi (no DroidCam Client)

If the demo only shows the DroidCam placeholder, skip the Client completely:

1. **Quit the DroidCam Client** fully: right-click its icon in the system tray (bottom right, near the
   clock, maybe under the ^ arrow) → *Exit*, or Task Manager → DroidCam → *End task*.
2. **Close every browser tab** that shows the phone video (the phone app serves only one viewer).
3. On the phone, open the **DroidCam app** (screen on) and note its **WiFi IP**, e.g. `10.0.0.74`.
4. Run (with your IP):

       python live_demo.py --droidcam 10.0.0.74

The demo connects to the phone's video (its own default size first), keeps only the newest frame,
accepts only complete JPEG pictures (anything else is dropped) and reconnects by itself, waiting a
couple of seconds between tries so the phone isn't kept "busy". Addresses the phone doesn't have
(HTTP 404) are asked only once. When it connects it prints one **`DroidCam check:`** line per address
it tried, e.g.

    DroidCam check: http://10.0.0.74:4747/video | type=multipart/x-mixed-replace; boundary=--dcmjpeg | first bytes=ff d8 ff e0 ... | 1280x720 JPEG (row/col change 1.4/1.2) ok

It also saves the **first picture exactly as the phone sent it** to `droidcam_sample.jpg` next to
`live_demo.py` (the path is printed). If the line says the picture looks **SCRAMBLED** (copies side by
side, green lines), the stream is still shown, with a note. If the window actually looks fine, run with
`--no-garble-check`. If it really looks wrong, send `droidcam_sample.jpg` and the `DroidCam check:`
lines when asking for help. `--save-frame other.jpg` saves it somewhere else (`--save-frame ""`
turns it off). The FPS at the top left counts new pictures from the phone (usually 15-30).

If it says the phone is **busy**, repeat steps 1-3. If it can't connect, check that the phone and
laptop are on the same Wi-Fi (no guest network, no VPN) and that the IP matches the app.
`--droidcam 10.0.0.74:4747` or a full `http://10.0.0.74:4747/video?640x480` URL also work (a full
URL is used exactly as given). (The free app adds a small "droidcam.app" watermark; that's fine.)

Camera trouble with the Client instead? `--backend dshow`, `--backend msmf` or `--backend any`
picks the Windows camera API (the default tries DirectShow, then Media Foundation, then any).

## 3. Run

    python live_demo.py

It tries cameras 0-5 and uses the first one that gives a picture (the terminal says which).

- **Red warning "This looks like the DroidCam placeholder"** at the bottom: the demo is getting the
  placeholder, not your phone. Open the DroidCam Client, press Start (connect), and check that it shows
  your phone video. The warning disappears by itself once the picture is live.
- **Wrong camera** (e.g. a laptop webcam): press **n**. It tries the other cameras one by one and says
  in the window and the terminal which one it switched to. If no other camera works it stays on the
  current one and says "no other camera gives a picture"; it never closes.
- `python live_demo.py --list` checks cameras 0-5 and prints each one's name (when Windows tells it),
  resolution and whether it is a **live picture** or looks like a **placeholder**. Then start the right
  one, e.g. `python live_demo.py --camera 1`.
- A camera whose name contains **DroidCam** is tried first. To prefer another name, give any part of it
  (upper/lower case doesn't matter): `python live_demo.py --device-name droidcam`. Showing the names
  next to the numbers needs the small `pygrabber` package (in `requirements.txt`; or
  `python -m pip install pygrabber`). Without it the demo still works and `--list` prints the names
  without numbers.
- If something goes wrong, the program prints the full error and waits for **Enter** instead of
  closing, so you can copy the message.

Write a word in **dark pen, large** (letters ~1-2 cm), on plain white paper. Hold the phone ~20-30 cm
away, fill most of the **green box** with the word, and keep it steady and well lit.

| Key | What it does |
|---|---|
| `q` (or Esc) | quit |
| `space` | freeze / unfreeze |
| `c` | capture: freezes, then **type the true word**, press **Enter** to save (Esc cancels) |
| `a` / `d` | guide box narrower / wider |
| `w` / `s` | guide box taller / shorter |
| `n` | next camera |
| `m` | switch to PAGE mode (beta) |

The current mode is shown at the top left. **WORD mode** (above) is the default and the reliable one.

### PAGE mode (BETA)

Press **m** to switch to PAGE mode (press **m** again to go back). It keeps a running transcript of a
whole page, either one you are writing right now or an old notebook page. Hold the phone so the whole page is in view, and keep it steady.
About twice a second it finds the paper, straightens it, cuts it into lines and words and reads every word.
A word goes into the transcript (right-hand panel) only after its box has stayed put for a few readings.
**Green box** = in the transcript, **yellow** = not yet. Each word is added once; its reading can still improve.

**PAGE mode is beta and clearly less accurate than WORD mode.** On top of the model's own errors, it can
cut words in the wrong place (joined or split words), miss small punctuation, and get confused when the page
moves a lot or your hand covers it. Use WORD mode for the test set and for the demo's accuracy.

| Key (PAGE mode) | What it does |
|---|---|
| `r` | reset the transcript |
| `s` | save the transcript to `transcripts\page_<date>_<time>.txt` |
| `space` | freeze / unfreeze |
| `c` | page capture: saves **every word crop** of the page into `webcam_test` with the model's **guess** as the label |
| `m` | back to WORD mode |
| `q` | quit |

Page capture is a quick way to turn an old notebook page into test words, but **the labels are only guesses**:
those rows have `needs_check` = 1 in `labels.csv`, and the Step 5 notebook **does not score them** until you fix
the label and change `needs_check` to 0. Scoring needs the true words.

The small window **"What the model sees"** is the 32-pixel-high image the CRNN reads. It should look
like an IAM word: dark letters on a white background, cropped tightly. If it is empty or messy,
move closer, add light, or write darker.

## 4. Collect the webcam test set (~150 words)

1. Write ~150 different words (mix short and long words, a few capitalized words; ideally several people's handwriting).
2. For each word: put it in the green box, press **c**, type exactly what is written (capitals and
   punctuation included), press **Enter**. The counter at the top shows how many are saved.
3. Everything goes into the `webcam_test` folder next to `live_demo.py`: `images\0001.png`, ...
   and `labels.csv`. You can stop and continue later; numbering continues.
4. Upload: open the project folder on Google Drive in the browser and **drag the whole
   `webcam_test` folder** into it (next to `archive.zip`). Then run the Step 5 notebook again: it scores
   the CRNN, Tesseract and the EMNIST baseline on your words.

Typo in a label? Open `webcam_test\labels.csv` in Notepad and fix it (or delete that row).
Rows from a PAGE-mode page capture have `needs_check` = 1: correct the label, then set it to 0.

## Files

- `live_demo.py`: the demo. `python live_demo.py --help` lists every option.
- `crnn_model.pt`: the trained model (CPU weights + settings), `model_info.json`: where it came from.
- `src\`: the exact model, character-set and preprocessing code used in training (unchanged copies).
- `samples\`: a few IAM test words. Without a camera: `python live_demo.py --image samples\<file>.png`
  (it treats the picture as a camera frame and reads what's in the box).
