"""
BEAT ANALYSIS SERVER
====================
This is the backend server. It does two things:
1. Accepts an audio file upload from the website
2. Analyzes it with librosa to find beat timestamps
3. Stores the result so Roblox can fetch it

HOW TO RUN:
  pip install flask flask-cors librosa soundfile numpy scipy
  python app.py

It runs on port 5000 by default.
"""

import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import uuid
import os
import json
import time
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://typeoneappolo.github.io/roblox-rhythm/"}})


# where we store uploaded files temporarily
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# in-memory storage for beat data
# in a real production app you'd use a database, but this works fine for now
beat_storage = {}

# [NEW] tracks when each song was stored so we can expire old ones
storage_timestamps = {}

# sync signals — when Roblox starts a song, it writes here
# the website polls this to know when to auto-play the audio
sync_signals = {}

# stop signals — when a player leaves, the server writes here
# the website polls this to know when to stop playback
stop_signals = {}

# ========================================
# [NEW] CLEANUP CONFIG
# ========================================
# how long to keep files before auto-deleting (seconds)
# 2 hours = plenty of time to play, then they get cleaned up
FILE_MAX_AGE = 2 * 60 * 60       # 2 hours for audio/json files
MEMORY_MAX_AGE = 4 * 60 * 60     # 4 hours for in-memory beat data
CLEANUP_INTERVAL = 10 * 60       # run cleanup every 10 minutes


def cleanup_old_files():
    """
    Deletes uploaded audio files and JSON beat files that are older
    than FILE_MAX_AGE. Also purges old entries from beat_storage
    so memory doesn't grow forever.
    
    Runs on a background thread every CLEANUP_INTERVAL seconds.
    """
    while True:
        try:
            now = time.time()
            removed_files = 0
            removed_memory = 0
            
            # clean up files in the uploads folder
            if os.path.exists(UPLOAD_FOLDER):
                for filename in os.listdir(UPLOAD_FOLDER):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    if os.path.isfile(filepath):
                        file_age = now - os.path.getmtime(filepath)
                        if file_age > FILE_MAX_AGE:
                            os.remove(filepath)
                            removed_files += 1
            
            # clean up in-memory beat storage
            stale_ids = []
            for song_id, timestamp in list(storage_timestamps.items()):
                if now - timestamp > MEMORY_MAX_AGE:
                    stale_ids.append(song_id)
            
            for song_id in stale_ids:
                beat_storage.pop(song_id, None)
                storage_timestamps.pop(song_id, None)
                sync_signals.pop(song_id, None)
                removed_memory += 1
            
            if removed_files > 0 or removed_memory > 0:
                print(f"[Cleanup] Removed {removed_files} files, {removed_memory} memory entries")
        
        except Exception as e:
            print(f"[Cleanup] Error: {e}")
        
        time.sleep(CLEANUP_INTERVAL)


def analyze_audio(filepath):
    """
    Advanced beat analysis using:
    1. HPSS — separates percussive (drums) from harmonic (melody/chords)
    2. Per-band onset detection — runs onset detection on 4 separate 
       frequency ranges so each lane gets its OWN set of onsets
    3. Mel spectrogram — perceptually weighted so bass doesn't dominate
    
    This produces a much more balanced and musical note map than the
    old "pick dominant frequency band" approach.
    """
    
    # Fix 1: Load ONLY the skeleton of the song. 
    # sr=8000 is 'telephone quality', but perfect for finding beats.
    y, sr = librosa.load(filepath, sr=8000, mono=True, duration=180)

    # Fix 2: Delete the audio from memory the moment we get the beat track
    # This prevents the RAM spike from staying high.
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # Fix 3: Convert frames to timestamps immediately
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Cleanup: Manually tell Python to clear the audio data
    del y

    return {
          "bpm": round(float(tempo), 2),
          "beats": beat_times.tolist()
      }
      
    # =============================================
    # STEP 1: Harmonic-Percussive Source Separation
    # =============================================
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # beat track on percussive signal (more stable than full mix)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    
    # =============================================
    # STEP 2: Mel spectrogram in 4 frequency bands
    # =============================================
    # [CHANGED] use a MIX of percussive + harmonic so we detect vocals/melody AND drums
    # 60% percussive keeps rhythmic backbone, 40% harmonic catches sustained
    # notes like vocals, synth leads, guitar riffs, etc.
    y_mixed = 0.6 * y_percussive + 0.4 * y_harmonic
    
    n_mels = 128
    S = librosa.feature.melspectrogram(y=y_mixed, sr=sr, n_mels=n_mels)
    
    # [CHANGED] separate harmonic-only spectrogram for hold note detection
    # sustained energy (long vocal notes, synth pads) lives in the harmonic
    # signal — percussive energy dies off instantly so it was barely triggering holds
    S_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=n_mels)
    
    # [CHANGED] pure percussive spectrogram — catches kicks/drums during
    # instrumental sections where the harmonic signal is silent. Without this,
    # the 40% harmonic component in the mix dilutes the percussive peaks
    # to zero during drops (no vocals = 40% of mix is silence)
    S_percussive = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=n_mels)
    
    # split into 4 bands (equal mel bands = perceptually equal ranges)
    bands_per_lane = n_mels // 4
    band_specs = []
    band_specs_harmonic = []   # harmonic-only bands for hold detection
    band_specs_percussive = [] # [CHANGED] percussive-only for instrumental sections
    for i in range(4):
        start_idx = i * bands_per_lane
        end_idx = (i + 1) * bands_per_lane    # [FIX] was "end" which is a reserved word
        band_specs.append(S[start_idx:end_idx, :])
        band_specs_harmonic.append(S_harmonic[start_idx:end_idx, :])
        band_specs_percussive.append(S_percussive[start_idx:end_idx, :])
    
    # =============================================
    # STEP 3: Onset detection per band
    # =============================================
    beat_data = []
    
    for lane_idx in range(4):
        lane = lane_idx + 1
        
        # compute onset strength for just this frequency band
        band_onset = np.sum(band_specs[lane_idx], axis=0)
        
        # smooth it slightly to avoid detecting noise
        from scipy.ndimage import uniform_filter1d
        band_onset = uniform_filter1d(band_onset, size=3)
        
        # normalize this band's onset envelope to 0-1
        band_max = float(np.max(band_onset))
        if band_max <= 0:
            continue
        band_onset_norm = band_onset / band_max
        
        # [CHANGED] compute sustained energy for hold note detection
        # use the HARMONIC spectrogram — sustained vocals, synths, guitar etc.
        # live in the harmonic signal. percussive energy dies off instantly
        # so it was almost never triggering holds before.
        band_energy = np.mean(band_specs_harmonic[lane_idx], axis=0)
        band_energy_max = float(np.max(band_energy))
        band_energy_norm = band_energy / (band_energy_max + 1e-8)
        
        # [CHANGED] also blend in a bit of the mixed energy to catch
        # percussive sustains like cymbal crashes, sustained bass hits
        band_energy_mixed = np.mean(band_specs[lane_idx], axis=0)
        be_mixed_max = float(np.max(band_energy_mixed))
        be_mixed_norm = band_energy_mixed / (be_mixed_max + 1e-8)
        band_energy_norm = np.maximum(band_energy_norm, be_mixed_norm * 0.5)
        
        # [CHANGED] hold detection parameters — lowered thresholds to catch more holds
        hold_energy_threshold = 0.10    # was 0.18 — energy must stay above this to count as sustain
        min_hold_seconds = 0.22         # was 0.30 — shorter minimum so quick sustains register
        max_hold_seconds = 5.0          # was 4.0 — allow longer holds for ballads/slow songs
        hop_duration = 512.0 / sr       # seconds per spectrogram frame
        max_hold_frames = int(max_hold_seconds / hop_duration)
        
        # find peaks in this band's onset envelope
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(
            band_onset_norm,
            height=0.2,
            prominence=0.1,
            distance=4
        )
        
        # [CHANGED] PERCUSSIVE FALLBACK — also detect onsets in the pure
        # percussive signal. During instrumental sections (hardstyle drops,
        # drum breaks, etc.) the harmonic signal is silent so the 40% harmonic
        # component in the mix dilutes peaks to nothing. Running a second pass
        # on pure percussive catches those kicks/hits, then we merge the two
        # peak sets and deduplicate so vocal sections don't get doubled.
        perc_onset = np.sum(band_specs_percussive[lane_idx], axis=0)
        perc_onset = uniform_filter1d(perc_onset, size=3)
        perc_max = float(np.max(perc_onset))
        if perc_max > 0:
            perc_onset_norm = perc_onset / perc_max
            perc_peaks, _ = find_peaks(
                perc_onset_norm,
                height=0.25,       # slightly higher threshold to avoid noise
                prominence=0.12,
                distance=4
            )
            # merge: only add percussive peaks that aren't within 3 frames of
            # an existing mixed peak (avoids doubling up during vocal sections)
            mixed_set = set(peaks)
            for pp in perc_peaks:
                is_duplicate = False
                for mp in peaks:
                    if abs(int(pp) - int(mp)) <= 3:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    peaks = np.append(peaks, pp)
            peaks = np.sort(peaks).astype(int)
        
        for peak_frame in peaks:
            t = float(librosa.frames_to_time(int(peak_frame), sr=sr))
            # [CHANGED] use max of mixed and percussive onset — peaks from the
            # percussive fallback might show low values in band_onset_norm since
            # the mixed signal was diluted. Taking the max means those peaks still
            # get proper strength values and don't get filtered by difficulty.
            mixed_str = float(band_onset_norm[peak_frame]) if peak_frame < len(band_onset_norm) else 0
            perc_str = float(perc_onset_norm[peak_frame]) if perc_max > 0 and peak_frame < len(perc_onset_norm) else 0
            strength = max(mixed_str, perc_str)
            
            # hold detection: walk forward through energy frames
            # to see how long the note sustains after onset
            hold_end_frame = peak_frame
            gap_allowance = 5  # [CHANGED] was 2 — allow dips from breaths, vibrato, transients
            gap_counter = 0
            
            for cf in range(peak_frame + 1, min(peak_frame + max_hold_frames, len(band_energy_norm))):
                if band_energy_norm[cf] >= hold_energy_threshold:
                    hold_end_frame = cf
                    gap_counter = 0
                else:
                    gap_counter += 1
                    if gap_counter > gap_allowance:
                        break
            
            hold_seconds = (hold_end_frame - peak_frame) * hop_duration
            hold_dur = round(hold_seconds, 3) if hold_seconds >= min_hold_seconds else 0
            
            beat_data.append({
                "time": round(t, 3),
                "strength": round(strength, 3),
                "lane": lane,
                "hold": hold_dur
            })
    
    # =============================================
    # STEP 4: Sort by time, deduplicate, enforce gaps
    # =============================================
    beat_data.sort(key=lambda x: x["time"])
    
    min_same_lane_gap = 0.10
    min_any_lane_gap = 0.03
    
    filtered = []
    last_time_per_lane = {1: -1.0, 2: -1.0, 3: -1.0, 4: -1.0}
    last_global_time = -1.0
    
    for beat in beat_data:
        lane = beat["lane"]
        t = beat["time"]
        
        if t - last_time_per_lane[lane] < min_same_lane_gap:
            continue
        
        if t - last_global_time < min_any_lane_gap:
            continue
        
        filtered.append(beat)
        last_time_per_lane[lane] = t
        last_global_time = t
    
    # second pass: truncate hold notes that overlap with the next note in the same lane
    for i, beat in enumerate(filtered):
        if beat.get("hold", 0) > 0:
            hold_end = beat["time"] + beat["hold"]
            # find next note in same lane
            for j in range(i + 1, len(filtered)):
                if filtered[j]["lane"] == beat["lane"]:
                    next_time = filtered[j]["time"]
                    # leave a 0.15s gap before the next note so player can release and re-tap
                    if hold_end > next_time - 0.15:
                        new_hold = next_time - beat["time"] - 0.15
                        beat["hold"] = round(max(0, new_hold), 3)
                    break
    
    # =============================================
    # [CHANGED] STEP 4b: Cap hold notes at ~35% of total
    # =============================================
    # the lowered hold thresholds catch way more sustained notes now,
    # but too many holds makes the map feel sluggish — 65/35 tap/hold
    # ratio keeps it punchy with holds as satisfying variety, not the norm.
    # converts the SHORTEST holds back to taps first since those are the
    # least interesting to actually hold anyway.
    hold_notes_idx = [i for i, b in enumerate(filtered) if b.get("hold", 0) > 0]
    total_filtered = len(filtered)
    
    if total_filtered > 0 and len(hold_notes_idx) > 0:
        hold_ratio = len(hold_notes_idx) / total_filtered
        target_max_ratio = 0.35
        
        if hold_ratio > target_max_ratio:
            # sort hold indices by hold duration ascending — kill shortest first
            hold_notes_idx.sort(key=lambda i: filtered[i]["hold"])
            
            target_hold_count = int(total_filtered * target_max_ratio)
            excess = len(hold_notes_idx) - target_hold_count
            
            for k in range(excess):
                filtered[hold_notes_idx[k]]["hold"] = 0
    
    beat_data = filtered
    
    # =============================================
    # STEP 5: Balance lanes if still lopsided
    # =============================================
    lane_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for b in beat_data:
        lane_counts[b["lane"]] += 1
    
    total = len(beat_data)
    if total > 0:
        target_per_lane = total / 4
        
        for b in beat_data:
            lane = b["lane"]
            if lane_counts[lane] > target_per_lane * 2.0:
                min_lane = min(lane_counts, key=lane_counts.get)
                if lane_counts[min_lane] < target_per_lane * 0.5:
                    lane_counts[lane] -= 1
                    lane_counts[min_lane] += 1
                    b["lane"] = min_lane
    
    duration = round(float(librosa.get_duration(y=y, sr=sr)), 3)
    
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    
    return {
        "tempo": round(float(tempo), 1),
        "duration": duration,
        "totalBeats": len(beat_data),
        "beats": beat_data
    }


@app.route("/upload", methods=["POST"])
def upload_audio():
    """
    POST /upload
    Accepts a multipart form upload with an audio file.
    Analyzes it and returns a unique ID you can use to fetch the data later.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    song_id = str(uuid.uuid4())[:8]
    
    filepath = os.path.join(UPLOAD_FOLDER, f"{song_id}_{file.filename}")
    file.save(filepath)
    
    try:
        result = analyze_audio(filepath)
        result["songId"] = song_id
        result["filename"] = file.filename
        
        beat_storage[song_id] = result
        storage_timestamps[song_id] = time.time()    # [NEW] track when it was stored
        
        # save JSON backup
        json_path = os.path.join(UPLOAD_FOLDER, f"{song_id}.json")
        with open(json_path, "w") as f:
            json.dump(result, f)
        
        return jsonify({
            "success": True,
            "songId": song_id,
            "tempo": result["tempo"],
            "duration": result["duration"],
            "totalBeats": result["totalBeats"],
            "message": f"Analyzed {result['totalBeats']} beats at {result['tempo']} BPM"
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to analyze: {str(e)}"}), 500
    
    finally:
        # clean up the audio file immediately (we only need the JSON now)
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/beats/<song_id>", methods=["GET"])
def get_beats(song_id):
    """
    GET /beats/<song_id>
    This is what Roblox calls to get the beat data.
    """
    if song_id in beat_storage:
        storage_timestamps[song_id] = time.time()   # [NEW] refresh expiry on access
        return jsonify(beat_storage[song_id])
    
    json_path = os.path.join(UPLOAD_FOLDER, f"{song_id}.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            beat_storage[song_id] = data
            storage_timestamps[song_id] = time.time()  # [NEW]
            return jsonify(data)
    
    return jsonify({"error": "Song not found"}), 404


@app.route("/beats/<song_id>/roblox", methods=["GET"])
def get_beats_roblox(song_id):
    """
    GET /beats/<song_id>/roblox
    
    Same as /beats but formatted specifically for Roblox.
    """
    data = None
    if song_id in beat_storage:
        data = beat_storage[song_id]
        storage_timestamps[song_id] = time.time()   # [NEW]
    else:
        json_path = os.path.join(UPLOAD_FOLDER, f"{song_id}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
    
    if not data:
        return jsonify({"error": "Song not found"}), 404
    
    roblox_beats = []
    for beat in data["beats"]:
        lane = beat.get("lane", None)
        if lane is None:
            strength = beat["strength"]
            if strength < 0.25:
                lane = 1
            elif strength < 0.50:
                lane = 2
            elif strength < 0.75:
                lane = 3
            else:
                lane = 4
        
        roblox_beats.append({
            "t": beat["time"],
            "s": beat.get("strength", 0.5),
            "l": lane,
            "h": beat.get("hold", 0)
        })
    
    return jsonify({
        "id": song_id,
        "bpm": data["tempo"],
        "dur": data["duration"],
        "n": len(roblox_beats),
        "fn": data.get("filename", song_id),   # filename for display in recent songs
        "beats": roblox_beats
    })


@app.route("/sync/<song_id>/start", methods=["POST"])
def sync_start(song_id):
    sync_signals[song_id] = True
    stop_signals.pop(song_id, None)  # clear any pending stop
    return jsonify({"ok": True})


@app.route("/sync/<song_id>/stop", methods=["POST"])
def sync_stop(song_id):
    stop_signals[song_id] = True
    sync_signals.pop(song_id, None)
    return jsonify({"ok": True})


@app.route("/sync/<song_id>/poll", methods=["GET"])
def sync_poll(song_id):
    # check for stop signal first
    if stop_signals.get(song_id):
        del stop_signals[song_id]
        return jsonify({"play": False, "stop": True})
    if sync_signals.get(song_id):
        del sync_signals[song_id]
        return jsonify({"play": True, "stop": False})
    return jsonify({"play": False, "stop": False})


@app.route("/health", methods=["GET"])
def health_check():
    """Health check — also shows how many songs are cached and files on disk."""
    file_count = len(os.listdir(UPLOAD_FOLDER)) if os.path.exists(UPLOAD_FOLDER) else 0
    return jsonify({
        "status": "ok",
        "songs_in_memory": len(beat_storage),
        "files_on_disk": file_count
    })


@app.route("/")
def serve_frontend():
    directory = os.path.dirname(os.path.abspath(__file__))
    return flask.send_from_directory(directory, "index.html")



if __name__ == "__main__":
    # [NEW] start background cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()

    print("=" * 50)
    print("BEAT ANALYSIS SERVER")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /upload              — upload an audio file")
    print("  GET  /beats/<id>          — get beat data (full)")
    print("  GET  /beats/<id>/roblox   — get beat data (roblox format)")
    print("  POST /sync/<id>/start     — signal website to play")
    print("  POST /sync/<id>/stop      — signal website to stop")
    print("  GET  /sync/<id>/poll      — website polls this")
    print("  GET  /health              — check if server is running")
    print()
    print(f"  [Cleanup] Auto-cleanup every {CLEANUP_INTERVAL // 60} min")
    print(f"  [Cleanup] Files expire after {FILE_MAX_AGE // 3600} hours")
    print(f"  [Cleanup] Memory expires after {MEMORY_MAX_AGE // 3600} hours")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, debug=False)


