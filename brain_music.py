import redis
import time
import pyaudio
import numpy as np
import threading
from collections import deque
import sounddevice as sd

r = redis.Redis()
RATE = 44100
DEVICE = None

C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
NOTE_NAMES = {48:'C3',50:'D3',52:'E3',53:'F3',55:'G3',57:'A3',59:'B3',
              60:'C4',62:'D4',64:'E4',65:'F4',67:'G4',69:'A4',71:'B4',72:'C5',74:'D5',76:'E5',
              77:'F5',79:'G5',81:'A5',83:'B5',84:'C6',86:'D6',88:'E6',89:'F6',91:'G6'}

C_MAJOR_CHORDS = [
    [48, 52, 55],
    [50, 53, 57],
    [52, 55, 59],
    [53, 57, 60],
    [55, 59, 62],
    [57, 60, 64],
    [59, 62, 65],
    [60, 64, 67],
]
CHORD_NAMES  = ['Cmaj','Dmin','Emin','Fmaj','Gmaj','Amin','Bdim','Cmaj+']
STATE_LABELS = ['deep-delta','theta','theta/alpha','alpha','alpha','alpha/beta','beta','beta']

CHORD_MELODY_POOL = {
    0: [48, 52, 55, 60, 64, 67],
    1: [50, 53, 57, 62, 65, 69],
    2: [52, 55, 59, 60, 64, 67],
    3: [53, 57, 60, 65, 69, 72],
    4: [55, 59, 62, 67, 71, 74],
    5: [57, 60, 64, 69, 72, 76],
    6: [59, 62, 65, 71, 74],
    7: [60, 64, 67, 72, 76],
}

# Sweet extensions per chord — 7ths, 9ths, 11ths high up
CHORD_EXTENSIONS = {
    0: [79, 83, 84, 88],   # Cmaj — G5 B5 C6 E6 (maj7, 9th)
    1: [81, 84, 86, 89],   # Dmin — A5 C6 D6 F6 (min7, 9th)
    2: [79, 83, 86, 88],   # Emin — G5 B5 D6 E6 (min7, 9th)
    3: [81, 84, 88, 89],   # Fmaj — A5 C6 E6 F6 (maj7, 9th)
    4: [79, 83, 86, 89],   # Gmaj — G5 B5 D6 F6 (dom7, 9th)
    5: [81, 84, 88, 91],   # Amin — A5 C6 E6 G6 (min7, 9th)
    6: [83, 86, 88, 91],   # Bdim — B5 D6 E6 G6 (dim7)
    7: [84, 88, 91, 91],   # Cmaj+ — C6 E6 G6 (high maj7)
}

raw = {'alpha': deque(maxlen=30), 'delta': deque(maxlen=30),
       'theta': deque(maxlen=30), 'beta':  deque(maxlen=30)}
ema = {'alpha': 0.5, 'delta': 0.5, 'theta': 0.5, 'beta': 0.5}
EMA_ALPHA = 0.25

accumulated_score = deque(maxlen=15)
last_chord_time = 0
current_chord_idx = 3
current_chord = C_MAJOR_CHORDS[3]
iteration = 0
lock = threading.Lock()

def midi_to_freq(note):
    return 440.0 * (2 ** ((note - 69) / 12.0))

def read_band(key, name):
    val = r.get(key)
    if not val:
        return ema[name], 0.0
    v = float(val)
    raw[name].append(v)
    lo = min(raw[name])
    hi = max(raw[name])
    norm = (v - lo) / (hi - lo) if hi != lo else 0.5
    ema[name] = EMA_ALPHA * norm + (1 - EMA_ALPHA) * ema[name]
    return ema[name], v

def compute_all():
    a, a_raw = read_band('spectral.channel1.alpha', 'alpha')
    d, d_raw = read_band('spectral.channel1.delta', 'delta')
    t, t_raw = read_band('spectral.channel1.theta', 'theta')
    b, b_raw = read_band('spectral.channel1.beta',  'beta')
    return a, d, t, b, a_raw, d_raw, t_raw, b_raw

def compute_score(a, d, t, b):
    total = a + d + t + b + 0.001
    score = (a/total * 2.0) + (b/total * 1.5) - (d/total * 1.0)
    return max(0.0, min(1.0, score / 2.0))

def pick_melody_note():
    pool = CHORD_MELODY_POOL[current_chord_idx]
    a = ema['alpha']
    b = ema['beta']
    base_idx = int(a * (len(pool) - 1))
    jitter = max(1, int(b * 3.0))
    idx = base_idx + np.random.randint(-jitter, jitter + 1)
    return pool[max(0, min(idx, len(pool) - 1))]

def bar(val, width=20):
    filled = int(val * width)
    return '█' * filled + '░' * (width - filled)

def lowpass_filter(wave, cutoff_norm):
    cutoff = 0.01 + cutoff_norm * 0.99
    rc = 1.0 - cutoff
    filtered = np.zeros_like(wave)
    prev = 0.0
    for i in range(len(wave)):
        prev = cutoff * wave[i] + rc * prev
        filtered[i] = prev
    return filtered

def apply_delay(wave, delay_time, feedback, mix):
    delay_samples = int(delay_time * RATE)
    if delay_samples == 0:
        return wave
    output = wave.copy()
    for i in range(delay_samples, len(wave)):
        output[i] += feedback * output[i - delay_samples]
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * np.max(np.abs(wave))
    return wave * (1 - mix) + output * mix

def make_wave(freqs, duration, volume=0.4, brightness=0.5,
              cutoff=0.8, attack=0.15, delay_time=0.3,
              delay_feedback=0.3, delay_mix=0.2):
    t = np.linspace(0, duration, int(RATE * duration), False)
    wave = np.zeros(len(t))
    for freq in freqs:
        wave += np.sin(2 * np.pi * freq * t)
        wave += brightness * 0.5  * np.sin(2 * np.pi * freq * 2 * t)
        wave += brightness * 0.25 * np.sin(2 * np.pi * freq * 3 * t)
        wave += (brightness**2) * 0.15 * np.sin(2 * np.pi * freq * 4 * t)
        wave += (brightness**3) * 0.08 * np.sin(2 * np.pi * freq * 5 * t)
    wave = wave / len(freqs)
    wave = lowpass_filter(wave, cutoff)
    attack_s = max(0.01, min(attack, duration * 0.8))
    fade_s = min(int(RATE * 0.15), len(wave) // 4)
    attack_samples = int(RATE * attack_s)
    envelope = np.ones(len(t))
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-fade_s:] = np.linspace(1, 0, fade_s)
    wave = wave * envelope
    wave = apply_delay(wave, delay_time, delay_feedback, delay_mix)
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    wave = wave * volume
    stereo = np.column_stack([wave, wave])
    return stereo.astype(np.float32).tobytes()

def make_glide_wave(freq_start, freq_end, duration, volume=0.12,
                    brightness=0.4, cutoff=0.35, attack=0.4,
                    delay_time=0.4, delay_feedback=0.97, delay_mix=0.92,
                    tail=5.0):
    """Smooth gliding tone between two frequencies with long reverb tail."""
    n = int(RATE * duration)
    t = np.linspace(0, duration, n, False)
    interp = (1 - np.cos(np.pi * np.linspace(0, 1, n))) / 2
    freq = freq_start + (freq_end - freq_start) * interp
    phase = np.cumsum(freq) / RATE
    wave = np.sin(2 * np.pi * phase)
    wave += brightness * 0.4 * np.sin(2 * np.pi * phase * 2)
    wave += (brightness**2) * 0.2 * np.sin(2 * np.pi * phase * 3)
    wave = wave / 1.6
    wave = lowpass_filter(wave, cutoff)
    attack_s = max(0.05, min(attack, duration * 0.7))
    fade_s = min(int(RATE * 1.5), n // 2)
    attack_samples = int(RATE * attack_s)
    envelope = np.ones(n)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-fade_s:] = np.linspace(1, 0, fade_s)
    wave = wave * envelope
    # append silence so delay tail can ring out fully
    tail_samples = int(RATE * tail)
    full = np.concatenate([wave, np.zeros(tail_samples)])
    # apply delay across full length so tail blooms naturally
    full = apply_delay(full, delay_time, delay_feedback, delay_mix)
    peak = np.max(np.abs(full))
    if peak > 0:
        full = full / peak * volume
    stereo = np.column_stack([full, full])
    return stereo.astype(np.float32).tobytes()

def get_fx_params(a, d, t, b):
    cutoff = max(0.1, min(1.0, 0.2 + (b * 0.6) + (a * 0.2) - (d * 0.1)))
    attack = max(0.05, min(0.8, 0.5 - (b * 0.4) + (d * 0.3)))
    delay_time = max(0.05, min(0.8, 0.1 + (t * 0.5) + (d * 0.3) - (b * 0.1)))
    delay_feedback = max(0.05, min(0.6, 0.1 + (a * 0.4) - (b * 0.1)))
    delay_mix = max(0.05, min(0.5, 0.1 + (t * 0.3) + (d * 0.2)))
    return cutoff, attack, delay_time, delay_feedback, delay_mix

p = pyaudio.PyAudio()
chord_stream  = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE, output=True, output_device_index=DEVICE)
melody_stream = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE, output=True, output_device_index=DEVICE)
run_stream    = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE, output=True, output_device_index=DEVICE)

def chord_thread():
    global current_chord, current_chord_idx, last_chord_time, iteration
    while True:
        now = time.time()
        a, d, t, b, a_raw, d_raw, t_raw, b_raw = compute_all()
        score = compute_score(a, d, t, b)
        accumulated_score.append(score)
        avg_score = np.mean(accumulated_score)
        variance = np.std(accumulated_score) if len(accumulated_score) > 3 else 0
        change_interval = max(4, min(8, 8 - (variance * 50)))
        chord_changed = False

        if now - last_chord_time >= change_interval:
            idx = int(score * (len(C_MAJOR_CHORDS) - 1))
            idx = max(0, min(idx, len(C_MAJOR_CHORDS) - 1))
            new_chord = C_MAJOR_CHORDS[idx]
            chord_changed = new_chord != current_chord
            current_chord = new_chord
            current_chord_idx = idx
            last_chord_time = now

        cutoff, attack, delay_time, delay_feedback, delay_mix = get_fx_params(a, d, t, b)

        with lock:
            iteration += 1
            time_left = max(0, change_interval - (now - last_chord_time))
            print(f"\n{'='*52}")
            print(f"  iter {iteration:04d}  |  state: {STATE_LABELS[current_chord_idx]}")
            print(f"{'='*52}")
            print(f"  delta  {bar(d)} {d:.2f}  ({d_raw:>10,.0f})")
            print(f"  theta  {bar(t)} {t:.2f}  ({t_raw:>10,.0f})")
            print(f"  alpha  {bar(a)} {a:.2f}  ({a_raw:>10,.0f})")
            print(f"  beta   {bar(b)} {b:.2f}  ({b_raw:>10,.0f})")
            print(f"  score  {bar(score)} {score:.2f}  var: {variance:.3f}")
            print(f"  chord  {CHORD_NAMES[current_chord_idx]} {current_chord}  next: {time_left:.1f}s{'  ← NEW' if chord_changed else ''}")
            print(f"  fx     cutoff:{cutoff:.2f}  attack:{attack:.2f}s  delay:{delay_time:.2f}s  fb:{delay_feedback:.2f}  mix:{delay_mix:.2f}")

        freqs = [midi_to_freq(n) for n in current_chord]
        chord_stream.write(make_wave(freqs, 5.0, volume=0.25, brightness=avg_score,
                                     cutoff=cutoff, attack=attack,
                                     delay_time=delay_time, delay_feedback=delay_feedback,
                                     delay_mix=delay_mix))

def melody_thread():
    time.sleep(2.5)
    while True:
        a, d, t, b, *_ = compute_all()
        n_notes = 2 if b > 0.65 else 1
        note_dur = 5.0 / n_notes
        cutoff, attack, delay_time, delay_feedback, delay_mix = get_fx_params(a, d, t, b)
        phrase = [pick_melody_note() for _ in range(n_notes)]
        note_str = ' → '.join(NOTE_NAMES.get(n, str(n)) for n in phrase)
        with lock:
            print(f"  melody {note_str}")
        for note in phrase:
            freq = midi_to_freq(note)
            melody_stream.write(make_wave([freq], note_dur, volume=0.35,
                                          brightness=a, cutoff=cutoff,
                                          attack=attack, delay_time=delay_time,
                                          delay_feedback=delay_feedback,
                                          delay_mix=delay_mix))

def run_thread():
    """Slow gliding high notes through chord extensions."""
    time.sleep(1.25)
    prev_note = None
    while True:
        compute_all()
        b = ema['beta']
        t = ema['theta']
        a = ema['alpha']

        trigger_prob = (b * 0.4) + (t * 0.3) + 0.7
        if np.random.random() < trigger_prob:
            extensions = CHORD_EXTENSIONS[current_chord_idx]
            # pick target note from extensions, biased by alpha
            idx = int(a * (len(extensions) - 1))
            idx = max(0, min(idx + np.random.randint(-1, 2), len(extensions) - 1))
            target_note = extensions[idx]

            # glide from previous note or a neighboring extension
            if prev_note is None:
                prev_note = extensions[0]
            start_note = prev_note

            # glide duration: slower in theta/delta, faster in beta
            glide_dur = max(1.5, min(5.0, 3.0 + (t * 1.5) - (b * 1.0)))

            start_freq = midi_to_freq(start_note)
            end_freq   = midi_to_freq(target_note)

            start_name = NOTE_NAMES.get(start_note, str(start_note))
            end_name   = NOTE_NAMES.get(target_note, str(target_note))
            with lock:
                print(f"  glide  {start_name} → {end_name}  ({glide_dur:.1f}s)")

            # volume: whisper quiet, alpha brightens slightly
            vol = 0.06 + (a * 0.06)
            run_stream.write(make_glide_wave(start_freq, end_freq, glide_dur, volume=vol,
                                             brightness=a * 0.5,
                                             cutoff=0.2,
                                             attack=0.6,
                                             delay_time=0.45,
                                             delay_feedback=0.75,
                                             delay_mix=0.85))
            prev_note = target_note

        wait = max(0.5, 2.0 - (b * 1.0) - (t * 0.5))
        time.sleep(wait)

print("Starting brain music - calibrating...")
try:
    t1 = threading.Thread(target=chord_thread, daemon=True)
    t2 = threading.Thread(target=melody_thread, daemon=True)
    t3 = threading.Thread(target=run_thread, daemon=True)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    chord_stream.stop_stream()
    chord_stream.close()
    melody_stream.stop_stream()
    melody_stream.close()
    run_stream.stop_stream()
    run_stream.close()
    p.terminate()
