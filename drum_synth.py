import redis
import time
import pyaudio
import numpy as np
from collections import deque

r = redis.Redis()
RATE = 44100
DEVICE = None

raw = {'alpha': deque(maxlen=30), 'delta': deque(maxlen=30),
       'theta': deque(maxlen=30), 'beta':  deque(maxlen=30)}
ema = {'alpha': 0.5, 'delta': 0.5, 'theta': 0.5, 'beta': 0.5}
EMA_ALPHA = 0.25

def read_band(key, name):
    val = r.get(key)
    if not val:
        return ema[name]
    v = float(val)
    raw[name].append(v)
    lo = min(raw[name])
    hi = max(raw[name])
    norm = (v - lo) / (hi - lo) if hi != lo else 0.5
    ema[name] = EMA_ALPHA * norm + (1 - EMA_ALPHA) * ema[name]
    return ema[name]

def apply_reverb(wave, room=0.92, damp=0.3, mix=0.85):
    """Schroeder reverb — 4 comb filters + 2 allpass for lush room."""
    n = len(wave)
    comb_delays = [int(RATE * d) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
    allpass_delays = [int(RATE * d) for d in [0.005, 0.0017]]
    out = np.zeros(n)
    # comb filters
    for delay in comb_delays:
        buf = np.zeros(delay)
        idx = 0
        comb_out = np.zeros(n)
        for i in range(n):
            buf_out = buf[idx]
            buf[idx] = wave[i] + buf_out * room * (1 - damp)
            comb_out[i] = buf_out
            idx = (idx + 1) % delay
        out += comb_out
    out /= len(comb_delays)
    # allpass filters
    for delay in allpass_delays:
        buf = np.zeros(delay)
        idx = 0
        ap_out = np.zeros(n)
        for i in range(n):
            buf_out = buf[idx]
            ap_out[i] = -out[i] + buf_out
            buf[idx] = out[i] + buf_out * 0.5
            idx = (idx + 1) % delay
        out = ap_out
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * np.max(np.abs(wave))
    return wave * (1 - mix) + out * mix

def apply_echo(wave, delay1=0.35, delay2=0.65, delay3=1.1,
               fb1=0.6, fb2=0.4, fb3=0.2, mix=0.7):
    out = wave.copy()
    d1 = int(delay1 * RATE)
    d2 = int(delay2 * RATE)
    d3 = int(delay3 * RATE)
    for i in range(len(out)):
        if i >= d1: out[i] += fb1 * out[i - d1]
        if i >= d2: out[i] += fb2 * out[i - d2]
        if i >= d3: out[i] += fb3 * out[i - d3]
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * np.max(np.abs(wave))
    return wave * (1 - mix) + out * mix

def apply_shimmer(wave, pitch_ratio=2.01, mix=0.3):
    n = len(wave)
    indices = np.clip((np.arange(n) * pitch_ratio).astype(int), 0, n - 1)
    shifted = wave[indices]
    env = np.linspace(0, 1, n) * np.linspace(1, 0, n)
    return wave + shifted * env * mix

def process_fx(wave, echo_mix, shimmer_mix, room, damp, reverb_mix):
    wave = apply_reverb(wave, room=room, damp=damp, mix=reverb_mix)
    wave = apply_shimmer(wave, mix=shimmer_mix)
    wave = apply_echo(wave, mix=echo_mix)
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    return wave

def make_808(volume=0.7, pitch=55, decay=1.2, distort=0.0,
             echo_mix=0.7, shimmer_mix=0.3, room=0.92, damp=0.3, reverb_mix=0.85):
    n = int(RATE * decay)
    t = np.linspace(0, decay, n)
    punch_freq = pitch * 3.5
    freq = punch_freq * np.exp(-t * 6) + pitch * (1 - np.exp(-t * 6))
    wave = np.sin(2 * np.pi * np.cumsum(freq) / RATE)
    env = np.exp(-t * 40) * 0.4 + np.exp(-t * (1.5 / decay)) * 0.6
    wave = wave * env
    if distort > 0:
        wave = np.tanh(wave * (1 + distort * 4)) / (1 + distort)
    wave = process_fx(wave, echo_mix, shimmer_mix, room, damp, reverb_mix)
    wave = wave * volume
    stereo = np.column_stack([wave, wave])
    return stereo.astype(np.float32).tobytes()

def make_heartbeat(volume=0.6, pitch=50, tempo=60, distort=0.0,
                   echo_mix=0.75, shimmer_mix=0.35, room=0.92, damp=0.3, reverb_mix=0.85):
    beat_dur = 60.0 / tempo
    lub_dur  = 0.12
    gap_dur  = 0.08
    dub_dur  = 0.18
    rest_dur = max(0.05, beat_dur - lub_dur - gap_dur - dub_dur)

    def thud(dur, vol, p):
        n = int(RATE * dur)
        t = np.linspace(0, dur, n)
        freq = p * 2.5 * np.exp(-t * 15) + p * (1 - np.exp(-t * 15))
        wave = np.sin(2 * np.pi * np.cumsum(freq) / RATE)
        env  = np.exp(-t * (4 / dur))
        wave = wave * env * vol
        if distort > 0:
            wave = np.tanh(wave * (1 + distort * 3)) / (1 + distort)
        return wave

    lub  = thud(lub_dur, volume, pitch * 1.1)
    gap  = np.zeros(int(RATE * gap_dur))
    dub  = thud(dub_dur, volume * 0.75, pitch)
    rest = np.zeros(int(RATE * rest_dur))
    wave = np.concatenate([lub, gap, dub, rest])
    wave = process_fx(wave, echo_mix, shimmer_mix, room, damp, reverb_mix)
    wave = wave * volume
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * volume
    stereo = np.column_stack([wave, wave])
    return stereo.astype(np.float32).tobytes()

def make_silence(dur):
    n = int(RATE * dur)
    return np.zeros((n, 2), dtype=np.float32).tobytes()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE,
                output=True, output_device_index=DEVICE)

print("Starting ethereal 808 heartbeat...")

iteration = 0
try:
    while True:
        a = read_band('spectral.channel1.alpha', 'alpha')
        d = read_band('spectral.channel1.delta', 'delta')
        t = read_band('spectral.channel1.theta', 'theta')
        b = read_band('spectral.channel1.beta',  'beta')

        iteration += 1

        bpm         = max(28, min(72, 28 + (b * 24) + (a * 12) - (d * 8)))
        pitch       = max(28, min(60, 36 + (a * 18) - (d * 8)))
        decay       = max(0.5, min(2.5, 0.8 + (t * 1.0) + (d * 0.5)))
        distort     = b * 0.4
        volume      = max(0.15, min(0.65, 0.3 + (a * 0.3) - (d * 0.1)))
        echo_mix    = max(0.5, min(0.92, 0.6 + (t * 0.2) + (d * 0.15)))
        shimmer_mix = max(0.1, min(0.5, 0.15 + (a * 0.35)))
        # reverb gets massive in delta, tighter in beta
        room        = max(0.7, min(0.98, 0.85 + (d * 0.1) + (t * 0.05) - (b * 0.1)))
        damp        = max(0.1, min(0.7, 0.4 - (t * 0.2) - (d * 0.1) + (b * 0.2)))
        reverb_mix  = max(0.6, min(0.95, 0.75 + (d * 0.15) + (t * 0.05)))

        if t > 0.5:
            mode = 'heartbeat'
            stream.write(make_heartbeat(volume=volume, pitch=pitch, tempo=bpm,
                                        distort=distort, echo_mix=echo_mix,
                                        shimmer_mix=shimmer_mix, room=room,
                                        damp=damp, reverb_mix=reverb_mix))
        else:
            mode = '808     '
            stream.write(make_808(volume=volume, pitch=pitch, decay=decay,
                                  distort=distort, echo_mix=echo_mix,
                                  shimmer_mix=shimmer_mix, room=room,
                                  damp=damp, reverb_mix=reverb_mix))
            beat_dur = 60.0 / bpm
            stream.write(make_silence(max(0.05, beat_dur - decay)))

        print(f"  iter {iteration:04d}  {mode}  bpm:{bpm:.1f}  room:{room:.2f}  damp:{damp:.2f}  reverb:{reverb_mix:.2f}  echo:{echo_mix:.2f}")

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
