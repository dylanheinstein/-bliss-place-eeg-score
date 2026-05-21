# Bliss Place — EEG-Driven Interactive Score

Real-time brainwave-controlled music system for the film *Bliss Place*, directed by Gideon Buddenhagen and Noah Zielinski.

## Concept

*Bliss Place* is a psychological science fiction thriller in which an office space transports a character between parallel life outcomes depending on what they prioritize — work or relationship. The EEG score reads the viewer\'s brainwave patterns in real time as they watch the film, generating a living score that reflects their neurological state.

## Hardware

- NeuroPawn Knight Board — 4-channel EEG
- Sample rate: 125 Hz
- Gain: 12

## Signal Chain

NeuroPawn -> BrainFlow -> FieldTrip Buffer -> EEGsynth Spectral -> Redis -> Brain Music Engine

## Brain to Music Mapping

- Delta: 808 tempo, low pitch, heavy reverb
- Theta: heartbeat double-thud, long delay, dreamy glides
- Alpha: chord brightness, harmonic overtones, glide height
- Beta: filter cutoff, melodic density, run speed

## Scripts

- brain_music.py — main synthesis engine
- drum_synth.py — 808/heartbeat pulse engine
- neuropawn_stream.py — BrainFlow to FieldTrip buffer stream

## Startup

Tab 1: cd ~/Desktop/eegsynth-master/src/module/buffer && python3 buffer.py -i ~/Desktop/eegsynth-master/patches/myfirstpatch/buffer.ini
Tab 2: python3 ~/Desktop/neuropawn_stream.py
Tab 3: cd ~/Desktop/eegsynth-master/src/module/spectral && python3 spectral.py -i ~/Desktop/eegsynth-master/patches/myfirstpatch/spectral.ini
Tab 4: python3 ~/Desktop/brain_music.py
Tab 5: cd ~/Desktop/eegsynth-master/src/module/plotsignal && python3 plotsignal.py -i ~/Desktop/eegsynth-master/patches/myfirstpatch/plotsignal.ini

## Film

Bliss Place — dir. Gideon Buddenhagen and Noah Zielinski
Psychological science fiction thriller
