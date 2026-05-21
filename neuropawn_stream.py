import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import sys
sys.path.append('/Users/Dylan/Desktop/eegsynth-master/src/lib')
import FieldTrip

SERIAL_PORT = '/dev/cu.usbserial-A5069RR4'
NUM_CHANNELS = 4
FT_HOST = 'localhost'
FT_PORT = 1972

params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT
params.timeout = 15
params.other_info = '{"gain": 12}'

board = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, params)
board_id = board.get_board_id()
eeg_channels = BoardShim.get_exg_channels(board_id)[:NUM_CHANNELS]
sample_rate = BoardShim.get_sampling_rate(board_id)
print(f"Sample rate: {sample_rate}, EEG channels: {eeg_channels}")

ft = FieldTrip.Client()
ft.connect(FT_HOST, FT_PORT)
ft.putHeader(NUM_CHANNELS, sample_rate, FieldTrip.DATATYPE_FLOAT32)

board.prepare_session()
board.start_stream(450000)
time.sleep(3)

for ch in range(1, NUM_CHANNELS + 1):
    time.sleep(1)
    board.config_board(f"chon_{ch}_12")
    print(f"Enabled channel {ch}")
    time.sleep(2)
    board.config_board(f"rldadd_{ch}")
    print(f"RLD added channel {ch}")
    time.sleep(1)

print("Streaming... Ctrl-C to stop")
try:
    while True:
        data = board.get_board_data()
        if data.shape[1] > 0:
            chunk = data[eeg_channels, :].T.astype(np.float32)
            ft.putData(chunk)
        time.sleep(0.02)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    board.stop_stream()
    board.release_session()
    ft.disconnect()
    print("Done.")
