import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src" / "realtime_mapping_scripts"
sys.path.insert(0, str(SCRIPTS))

from receive_scan import HEADER, PACKET_SIZE, POINT, POINTS_PER_PACKET, decode_packet  # noqa: E402


class M360TimingTest(unittest.TestCase):
    def test_timestamp_is_little_endian_unsigned_32_32(self):
        seconds = 9870
        fraction = 0.25
        low = round(fraction * 2**32)
        raw = bytearray(PACKET_SIZE)
        HEADER.pack_into(raw, 0, 0x68, 7, 25, 115, low, seconds)
        for index in range(POINTS_PER_PACKET):
            POINT.pack_into(raw, HEADER.size + index * POINT.size, 1000, 1)
        packet = decode_packet(bytes(raw), host_time=123.0)
        self.assertAlmostEqual(packet.timestamp, seconds + fraction, places=9)

    def test_manual_point_interval_at_25_hz(self):
        interval = 1.0 / (25 * 2300)
        self.assertAlmostEqual(interval, 17.391304347826086e-6, places=15)
        self.assertAlmostEqual(114 * interval, 0.001982608695652174, places=15)


if __name__ == "__main__":
    unittest.main()
