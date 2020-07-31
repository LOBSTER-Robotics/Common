import unittest

from lobster_common.time_units import Seconds, Milliseconds, Microseconds, Nanoseconds


class TimeUnitTest(unittest.TestCase):

    def test_seconds(self):

        seconds = 42

        time = Seconds(seconds)

        self.assertEqual(time.seconds, seconds)
        self.assertEqual(time.milliseconds, seconds * 1e3)
        self.assertEqual(time.microseconds, seconds * 1e6)
        self.assertEqual(time.nanoseconds, seconds * 1e9)

    def test_milliseconds(self):

        milliseconds = 42

        time = Milliseconds(milliseconds)

        self.assertEqual(time.seconds, milliseconds / 1e3)
        self.assertEqual(time.milliseconds, milliseconds)
        self.assertEqual(time.microseconds, milliseconds * 1e3)
        self.assertEqual(time.nanoseconds, milliseconds * 1e6)

    def test_microseconds(self):

        microseconds = 42

        time = Microseconds(microseconds)

        self.assertEqual(time.seconds, microseconds / 1e6)
        self.assertEqual(time.milliseconds, microseconds / 1e3)
        self.assertEqual(time.microseconds, microseconds)
        self.assertEqual(time.nanoseconds, microseconds * 1e3)

    def test_nanoseconds(self):
        nanoseconds = 42

        time = Nanoseconds(nanoseconds)

        self.assertEqual(time.seconds, nanoseconds / 1e9)
        self.assertEqual(time.milliseconds, nanoseconds / 1e6)
        self.assertEqual(time.microseconds, nanoseconds / 1e3)
        self.assertEqual(time.nanoseconds, nanoseconds)

    def test_addition(self):
        self.assertEqual((Seconds(30) + Milliseconds(500)).seconds, 30.5)

    def test_non_time_addition(self):
        with self.assertRaises(TypeError):
            Seconds(3) + 5
