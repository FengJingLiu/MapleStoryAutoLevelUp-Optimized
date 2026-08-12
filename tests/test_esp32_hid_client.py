import unittest
from collections import deque

from src.input.Esp32HidClient import (
    Esp32HidClient,
    Esp32HidTapUncertainError,
    usage_from_text,
)


class FakeSocket:
    """Small line-oriented ESP32 protocol double used by the client tests."""

    def __init__(self, status_response=None, fail_on_send=None):
        self.status_response = status_response or (
            "OK WIFI=1 BLE_CONNECTED=1 BLE_READY=1"
        )
        self.commands = []
        self.connected_to = None
        self.timeouts = []
        self.closed = False
        self.shutdown_calls = []
        self._responses = deque()
        self._fail_on_send = set(fail_on_send or ())

    def connect(self, address):
        self.connected_to = address

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def sendall(self, payload):
        if self.closed:
            raise OSError("socket is closed")
        self.assert_ascii_line(payload)
        command = payload.decode("ascii").rstrip("\r\n")
        self.commands.append(command)
        if command in self._fail_on_send:
            self._fail_on_send.remove(command)
            raise ConnectionResetError(f"simulated disconnect during {command}")
        self._responses.append((self.response_for(command) + "\n").encode("ascii"))

    @staticmethod
    def assert_ascii_line(payload):
        if not isinstance(payload, bytes):
            raise AssertionError(f"expected bytes, got {type(payload)!r}")
        if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
            raise AssertionError(f"expected one newline-terminated command: {payload!r}")
        payload.decode("ascii")

    def response_for(self, command):
        if command == "STATUS":
            return self.status_response
        if command == "PING":
            return "PONG"
        if command == "RELEASE_ALL":
            return "OK RELEASE_ALL"
        if command.startswith("DOWN "):
            return f"OK DOWN {command.split(maxsplit=1)[1]}"
        if command.startswith("UP "):
            return f"OK UP {command.split(maxsplit=1)[1]}"
        if command.startswith("TAP "):
            _, usage, duration = command.split()
            return f"OK TAP {usage} {duration}ms"
        if command == "STATE" or command.startswith("STATE "):
            return "OK STATE"
        raise AssertionError(f"unexpected command: {command!r}")

    def recv(self, size):
        if not self._responses:
            raise AssertionError("client attempted to read without sending a command")
        response = self._responses.popleft()
        if len(response) <= size:
            return response
        self._responses.appendleft(response[size:])
        return response[:size]

    def shutdown(self, how):
        self.shutdown_calls.append(how)

    def close(self):
        self.closed = True


class FakeSocketFactory:
    """Accept both create_connection-style and socket-constructor-style calls."""

    def __init__(self, fake_socket):
        self.sockets = (
            list(fake_socket)
            if isinstance(fake_socket, (list, tuple))
            else [fake_socket]
        )
        self.socket = self.sockets[0]
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        socket_index = min(len(self.calls) - 1, len(self.sockets) - 1)
        selected_socket = self.sockets[socket_index]
        if args and isinstance(args[0], tuple):
            selected_socket.connected_to = args[0]
        return selected_socket


class Esp32HidClientTests(unittest.TestCase):
    def make_client(self, status_response=None, **kwargs):
        fake_socket = FakeSocket(status_response=status_response)
        factory = FakeSocketFactory(fake_socket)
        client = Esp32HidClient(
            "esp32.test",
            socket_factory=factory,
            heartbeat_interval=0,
            **kwargs,
        )
        return client, fake_socket, factory

    def test_default_key_names_map_to_usb_hid_usages(self):
        expected = {
            "a": 0x04,
            "e": 0x08,
            "p": 0x13,
            "q": 0x14,
            "w": 0x1A,
            "z": 0x1D,
            "1": 0x1E,
            "0": 0x27,
            "space": 0x2C,
            "home": 0x4A,
            "left": 0x50,
            "down": 0x51,
            "up": 0x52,
            "ctrl": 0xE0,
            "shift": 0xE1,
            "alt": 0xE2,
        }
        for key, usage in expected.items():
            with self.subTest(key=key):
                self.assertEqual(usage_from_text(key), usage)
                self.assertEqual(usage_from_text(key.upper()), usage)

        self.assertEqual(usage_from_text("0x04"), 0x04)
        with self.assertRaises(ValueError):
            usage_from_text("not-a-key")

    def test_symbol_and_common_aliases_map_to_expected_usages(self):
        symbols = {
            "-": 0x2D,
            "=": 0x2E,
            "[": 0x2F,
            "]": 0x30,
            "\\": 0x31,
            ";": 0x33,
            "'": 0x34,
            "`": 0x35,
            ",": 0x36,
            ".": 0x37,
            "/": 0x38,
        }
        aliases = {
            "return": 0x28,
            "escape": 0x29,
            "equals": 0x2E,
            "left bracket": 0x2F,
            "right-bracket": 0x30,
            "quote": 0x34,
            "backtick": 0x35,
            "period": 0x37,
            "prtsc": 0x46,
            "ins": 0x49,
            "pgup": 0x4B,
            "del": 0x4C,
            "pgdn": 0x4E,
            "control": 0xE0,
            "left_control": 0xE0,
            "option": 0xE2,
            "windows": 0xE3,
            "cmd": 0xE3,
        }
        for key, usage in {**symbols, **aliases}.items():
            with self.subTest(key=key):
                self.assertEqual(usage_from_text(key), usage)

    def test_connect_uses_default_port_then_checks_status_and_releases_keys(self):
        client, fake_socket, factory = self.make_client(
            connect_timeout=1.25,
            request_timeout=2.5,
            reconnect_interval=0.01,
        )
        self.addCleanup(client.close)

        self.assertEqual(fake_socket.connected_to, ("esp32.test", 3333))
        self.assertTrue(factory.calls)
        self.assertEqual(fake_socket.commands, ["STATUS", "RELEASE_ALL"])
        self.assertIn(2.5, fake_socket.timeouts)

    def test_key_down_and_up_are_deduplicated(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()

        client.key_down("left")
        client.key_down("LEFT")
        client.key_up("left")
        client.key_up("LEFT")

        self.assertEqual(fake_socket.commands, ["DOWN 0x50", "UP 0x50"])

    def test_safe_down_reconnects_then_replays_after_a_socket_drop(self):
        first_socket = FakeSocket(fail_on_send={"DOWN 0x50"})
        second_socket = FakeSocket()
        factory = FakeSocketFactory([first_socket, second_socket])
        client = Esp32HidClient(
            "esp32.test",
            socket_factory=factory,
            heartbeat_interval=0,
            reconnect_interval=0,
        )
        self.addCleanup(client.close)

        self.assertTrue(client.key_down("left"))

        self.assertEqual(
            first_socket.commands,
            ["STATUS", "RELEASE_ALL", "DOWN 0x50"],
        )
        self.assertEqual(
            second_socket.commands,
            ["STATUS", "RELEASE_ALL", "DOWN 0x50"],
        )
        self.assertEqual(len(factory.calls), 2)
        self.assertTrue(first_socket.closed)

    def test_tap_uses_requested_duration_without_changing_held_state(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()

        client.key_down("left")
        client.tap("space", 75)
        client.key_down("left")

        self.assertEqual(
            fake_socket.commands,
            ["DOWN 0x50", "TAP 0x2C 75",],
        )

    def test_tap_rejects_a_key_that_is_already_held(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()
        client.set_state(["up"])

        with self.assertRaisesRegex(RuntimeError, "already held"):
            client.tap("up")

        self.assertEqual(fake_socket.commands, ["STATE 0x52"])

    def test_failed_tap_is_not_replayed_when_the_next_request_reconnects(self):
        first_socket = FakeSocket(fail_on_send={"TAP 0x2C 75"})
        second_socket = FakeSocket()
        factory = FakeSocketFactory([first_socket, second_socket])
        client = Esp32HidClient(
            "esp32.test",
            socket_factory=factory,
            heartbeat_interval=0,
            reconnect_interval=0,
        )
        self.addCleanup(client.close)

        with self.assertRaises(Esp32HidTapUncertainError):
            client.tap("space", 75)

        self.assertEqual(len(factory.calls), 1)
        self.assertEqual(
            first_socket.commands,
            ["STATUS", "RELEASE_ALL", "TAP 0x2C 75"],
        )

        self.assertIn("BLE_READY=1", client.status())
        self.assertEqual(len(factory.calls), 2)
        self.assertEqual(
            second_socket.commands,
            ["STATUS", "RELEASE_ALL", "STATUS"],
        )
        self.assertNotIn("TAP 0x2C 75", second_socket.commands)

    def test_set_state_deduplicates_keys_and_unchanged_reports(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()

        client.set_state(["left", "up", "LEFT"])
        client.set_state(["LEFT", "UP"])
        client.set_state(["right"])

        self.assertEqual(
            fake_socket.commands,
            ["STATE 0x50 0x52", "STATE 0x4F"],
        )

    def test_held_state_is_periodically_reasserted_after_unobserved_ble_drop(self):
        client, fake_socket, _ = self.make_client(state_refresh_interval=1.0)
        self.addCleanup(client.close)
        fake_socket.commands.clear()

        client.set_state(["left"])
        client._last_state_write -= 2.0
        client.set_state(["left"])

        self.assertEqual(fake_socket.commands, ["STATE 0x50", "STATE 0x50"])

    def test_status_sends_a_fresh_request(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()

        status = client.status()

        self.assertEqual(fake_socket.commands, ["STATUS"])
        self.assertIn("BLE_READY=1", str(status))

    def test_ble_drop_invalidates_held_state_without_a_tcp_disconnect(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()
        client.key_down("left")

        fake_socket.status_response = (
            "OK WIFI=1 BLE_CONNECTED=0 BLE_READY=0"
        )
        client.status()
        fake_socket.status_response = (
            "OK WIFI=1 BLE_CONNECTED=1 BLE_READY=1"
        )
        client.set_state(["left"])

        self.assertEqual(
            fake_socket.commands,
            ["DOWN 0x50", "STATUS", "STATE 0x50"],
        )

    def test_ble_not_ready_command_error_invalidates_held_state(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()
        client.set_state(["left"])

        original_response_for = fake_socket.response_for

        def response_for(command):
            if command.startswith("TAP "):
                return "ERR BLE_NOT_READY"
            return original_response_for(command)

        fake_socket.response_for = response_for
        with self.assertRaisesRegex(Esp32HidTapUncertainError, "BLE_NOT_READY"):
            client.tap("a")

        fake_socket.response_for = original_response_for
        client.set_state(["left"])
        self.assertEqual(
            fake_socket.commands,
            ["STATE 0x50", "TAP 0x04 50", "STATE 0x50"],
        )

    def test_hid_send_failure_invalidates_held_state(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()
        client.set_state(["left"])

        original_response_for = fake_socket.response_for

        def response_for(command):
            if command.startswith("TAP "):
                return "ERR HID_SEND_FAILED"
            return original_response_for(command)

        fake_socket.response_for = response_for
        with self.assertRaisesRegex(Esp32HidTapUncertainError, "HID_SEND_FAILED"):
            client.tap("a")

        fake_socket.response_for = original_response_for
        client.set_state(["left"])
        self.assertEqual(
            fake_socket.commands,
            ["STATE 0x50", "TAP 0x04 50", "STATE 0x50"],
        )

    def test_failed_tap_forces_an_empty_release_state(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()

        original_response_for = fake_socket.response_for

        def response_for(command):
            if command.startswith("TAP "):
                return "ERR HID_SEND_FAILED"
            return original_response_for(command)

        fake_socket.response_for = response_for
        with self.assertRaises(Esp32HidTapUncertainError):
            client.tap("a")

        fake_socket.response_for = original_response_for
        client.set_state([])
        self.assertEqual(fake_socket.commands, ["TAP 0x04 50", "STATE"])

    def test_unexpected_tap_ack_is_treated_as_uncertain(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()
        original_response_for = fake_socket.response_for

        def response_for(command):
            if command.startswith("TAP "):
                return "OK TAP 0x04 999ms"
            return original_response_for(command)

        fake_socket.response_for = response_for
        with self.assertRaisesRegex(
            Esp32HidTapUncertainError, "unexpected TAP response"
        ):
            client.tap("a", 50)

        self.assertEqual(fake_socket.commands, ["TAP 0x04 50"])

    def test_oversized_tap_response_is_uncertain_and_disconnects(self):
        client, fake_socket, _ = self.make_client()
        self.addCleanup(client.close)
        fake_socket.commands.clear()
        original_response_for = fake_socket.response_for

        def response_for(command):
            if command.startswith("TAP "):
                return "X" * 5000
            return original_response_for(command)

        fake_socket.response_for = response_for
        with self.assertRaisesRegex(
            Esp32HidTapUncertainError, "response is too long"
        ):
            client.tap("a", 50)

        self.assertTrue(fake_socket.closed)

    def test_close_releases_held_keys_and_is_idempotent(self):
        client, fake_socket, _ = self.make_client()
        fake_socket.commands.clear()
        client.key_down("a")

        client.close()
        commands_after_first_close = list(fake_socket.commands)
        client.close()

        self.assertEqual(
            commands_after_first_close,
            ["DOWN 0x04", "RELEASE_ALL"],
        )
        self.assertEqual(fake_socket.commands, commands_after_first_close)
        self.assertTrue(fake_socket.closed)

    def test_constructor_rejects_a_ble_link_that_is_not_ready(self):
        fake_socket = FakeSocket(
            "OK WIFI=1 BLE_CONNECTED=1 BLE_READY=0"
        )
        factory = FakeSocketFactory(fake_socket)

        with self.assertRaisesRegex(
            (ConnectionError, RuntimeError), r"BLE|BLE_READY|ready"
        ):
            Esp32HidClient(
                "esp32.test",
                socket_factory=factory,
                heartbeat_interval=0,
                reconnect_interval=0.01,
            )

        self.assertEqual(fake_socket.commands[0], "STATUS")
        self.assertNotIn("DOWN", " ".join(fake_socket.commands))


if __name__ == "__main__":
    unittest.main()
