from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kis import dres
from kis.dres_api import DRESConnectRequest, create_dres_router


RUN = "11111111-2222-4333-8444-555555555555"
SESSION = "synthetic-test-session-1234567890"
SELECTION = {"video_id": "L01_V001", "image_number": 2, "frame_id": 150}
SNAPSHOT = {"evaluation_id": RUN, "name": "Test run", "evaluation_status": "ACTIVE",
            "task_id": "task-run-one", "task_status": "RUNNING", "time_left": 10000,
            "can_submit": True, "task": {"name": "KIS test", "taskType": "KIS"}}


class FakeReply(BytesIO):
    def __init__(self, data, status=200):
        super().__init__(json.dumps(data).encode())
        self.status = status


class FakeOpener:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class DRESTransportTest(unittest.TestCase):
    def test_auth_is_cookie_not_url_and_no_secret_returned(self):
        opener = FakeOpener(FakeReply({"username": "tester", "role": "PARTICIPANT", "sessionId": SESSION}))
        client = dres.DRESClient("https://example.com/dres", SESSION, opener)
        user = client.user()
        request, timeout = opener.calls[0]
        self.assertEqual(request.full_url, "https://example.com/dres/api/v2/user")
        self.assertEqual(request.get_header("Cookie"), "SESSIONID=" + SESSION)
        self.assertEqual(timeout, 15)
        self.assertNotIn(SESSION, str(user))

    def test_login_secret_never_returned_or_repr(self):
        opener = FakeOpener(FakeReply({"username": "tester", "role": "PARTICIPANT", "sessionId": SESSION}))
        client = dres.DRESClient("https://example.com", opener=opener)
        user = client.login("tester", "secret-password")
        self.assertEqual(json.loads(opener.calls[0][0].data)["password"], "secret-password")
        self.assertNotIn(SESSION, str(user))
        self.assertNotIn("secret-password", str(user))
        self.assertNotIn("secret-password", repr(DRESConnectRequest(evaluation_id=RUN, username="tester", password="secret-password")))

    def test_error_body_url_and_session_are_not_echoed(self):
        error = HTTPError("https://example.com/?session=" + SESSION, 401, "Unauthorized", {}, BytesIO(SESSION.encode()))
        opener = FakeOpener(error)
        with self.assertRaises(dres.DRESError) as context:
            dres.DRESClient("https://example.com", SESSION, opener).user()
        self.assertEqual(context.exception.status_code, 401)
        self.assertNotIn(SESSION, str(context.exception))

    def test_submit_2xx_with_non_json_body_counts_as_received_and_errors_are_logged(self):
        reply = FakeReply({}, status=202); reply.seek(0); reply.truncate(); reply.write(b"<html>ok</html>"); reply.seek(0)
        client = dres.DRESClient("https://example.com", SESSION, FakeOpener(reply))
        self.assertEqual(client.submit(RUN, {"answerSets": []}), {"http_status": 202, "accepted": True, "verdict": None, "pending": True})
        error = HTTPError("https://example.com/", 503, "Busy", {}, BytesIO(b"upstream busy " + SESSION.encode()))
        with self.assertLogs("kis.dres", "WARNING") as logs, self.assertRaises(dres.DRESError):
            dres.DRESClient("https://example.com", SESSION, FakeOpener(error)).submit(RUN, {"answerSets": []})
        self.assertIn("HTTP 503 upstream busy", logs.output[0])
        self.assertNotIn(SESSION, logs.output[0])

    def test_explicit_dres_refusal_is_definite_and_shows_reason(self):
        body = json.dumps({"status": False, "description": "Could not find media item with name 'L15_V008'. Maybe collection name is required."}).encode()
        error = HTTPError("https://example.com/", 500, "Server Error", {}, BytesIO(body))
        with self.assertRaises(dres.DRESError) as context:
            dres.DRESClient("https://example.com", SESSION, FakeOpener(error)).submit(RUN, {"answerSets": []})
        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("L15_V008", str(context.exception))
        self.assertIn("chưa ghi nhận", str(context.exception))

    def test_snapshot_uses_visible_run_state_and_current_task(self):
        opener = FakeOpener(FakeReply([{"id": RUN, "name": "Test"}]),
            FakeReply({"evaluationStatus": "ACTIVE", "taskStatus": "RUNNING", "taskId": "task-one"}),
            FakeReply({"name": "KIS"}))
        snapshot = dres.DRESClient("https://example.com", SESSION, opener).snapshot(RUN)
        self.assertTrue(snapshot["can_submit"])
        self.assertEqual(snapshot["task_id"], "task-one")
        self.assertTrue(all(call[0].method == "GET" for call in opener.calls))
        self.assertFalse(any("submit/" in call[0].full_url for call in opener.calls))

    def test_run_not_visible_is_rejected(self):
        with self.assertRaises(dres.DRESError):
            dres.DRESClient("https://example.com", SESSION, FakeOpener(FakeReply([]))).snapshot(RUN)

    def test_202_is_pending_not_correct_and_transport_failure_has_no_retry(self):
        opener = FakeOpener(FakeReply({"status": True, "submission": "INDETERMINATE"}, 202))
        result = dres.DRESClient("https://example.com", SESSION, opener).submit(RUN, {"answerSets": []})
        self.assertTrue(result["pending"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["verdict"], "INDETERMINATE")
        opener = FakeOpener(URLError("network failure with " + SESSION))
        with self.assertRaises(dres.DRESError) as error:
            dres.DRESClient("https://example.com", SESSION, opener).submit(RUN, {})
        self.assertIn("không tự retry", str(error.exception))
        self.assertNotIn(SESSION, str(error.exception))
        self.assertEqual(len(opener.calls), 1)

    def test_redirects_and_cookie_injection_are_rejected(self):
        self.assertIsNone(dres._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.example"))
        for token in ("short", "x" * 32 + "\r\nCookie: injected", "x; " + "y" * 32):
            with self.assertRaises(ValueError):
                dres.DRESClient("https://example.com", token, FakeOpener())

    def test_settings_are_server_side_and_non_loopback_http_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dres.local.json"
            settings = dres.load_dres_settings({"KIS_DRES_BASE_URL": "https://example.com/dres", "KIS_DRES_RUN_ID": RUN}, path)
            self.assertTrue(settings["configured"])
            for url in ("http://example.com", "https://user:pass@example.com", "https://example.com/?session=private", "https://example.com/#private"):
                with self.assertRaises(ValueError):
                    dres.load_dres_settings({"KIS_DRES_BASE_URL": url}, path)


class MappingAndPreviewTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "L01/L01_V001.csv"
        self.path.parent.mkdir()
        self.path.write_text("n,pts_time,fps,frame_idx\n1,0,25,0\n2,6.0005,25,150\n", encoding="utf-8")
        self.client = Mock(base_url="https://example.com", snapshot=Mock(return_value=deepcopy(SNAPSHOT)),
                           submit=Mock(return_value={"accepted": True, "verdict": "WRONG"}))
        self.connection = dres.DRESConnection(self.client, RUN, {"username": "tester"})
        original = dres.mapped_submission
        self.patcher = patch.object(dres, "mapped_submission", side_effect=lambda **kwargs: original(mapping_dir=self.root, **kwargs))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_milliseconds_come_from_pts_time_not_frame_or_image_number(self):
        answer = dres.mapped_submission(**SELECTION, groups=("L01",))
        self.assertEqual(answer["timestamp_ms"], 6001)
        self.assertEqual(answer["answer"], {"mediaItemName": "L01_V001", "start": 6001, "end": 6001})

    def test_frame_offset_shifts_whole_frames_from_mapped_keyframe(self):
        answer = dres.mapped_submission(**SELECTION, frame_offset=-1, groups=("L01",))
        self.assertEqual((answer["timestamp_ms"], answer["submitted_frame_id"], answer["frame_offset"]), (5961, 149, -1))
        self.assertEqual(answer["answer"], {"mediaItemName": "L01_V001", "start": 5961, "end": 5961})
        self.assertNotIn("frame_offset", dres.mapped_submission(**SELECTION, groups=("L01",)))
        with self.assertRaises(ValueError):
            dres.mapped_submission(**SELECTION, frame_offset=-151, groups=("L01",))
        with self.assertRaises(ValueError):
            dres.mapped_submission(**SELECTION, frame_offset=251, groups=("L01",))

    def test_identity_out_of_scope_and_bad_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            dres.mapped_submission(**{**SELECTION, "frame_id": 151}, groups=("L01",))
        with self.assertRaises(ValueError):
            dres.mapped_submission(**SELECTION, groups=("L02",))
        self.path.write_text("n,pts_time,frame_idx\n2,nan,150\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            dres.mapped_submission(**SELECTION, groups=("L01",))

    def test_preview_is_read_only_and_confirm_is_one_use_with_task_id(self):
        preview = self.connection.preview(SELECTION, ("L01",))
        self.client.submit.assert_not_called()
        self.assertEqual(preview["payload"]["answerSets"][0]["taskId"], "task-run-one")
        result = self.connection.submit_preview(preview["preview_id"], ("L01",))
        self.assertEqual(result["verdict"], "WRONG")
        self.client.submit.assert_called_once_with(RUN, preview["payload"])
        with self.assertRaises(dres.DRESError):
            self.connection.submit_preview(preview["preview_id"], ("L01",))
        self.assertEqual(self.client.submit.call_count, 1)

    def test_selected_frame_from_previous_official_task_is_rejected_before_mapping(self):
        with patch.object(dres, "mapped_submission") as mapper:
            with self.assertRaises(dres.DRESError) as error:
                self.connection.preview(SELECTION, ("L01",), task_id="previous-task")
        self.assertEqual(error.exception.status_code, 409)
        mapper.assert_not_called()
        self.client.submit.assert_not_called()

    def test_task_change_ended_expired_or_used_preview_never_sends(self):
        for changed in ({"task_id": "task-two"}, {"can_submit": False}, {}):
            self.client.snapshot.return_value = deepcopy(SNAPSHOT)
            preview = self.connection.preview(SELECTION, ("L01",))
            if changed:
                self.client.snapshot.return_value.update(changed)
            else:
                self.connection.previews[preview["preview_id"]]["expires"] = 0
            with self.assertRaises(dres.DRESError):
                self.connection.submit_preview(preview["preview_id"], ("L01",))
        self.client.submit.assert_not_called()

    def test_mapping_change_and_uncertain_send_consume_preview(self):
        preview = self.connection.preview(SELECTION, ("L01",))
        self.path.write_text("n,pts_time,frame_idx\n2,7,150\n", encoding="utf-8")
        with self.assertRaises(dres.DRESError):
            self.connection.submit_preview(preview["preview_id"], ("L01",))
        self.client.submit.assert_not_called()
        preview = self.connection.preview(SELECTION, ("L01",))
        self.client.submit.side_effect = dres.DRESError("unknown receipt")
        with self.assertRaises(dres.DRESError):
            self.connection.submit_preview(preview["preview_id"], ("L01",))
        with self.assertRaises(dres.DRESError):
            self.connection.submit_preview(preview["preview_id"], ("L01",))
        self.assertEqual(self.client.submit.call_count, 1)

    def test_qa_task_blocks_frame_and_text_preview_sends_only_manual_answer(self):
        self.client.snapshot.return_value["task"]["taskType"] = "LSC Question & Answer Text"
        with self.assertRaises(dres.DRESError):
            self.connection.preview(SELECTION, ("L01",))
        preview = self.connection.preview_text("  user checked answer  ")
        self.client.submit.assert_not_called()
        self.assertEqual(preview["submission_type"], "text")
        self.assertEqual(preview["payload"], {"answerSets": [{"taskId": "task-run-one", "answers": [{"text": "user checked answer"}]}]})
        self.connection.submit_preview(preview["preview_id"], ("L01",))
        self.client.submit.assert_called_once_with(RUN, preview["payload"])

    def test_text_answer_for_wrong_task_or_empty_answer_is_rejected(self):
        with self.assertRaises(dres.DRESError):
            self.connection.preview_text("some answer")
        for text in (" ", "x" * 4001):
            with self.assertRaises(ValueError):
                self.connection.preview_text(text)
        self.client.submit.assert_not_called()

    def test_qa_task_or_answer_mode_change_after_preview_never_sends(self):
        self.client.snapshot.return_value["task"]["taskType"] = "LSC Question & Answer Text"
        preview = self.connection.preview_text("some answer")
        self.client.snapshot.return_value["task"]["taskType"] = "KIS Visual"
        with self.assertRaises(dres.DRESError):
            self.connection.submit_preview(preview["preview_id"], ("L01",))
        self.client.submit.assert_not_called()

    def test_connections_are_isolated_expiring_and_removed_from_ram(self):
        store = dres.DRESConnections()
        token = store.add(self.connection)
        self.assertIs(store.get(token), self.connection)
        with self.assertRaises(dres.DRESError):
            store.get("other-browser")
        self.connection.expires = 0
        with self.assertRaises(dres.DRESError):
            store.get(token)
        token = store.add(dres.DRESConnection(self.client, RUN, {}))
        store.remove(token)
        with self.assertRaises(dres.DRESError):
            store.get(token)


class DRESAPITest(unittest.TestCase):
    def setUp(self):
        self.remote = Mock(base_url="https://example.com", user=Mock(return_value={"username": "tester"}),
                           snapshot=Mock(return_value=deepcopy(SNAPSHOT)))
        app = FastAPI()
        app.include_router(create_dres_router(lambda: ("L01",), client_factory=lambda base: self.remote))
        self.client = TestClient(app, headers={"X-KIS-DRES": "1"})
        self.patcher = patch("kis.dres_api.load_dres_settings", return_value={"base_url": "https://example.com", "evaluation_id": RUN, "configured": True})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def connect(self):
        return self.client.post("/api/dres/connect", json={"evaluation_id": RUN, "session_id": SESSION})

    def test_connect_has_opaque_httponly_cookie_and_no_secret_response(self):
        response = self.connect()
        self.assertEqual(response.status_code, 200)
        cookie = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertNotIn(SESSION, cookie + response.text)
        self.remote.submit.assert_not_called()
        self.assertEqual(self.client.get("/api/dres/status").status_code, 200)

    def test_missing_or_cross_origin_headers_rejected_before_connection(self):
        response = self.client.post("/api/dres/connect", headers={"X-KIS-DRES": ""}, json={"evaluation_id": RUN, "session_id": SESSION})
        self.assertEqual(response.status_code, 403)
        response = self.client.post("/api/dres/connect", headers={"Origin": "https://evil.example"}, json={"evaluation_id": RUN, "session_id": SESSION})
        self.assertEqual(response.status_code, 403)
        self.remote.user.assert_not_called()

    def test_validation_does_not_echo_password_session_or_untrusted_base_url(self):
        for fields in ({"session_id": SESSION, "username": "tester", "password": "supersecret"},
                       {"session_id": SESSION, "base_url": "https://evil.example"}):
            response = self.client.post("/api/dres/connect", json={"evaluation_id": RUN, **fields})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(SESSION, response.text)
            self.assertNotIn("supersecret", response.text)
            self.assertNotIn("evil.example", response.text)
        self.remote.user.assert_not_called()

    def test_confirmed_false_never_sends_and_raw_payload_is_rejected(self):
        self.connect()
        response = self.client.post("/api/dres/submit", json={"preview_id": "x" * 32, "confirmed": False})
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/dres/submit", json={"preview_id": "x" * 32, "confirmed": True, "answerSets": []})
        self.assertEqual(response.status_code, 422)
        self.remote.submit.assert_not_called()

    def test_disconnect_does_not_logout_remote_session(self):
        self.connect()
        self.assertEqual(self.client.post("/api/dres/disconnect", json={}).status_code, 200)
        self.assertEqual(self.client.get("/api/dres/status").status_code, 401)
        self.remote.logout.assert_not_called()

    def test_qa_preview_route_has_no_submission_side_effect(self):
        self.connect()
        self.remote.snapshot.return_value["task"]["taskType"] = "LSC Question & Answer Text"
        response = self.client.post("/api/dres/preview/text", json={"answer_text": "my checked answer"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["payload"]["answerSets"][0]["answers"], [{"text": "my checked answer"}])
        self.remote.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
