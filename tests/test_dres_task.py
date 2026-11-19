import base64
from copy import deepcopy
from email.message import Message
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kis import dres
from kis.dres_api import create_dres_router
from tests.test_dres import FakeOpener, FakeReply, RUN, SESSION, SNAPSHOT


TEMPLATE = "template-test"
STATE = {"evaluationStatus": "ACTIVE", "taskStatus": "RUNNING", "taskId": SNAPSHOT["task_id"],
         "taskTemplateId": TEMPLATE, "timeElapsed": 15}
CURRENT = {**SNAPSHOT, "task_template_id": TEMPLATE}


def hint(*sequence):
    return FakeReply({"taskId": TEMPLATE, "sequence": list(sequence), "loop": False})


class DRESCurrentHintTest(unittest.TestCase):
    def client(self, *replies):
        return dres.DRESClient("https://example.com/dres", SESSION, FakeOpener(*replies))

    def test_only_current_released_text_accumulates_and_never_fetches_future_or_target(self):
        client = self.client(hint(
            {"contentType": "TEXT", "offset": 0, "content": "old text"},
            {"contentType": "TEXT", "offset": 10, "content": "current <text>"},
            {"contentType": "TEXT", "offset": 30, "content": "future secret"}), FakeReply(STATE))
        result = client.task_content(CURRENT)
        self.assertEqual(result["elements"], [
            {"type": "TEXT", "index": 0, "offset": 0, "text": "old text"},
            {"type": "TEXT", "index": 1, "offset": 10, "text": "current <text>"},
        ])
        urls = [call[0].full_url for call in client.opener.calls]
        self.assertNotIn("target", str(urls))
        self.assertNotIn("download", str(urls))
        self.assertNotIn("future secret", str(result))

    def test_manual_refresh_rereads_the_hint_sequence_for_new_sentences(self):
        client = self.client(
            hint({"contentType": "TEXT", "offset": 0, "content": "first"}),
            FakeReply(STATE),
            hint({"contentType": "TEXT", "offset": 0, "content": "first"},
                 {"contentType": "TEXT", "offset": 12, "content": "second"}),
            FakeReply(STATE),
        )
        self.assertEqual([item["text"] for item in client.task_content(CURRENT)["elements"]], ["first"])
        self.assertEqual([item["text"] for item in client.task_content(CURRENT, refresh=True)["elements"]],
                         ["first", "second"])

    def test_empty_element_clears_text_and_countdown_releases_nothing(self):
        for elapsed in (15, -1):
            with self.subTest(elapsed=elapsed):
                client = self.client(hint(
                    {"contentType": "TEXT", "offset": 0, "content": "old"},
                    {"contentType": "TEXT", "offset": 10}), FakeReply({**STATE, "timeElapsed": elapsed}))
                self.assertEqual(client.task_content(CURRENT)["elements"], [])

    def test_no_running_task_does_not_fetch_hint(self):
        client = self.client()
        self.assertEqual(client.task_content({**CURRENT, "can_submit": False})["status"], "waiting")
        self.assertEqual(client.opener.calls, [])

    def test_display_preserves_full_released_sentence_even_over_search_limit(self):
        client = self.client(hint({"contentType": "TEXT", "offset": 0, "content": "x" * 8001}), FakeReply(STATE))
        self.assertEqual(len(client.task_content(CURRENT)["elements"][0]["text"]), 8001)

    def test_changed_task_during_hint_read_discards_cached_media(self):
        client = self.client(hint({"contentType": "TEXT", "offset": 0, "content": "old"}),
                             FakeReply({**STATE, "taskId": "changed"}))
        with self.assertRaises(dres.DRESError) as error:
            client.task_content(CURRENT)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(client._hint_sequence, [])

    def test_inline_video_has_local_media_url_and_no_base64_in_task_response(self):
        encoded = base64.b64encode(b"video test bytes").decode()
        client = self.client(hint({"contentType": "VIDEO", "offset": 0, "content": encoded}),
                             FakeReply(STATE), FakeReply(STATE))
        result = client.task_content(CURRENT)
        self.assertEqual(result["elements"][0]["media_url"], f"/api/dres/task/media/{CURRENT['task_id']}/0")
        self.assertNotIn(encoded, str(result))
        self.assertEqual(client.task_media(CURRENT, CURRENT["task_id"], 0), (b"video test bytes", "video/mp4"))
        self.assertEqual(len(client.opener.calls), 3)  # hint reused, state rechecked

    def test_old_future_or_finished_hint_media_cannot_be_requested(self):
        client = self.client()
        with self.assertRaises(dres.DRESError):
            client.task_media(CURRENT, "old-task", 0)
        self.assertEqual(client.opener.calls, [])
        client = self.client(hint({"contentType": "VIDEO", "offset": 30, "content": "dGVzdA=="}), FakeReply(STATE))
        with self.assertRaises(dres.DRESError):
            client.task_media(CURRENT, CURRENT["task_id"], 0)

    def test_media_url_cannot_fetch_external_host_target_query_or_redirect(self):
        for url in ("https://evil.example/video.mp4", "/dres/api/v2/target/answer",
                    f"/dres/api/v2/evaluation/{RUN}/template/task/{TEMPLATE}/hint/0/media?session=secret",
                    f"/dres/api/v2/evaluation/{RUN}/template/task/OTHER/hint/0/media"):
            with self.subTest(url=url):
                client = self.client(hint({"contentType": "VIDEO", "offset": 0, "contentUrl": url}), FakeReply(STATE))
                with self.assertRaises(dres.DRESError):
                    client.task_media(CURRENT, CURRENT["task_id"], 0)
                self.assertEqual(len(client.opener.calls), 2)

    def test_canonical_current_hint_media_route_uses_cookie_and_size_bound(self):
        class MediaReply(BytesIO):
            status = 200
            headers = Message()
        MediaReply.headers["Content-Type"] = "video/mp4"
        path = f"/dres/api/v2/evaluation/{RUN}/template/task/{TEMPLATE}/hint/0/media"
        client = self.client(hint({"contentType": "VIDEO", "offset": 0, "contentUrl": path}),
                             FakeReply(STATE), MediaReply(b"video"), FakeReply(STATE))
        self.assertEqual(client.task_media(CURRENT, CURRENT["task_id"], 0), (b"video", "video/mp4"))
        request = client.opener.calls[-2][0]
        self.assertEqual(request.full_url, "https://example.com" + path)
        self.assertEqual(request.get_header("Cookie"), "SESSIONID=" + SESSION)

    def test_bad_base64_wrong_task_or_oversize_hint_fail_closed(self):
        client = self.client(hint({"contentType": "VIDEO", "offset": 0, "content": "invalid!!"}), FakeReply(STATE))
        with self.assertRaises(dres.DRESError):
            client.task_media(CURRENT, CURRENT["task_id"], 0)
        client = self.client(FakeReply({"taskId": "wrong", "sequence": []}))
        with self.assertRaises(dres.DRESError):
            client.task_content(CURRENT)
        client = self.client(hint({"contentType": "TEXT", "offset": 0, "content": "large" * 100}))
        with patch.object(dres, "MAX_HINT_BYTES", 100), self.assertRaises(dres.DRESError):
            client.task_content(CURRENT)

    def test_real_server_root_relative_media_url_is_rebased_under_dres_mount(self):
        class MediaReply(BytesIO):
            status = 200
            headers = Message()
        MediaReply.headers["Content-Type"] = "video/mp4"
        path = f"/api/v2/evaluation/{RUN}/template/task/{TEMPLATE}/hint/0/media"
        client = self.client(hint({"contentType": "VIDEO", "offset": 0, "contentUrl": path}),
                             FakeReply(STATE), MediaReply(b"video"), FakeReply(STATE))
        self.assertEqual(client.task_media(CURRENT, CURRENT["task_id"], 0), (b"video", "video/mp4"))
        self.assertEqual(client.opener.calls[-2][0].full_url, "https://example.com/dres" + path)

    def test_task_change_while_downloading_video_does_not_return_old_media(self):
        class MediaReply(BytesIO):
            status = 200
            headers = Message()
        MediaReply.headers["Content-Type"] = "video/mp4"
        path = f"/api/v2/evaluation/{RUN}/template/task/{TEMPLATE}/hint/0/media"
        client = self.client(hint({"contentType": "VIDEO", "offset": 0, "contentUrl": path}),
                             FakeReply(STATE), MediaReply(b"video"), FakeReply({**STATE, "taskId": "new-task"}))
        with self.assertRaises(dres.DRESError) as error:
            client.task_media(CURRENT, CURRENT["task_id"], 0)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(client._hint_sequence, [])


class DRESTaskAPITest(unittest.TestCase):
    def setUp(self):
        self.remote = Mock(base_url="https://example.com", user=Mock(return_value={"username": "tester"}),
                           snapshot=Mock(return_value=deepcopy(CURRENT)),
                           task_content=Mock(return_value={"status": "available", "elements": [{"type": "TEXT", "text": "boat"}]}),
                           task_media=Mock(return_value=(b"video bytes", "video/mp4")))
        app = FastAPI()
        app.include_router(create_dres_router(lambda: ("L01",), client_factory=lambda base: self.remote))
        self.client = TestClient(app, headers={"X-KIS-DRES": "1"})
        with patch("kis.dres_api.load_dres_settings", return_value={"base_url": "https://example.com", "configured": True}):
            self.assertEqual(self.client.post("/api/dres/connect", json={"evaluation_id": RUN, "session_id": SESSION}).status_code, 200)

    def test_task_and_media_are_private_no_store_and_never_submit(self):
        result = self.client.get("/api/dres/task")
        self.assertEqual(result.json()["content"]["elements"][0]["text"], "boat")
        self.assertEqual(result.headers["cache-control"], "no-store")
        self.remote.task_content.assert_called_with(deepcopy(CURRENT), refresh=False)
        result = self.client.get("/api/dres/task?refresh=1")
        self.assertEqual(result.status_code, 200)
        self.remote.task_content.assert_called_with(deepcopy(CURRENT), refresh=True)
        result = self.client.get(f"/api/dres/task/media/{CURRENT['task_id']}/0")
        self.assertEqual(result.content, b"video bytes")
        self.assertEqual(result.headers["cache-control"], "no-store")
        self.assertEqual(result.headers["content-type"], "video/mp4")
        self.remote.submit.assert_not_called()

    def test_no_custom_header_or_session_cannot_fetch_media(self):
        route = f"/api/dres/task/media/{CURRENT['task_id']}/0"
        self.assertEqual(self.client.get(route, headers={"X-KIS-DRES": ""}).status_code, 403)
        self.client.cookies.clear()
        self.assertEqual(self.client.get(route).status_code, 401)
        self.remote.task_media.assert_not_called()

    def test_participant_cannot_view_hint_keeps_login_but_shows_explicit_unavailable(self):
        self.remote.task_content.side_effect = dres.DRESError("Not allowed", 403)
        result = self.client.get("/api/dres/task")
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["connected"])
        self.assertEqual(result.json()["content"]["status"], "unavailable")
        self.assertEqual(result.json()["content"]["elements"], [])


class DRESOfficialSearchTest(unittest.TestCase):
    def setUp(self):
        self.elements = [{"type": "TEXT", "text": "Official boat query", "index": 0, "offset": 0}]
        self.client = Mock(base_url="https://example.com", snapshot=Mock(return_value=deepcopy(CURRENT)),
                           task_content=Mock(return_value={"status": "available", "elements": self.elements}),
                           task_media=Mock(return_value=(b"official-video", "video/mp4")))
        self.connection = dres.DRESConnection(self.client, RUN, {})
        frames = [{"video_id": "L01_V001", "frame_id": i} for i in range(200)]
        self.service = Mock(search=Mock(return_value={"input_type": "text", "results": frames}),
                            search_video=Mock(return_value={"input_type": "video", "results": frames}))

    def test_text_is_fetched_server_side_and_never_uses_a_browser_query(self):
        result = self.connection.search_task(CURRENT["task_id"], self.service)
        self.service.search.assert_called_once_with("Official boat query")
        self.service.search_video.assert_not_called()
        self.assertEqual(result["dres_task"]["source"], "dres_official")
        self.client.submit.assert_not_called()

    def test_video_bytes_are_fetched_from_official_hint_and_use_official_decoder(self):
        self.elements[:] = [{"type": "VIDEO", "index": 0, "media_url": "local", "offset": 0}]
        result = self.connection.search_task(CURRENT["task_id"], self.service)
        self.service.search_video.assert_called_once_with(b"official-video", official=True)
        self.service.search.assert_not_called()
        self.assertEqual(result["dres_task"]["input_type"], "video")

    def test_video_description_is_supplemental_never_replaces_official_video(self):
        self.elements[:] = [{"type": "VIDEO", "index": 0, "media_url": "local", "offset": 0}]
        result = self.connection.search_task(CURRENT["task_id"], self.service, "  red shirt  ")
        self.client.task_media.assert_called_once_with(CURRENT, CURRENT["task_id"], 0)
        self.service.search_video.assert_called_once_with(b"official-video", official=True, description="red shirt")

        self.service.search.assert_not_called()
        self.assertTrue(result["dres_task"]["has_supplemental_description"])
        self.client.submit.assert_not_called()

    def test_video_with_official_text_uses_both_text_and_video_for_hybrid_query(self):
        self.elements[:] = [
            {"type": "TEXT", "text": "first sentence", "index": 0, "offset": 0},
            {"type": "TEXT", "text": "second sentence", "index": 1, "offset": 30},
            {"type": "VIDEO", "index": 2, "media_url": "local", "offset": 0},
        ]
        self.client.snapshot.return_value = {**deepcopy(CURRENT), "task": {"taskType": "Visual KIS"}}
        result = self.connection.search_task(CURRENT["task_id"], self.service, "red shirt")
        self.service.search_video.assert_called_once_with(
            b"official-video", official=True, hint_text="first sentence\nsecond sentence",
            description="red shirt")
        self.assertEqual(result["dres_task"]["query_text_chars"], len("first sentence\nsecond sentence"))
        self.client.submit.assert_not_called()

    def test_text_task_uses_every_released_sentence(self):
        self.elements[:] = [
            {"type": "TEXT", "text": "first sentence", "index": 0, "offset": 0},
            {"type": "TEXT", "text": "second sentence", "index": 1, "offset": 30},
        ]
        self.connection.search_task(CURRENT["task_id"], self.service)
        self.service.search.assert_called_once_with("first sentence\nsecond sentence")

    def test_next_search_refreshes_full_sequence_without_reset_or_reconnect(self):
        # The first display is cached; the server later adds a sentence to its hint JSON.
        old = {"contentType": "TEXT", "offset": 0, "content": "first sentence"}
        new = {"contentType": "TEXT", "offset": 12, "content": "second sentence"}
        remote = dres.DRESClient("https://example.com/dres", SESSION, FakeOpener(
            hint(old), FakeReply(STATE),
            hint(old, new), FakeReply(STATE),
            hint(old, new), FakeReply(STATE)))
        remote.snapshot = Mock(return_value=deepcopy(CURRENT))
        connection = dres.DRESConnection(remote, RUN, {})
        self.assertEqual(connection.task()["content"]["elements"][0]["text"], "first sentence")
        result = connection.search_task(CURRENT["task_id"], self.service)
        self.service.search.assert_called_once_with("first sentence\nsecond sentence")
        self.assertEqual([row["text"] for row in result["dres_content"]["elements"]],
                         ["first sentence", "second sentence"])
        self.assertEqual(len(result["results"]), 200)
        self.assertEqual(sum("/hint" in req.full_url for req, _ in remote.opener.calls), 3)
        self.assertFalse(any("login" in req.full_url for req, _ in remote.opener.calls))

    def test_search_refreshes_before_and_after_inference_and_returns_used_content(self):
        result = self.connection.search_task(CURRENT["task_id"], self.service)
        self.assertEqual(self.client.task_content.call_count, 2)
        self.assertTrue(all(call.kwargs == {"refresh": True}
                            for call in self.client.task_content.call_args_list))
        self.assertEqual(result["dres_content"]["elements"], self.elements)
        self.assertEqual(result["dres_run"]["task_id"], CURRENT["task_id"])

    def test_full_text_over_limit_is_an_error_not_silent_truncation(self):
        self.elements[:] = [{"type": "TEXT", "text": "x" * 8001}]
        with self.assertRaises(dres.DRESError) as error:
            self.connection.search_task(CURRENT["task_id"], self.service)
        self.assertEqual(error.exception.status_code, 422)
        self.service.search.assert_not_called()

    def test_search_does_not_claim_success_with_incomplete_frame_pool(self):
        from kis.result_pool import IncompleteResultPool
        self.service.search.return_value = {"results": [{"video_id": "L01_V001", "frame_id": 1}]}
        with self.assertRaises(IncompleteResultPool):
            self.connection.search_task(CURRENT["task_id"], self.service)

    def test_legacy_hundred_result_pool_is_rejected_for_two_hundred_web(self):
        from kis.result_pool import IncompleteResultPool
        self.service.search.return_value = {"results": [
            {"video_id": "L01_V001", "frame_id": i} for i in range(100)]}
        with self.assertRaises(IncompleteResultPool) as error:
            self.connection.search_task(CURRENT["task_id"], self.service)
        self.assertEqual(error.exception.target, 200)

    def test_description_for_text_task_or_invalid_description_is_rejected(self):
        for value in ("x" * 2001, None):
            with self.assertRaises(ValueError):
                self.connection.search_task(CURRENT["task_id"], self.service, value)
        with self.assertRaises(dres.DRESError) as error:
            self.connection.search_task(CURRENT["task_id"], self.service, "not a replacement query")
        self.assertEqual(error.exception.status_code, 422)
        self.service.search.assert_not_called()
        self.service.search_video.assert_not_called()

    def test_missing_hint_waiting_and_stale_task_do_not_invoke_model(self):
        with self.assertRaises(dres.DRESError):
            self.connection.search_task("old", self.service)
        self.elements.clear()
        with self.assertRaises(dres.DRESError):
            self.connection.search_task(CURRENT["task_id"], self.service)
        self.service.search.assert_not_called()
        self.service.search_video.assert_not_called()

    def test_task_ends_or_changes_during_model_search_discards_response(self):
        self.client.snapshot.side_effect = [deepcopy(CURRENT), {**CURRENT, "task_id": "new-task"}]
        with self.assertRaises(dres.DRESError) as error:
            self.connection.search_task(CURRENT["task_id"], self.service)
        self.assertEqual(error.exception.status_code, 409)
        self.service.search.assert_called_once()

    def test_new_hint_in_same_task_during_search_discards_response(self):
        self.client.task_content.side_effect = [
            {"status": "available", "elements": self.elements},
            {"status": "available", "elements": [{"type": "TEXT", "text": "new hint"}]},
        ]
        with self.assertRaises(dres.DRESError) as error:
            self.connection.search_task(CURRENT["task_id"], self.service)
        self.assertEqual(error.exception.status_code, 409)

    def test_search_route_rejects_manual_text_video_or_url_fields(self):
        app = FastAPI()
        self.client.user.return_value = {"username": "tester"}
        app.include_router(create_dres_router(lambda: ("L01",), client_factory=lambda base: self.client,
                                             search_service_provider=lambda: self.service))
        local = TestClient(app, headers={"X-KIS-DRES": "1"})
        with patch("kis.dres_api.load_dres_settings", return_value={"base_url": "https://example.com", "configured": True}):
            self.assertEqual(local.post("/api/dres/connect", json={"evaluation_id": RUN, "session_id": SESSION}).status_code, 200)
        for field, value in (("query", "fake"), ("video_url", "https://evil.example"), ("video", "fake")):
            response = local.post("/api/dres/search", json={"task_id": CURRENT["task_id"], field: value})
            self.assertEqual(response.status_code, 422)
        self.service.search.assert_not_called()
        response = local.post("/api/dres/search", json={"task_id": CURRENT["task_id"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        for description in ("x" * 2001, None):
            self.assertEqual(local.post("/api/dres/search", json={"task_id": CURRENT["task_id"],
                "video_description": description}).status_code, 422)
        self.elements[:] = [{"type": "VIDEO", "index": 0, "offset": 0}]
        response = local.post("/api/dres/search", json={"task_id": CURRENT["task_id"], "video_description": "red shirt"})
        self.assertEqual(response.status_code, 200)
        self.service.search_video.assert_called_once_with(b"official-video", official=True, description="red shirt")

        self.service.search_video.return_value = {"results": [{"video_id": "L01_V001", "frame_id": 1}]}
        response = local.post("/api/dres/search", json={"task_id": CURRENT["task_id"]})
        self.assertEqual(response.status_code, 503)
        self.assertIn("200", response.json()["detail"])
        self.assertIn("1", response.json()["detail"])

    def test_planner_failure_message_is_not_hidden_by_generic_docker_error(self):
        from kis.planner_fallback import QueryPlanningError
        app = FastAPI()
        self.client.user.return_value = {"username": "tester"}
        app.include_router(create_dres_router(lambda: ("L01",), client_factory=lambda base: self.client,
                                             search_service_provider=lambda: self.service))
        local = TestClient(app, headers={"X-KIS-DRES": "1"})
        with patch("kis.dres_api.load_dres_settings", return_value={"base_url": "https://example.com", "configured": True}):
            local.post("/api/dres/connect", json={"evaluation_id": RUN, "session_id": SESSION})
        self.service.search.side_effect = QueryPlanningError("Groq failed (dns); OpenCode fallback failed.")
        response = local.post("/api/dres/search", json={"task_id": CURRENT["task_id"]})
        self.assertEqual(response.status_code, 503)
        self.assertIn("OpenCode", response.json()["detail"])
        self.assertNotIn("Docker", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
