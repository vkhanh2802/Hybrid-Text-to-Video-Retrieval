from pathlib import Path
import shutil
import subprocess
import unittest

from kis.paths import KIS_ROOT


@unittest.skipUnless(shutil.which("node"), "requires Node for frontend interaction tests")
class FrontendInputsTest(unittest.TestCase):
    def run_scenario(self, scenario):
        harness = Path(__file__).with_name("frontend_input_harness.cjs")
        result = subprocess.run([shutil.which("node"), str(harness), str(KIS_ROOT / "web/app.js"), scenario],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_only_official_dres_input_is_searched_and_old_task_results_are_cleared(self):
        self.run_scenario("official")

    def test_html_has_no_manual_text_or_video_query_inputs(self):
        html = (KIS_ROOT / "web/index.html").read_text(encoding="utf-8")
        for legacy in ('id="query"', 'id="video-file"', 'id="mode-text"', 'id="mode-video"'):
            self.assertNotIn(legacy, html)
        self.assertIn("Tìm theo đề DRES", html)
        self.assertIn('id="dres-video-description"', html)
        self.assertIn('id="dres-manual-toggle"', html)

    def test_no_automatic_polling_and_manual_refresh_only(self):
        script = (KIS_ROOT / "web/app.js").read_text(encoding="utf-8")
        self.assertNotIn("scheduleDresRefresh", script)
        self.assertNotIn("setInterval", script)
        self.assertNotIn("}, 5000)", script)
        self.assertIn("task?refresh=1", script)
        html = (KIS_ROOT / "web/index.html").read_text(encoding="utf-8")
        self.assertNotIn("Tự cập nhật mỗi 5 giây", html)

    def test_optional_description_only_for_official_video_and_reset_on_task_change(self):
        self.run_scenario("video-description")

    def test_dres_preview_cancel_explicit_confirm_and_double_click(self):
        self.run_scenario("dres")

    def test_dres_login_wrong_verdict_and_uncertain_send_have_no_retry(self):
        self.run_scenario("errors")

    def test_qa_is_manual_text_only_and_cleared_when_task_changes(self):
        self.run_scenario("qa")

    def test_current_task_text_video_cleanup_and_session_expiry(self):
        self.run_scenario("task")

    def test_two_hundred_cards_can_select_last_frame_without_llm_score(self):
        self.run_scenario("two-hundred")

    def test_restored_session_hides_large_login_panel(self):
        self.run_scenario("restored")

    def test_two_hundred_images_lazy_load_with_eight_request_limit_and_cleanup(self):
        self.run_scenario("images")

    def test_new_text_searches_full_prompt_without_reconnect_or_automatic_retry(self):
        self.run_scenario("full-text")

    def test_new_sentence_preserves_playing_media_and_video_description(self):
        self.run_scenario("media-no-reset")

    def test_temporal_is_separate_from_the_two_hundred_frame_grid(self):
        self.run_scenario("auxiliary-temporal")

    def test_manual_query_searches_without_dres_requests_or_submission_buttons(self):
        self.run_scenario("manual")

    def test_dres_task_optional_manual_query_does_not_flag_or_submit_task(self):
        self.run_scenario("dres-manual-query")

    def test_vlm_off_message_replaces_scoring_claim(self):
        self.run_scenario("vlm-off")

    def test_dres_frame_adjust_requests_fresh_validated_previews(self):
        self.run_scenario("frame-adjust")

    def test_similar_search_from_card_button_and_s_key_with_back(self):
        self.run_scenario("similar")

    def test_actual_planner_and_fallback_reason_are_visible_and_reset(self):
        self.run_scenario("planner")


if __name__ == "__main__":
    unittest.main()
