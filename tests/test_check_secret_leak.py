"""PR-12 Gate 0: scripts/check_secret_leak.py không được bỏ sót secret thật
lẫn không được báo động giả với placeholder hợp lệ của .env.example."""

from __future__ import annotations

import unittest

from scripts.check_secret_leak import check_forbidden_filenames, check_secret_patterns


class ForbiddenFilenameTests(unittest.TestCase):
    def test_env_local_is_forbidden(self) -> None:
        self.assertEqual(check_forbidden_filenames([".env.local"]), [".env.local"])

    def test_env_dot_something_local_is_forbidden(self) -> None:
        self.assertEqual(
            check_forbidden_filenames(["experiments/.env.fpt.local"]),
            ["experiments/.env.fpt.local"],
        )

    def test_bare_env_is_forbidden(self) -> None:
        self.assertEqual(check_forbidden_filenames([".env"]), [".env"])

    def test_env_example_is_allowed(self) -> None:
        self.assertEqual(check_forbidden_filenames([".env.example"]), [])

    def test_unrelated_files_are_allowed(self) -> None:
        self.assertEqual(
            check_forbidden_filenames(["online/config.py", "README.md"]), []
        )


class SecretPatternTests(unittest.TestCase):
    def _diff(self, filename: str, *added_lines: str) -> str:
        header = f"+++ b/{filename}\n"
        body = "\n".join(f"+{line}" for line in added_lines)
        return header + body

    def test_detects_a_real_looking_api_key_assignment(self) -> None:
        diff = self._diff(".env.local", "AIC_FPT_API_KEY=sk-proj-abcdEFGH12345678")
        issues = check_secret_patterns(diff)
        self.assertTrue(issues)
        self.assertIn(".env.local", issues[0])

    def test_placeholder_change_me_is_not_flagged(self) -> None:
        diff = self._diff(".env.example", "AIC_GPU_API_KEY=change-me")
        self.assertEqual(check_secret_patterns(diff), [])

    def test_empty_value_is_not_flagged(self) -> None:
        diff = self._diff(".env.example", "AIC_FPT_API_KEY=")
        self.assertEqual(check_secret_patterns(diff), [])

    def test_angle_bracket_placeholder_is_not_flagged(self) -> None:
        diff = self._diff("docs/example.md", "AIC_FPT_API_KEY=<SECRET>")
        self.assertEqual(check_secret_patterns(diff), [])

    def test_detects_bearer_token_in_code(self) -> None:
        diff = self._diff("bad.py", 'headers = {"Authorization": "Bearer sk-realtoken1234567890"}')
        self.assertTrue(check_secret_patterns(diff))

    def test_detects_openai_style_key_anywhere(self) -> None:
        diff = self._diff("notes.txt", "using key sk-abcdefghijklmnop1234 for testing")
        self.assertTrue(check_secret_patterns(diff))

    def test_detects_aws_access_key(self) -> None:
        diff = self._diff("bad.tf", "access_key = AKIAABCDEFGHIJKLMNOP")
        self.assertTrue(check_secret_patterns(diff))

    def test_detects_private_key_block(self) -> None:
        diff = self._diff("id_rsa", "-----BEGIN RSA PRIVATE KEY-----")
        self.assertTrue(check_secret_patterns(diff))

    def test_removed_lines_are_not_scanned(self) -> None:
        # Dòng bị XÓA (bắt đầu bằng '-') không phải nội dung sắp commit vào —
        # không được báo động vì đó, chỉ dòng THÊM MỚI mới đáng quan tâm.
        diff = "+++ b/.env.local\n-AIC_FPT_API_KEY=sk-realtoken1234567890\n"
        self.assertEqual(check_secret_patterns(diff), [])

    def test_diff_header_lines_are_not_scanned_as_content(self) -> None:
        diff = "+++ b/online/config.py\n+API_KEY = os.getenv('AIC_FPT_API_KEY')\n"
        # Dòng code đọc từ env (không phải giá trị secret thật) không được flag.
        self.assertEqual(check_secret_patterns(diff), [])

    def test_ordinary_code_change_is_clean(self) -> None:
        diff = self._diff("online/services/search.py", "def search(self, request):", "    return await self._search_impl(request)")
        self.assertEqual(check_secret_patterns(diff), [])

    def test_own_test_file_is_exempt_from_scanning_its_own_fixtures(self) -> None:
        # File này CHỦ Ý chứa chuỗi trông giống secret để test detector — nếu
        # không tự loại trừ, mọi lần thêm test case mới cho secret thật sẽ tự
        # chặn chính commit thêm test đó.
        diff = self._diff(
            "tests/test_check_secret_leak.py",
            'diff = self._diff(".env.local", "AIC_FPT_API_KEY=sk-realFAKEtoken1234567890")',
        )
        self.assertEqual(check_secret_patterns(diff), [])

    def test_exemption_does_not_leak_to_other_files_in_the_same_diff(self) -> None:
        diff = (
            self._diff("tests/test_check_secret_leak.py", "fixture = 'sk-abcdefghijklmnop1234'")
            + "\n"
            + self._diff(".env.local", "AIC_FPT_API_KEY=sk-realtoken1234567890abcdef")
        )
        issues = check_secret_patterns(diff)
        self.assertTrue(issues)
        self.assertTrue(all(".env.local" in item for item in issues))


if __name__ == "__main__":
    unittest.main()
