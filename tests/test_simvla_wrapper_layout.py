import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WRAPPERS = ROOT / 'architectures/simvla/wrappers'


class WrapperLayoutTest(unittest.TestCase):
    def test_legacy_groups(self):
        self.assertEqual(len(list((WRAPPERS / 'legacy/dcld').glob('*.sh'))), 8)
        self.assertEqual(len(list((WRAPPERS / 'legacy/latentloop').glob('*.sh'))), 10)
        self.assertFalse(list(WRAPPERS.glob('simvla_dcld_*.sh')))
        self.assertFalse(list(WRAPPERS.glob('simvla_latentloop_*.sh')))

    def test_shell_syntax(self):
        for path in WRAPPERS.rglob('*.sh'):
            with self.subTest(path=path):
                subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)

    def test_literal_shell_call_paths(self):
        pattern = re.compile(r'architectures/simvla/wrappers/[a-zA-Z0-9_/]+\.sh')
        for path in WRAPPERS.rglob('*'):
            if path.suffix not in ('.sh', '.py'):
                continue
            for match in pattern.findall(path.read_text()):
                with self.subTest(path=path, target=match):
                    self.assertTrue((ROOT / match).is_file())

    def test_shared_python_api_is_preserved(self):
        for name in ('dcld_eval/rollout_runner.py', 'simvla_dcld_eval.py',
                     'simvla_latentloop_condition_hook_check.py'):
            self.assertTrue((WRAPPERS / name).is_file())


if __name__ == '__main__':
    unittest.main()
