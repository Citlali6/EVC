import ast
import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'evisseg_evuav.yaml'


class WindowsCompatibilityTest(unittest.TestCase):
    def test_basedataset_import_does_not_load_sparse_extensions(self):
        basedataset = importlib.import_module('dataset.basedataset')

        self.assertIsNone(basedataset.HAIS_OP)
        self.assertIsNone(basedataset.spconv)
        self.assertTrue(callable(basedataset.voxelization_idx))
        self.assertTrue(callable(basedataset.voxelization))

    def test_missing_sparse_dependency_has_actionable_error(self):
        basedataset = importlib.import_module('dataset.basedataset')
        basedataset.HAIS_OP = None
        basedataset.spconv = None

        with mock.patch.object(
            basedataset.importlib,
            'import_module',
            side_effect=ModuleNotFoundError('missing test dependency'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'spconv.*sparse_weight=0'):
                basedataset.BaseDataLoader.custom_collate([])

        with mock.patch.object(
            basedataset.importlib,
            'import_module',
            side_effect=ModuleNotFoundError('missing test dependency'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'HAIS_OP.*sparse_weight=0'):
                basedataset._load_hais_op()

    def test_full_stream_entrypoints_import_without_sparse_extensions(self):
        code = (
            "import sys; "
            "sys.argv = ['compat-test', '--config', r'{}']; "
            "import test2; import submit_challenge2"
        ).format(CONFIG_PATH)
        completed = subprocess.run(
            [sys.executable, '-c', code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg='stdout:\n{}\nstderr:\n{}'.format(
                completed.stdout, completed.stderr
            ),
        )

    def test_validation_and_submission_share_temporal_memory_routing(self):
        for script_name in ('test2.py', 'submit_challenge2.py'):
            tree = ast.parse(
                (PROJECT_ROOT / script_name).read_text(encoding='utf-8'),
                filename=script_name,
            )
            memory_only_assignments = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == 'temporal_memory_only'
                    for target in node.targets
                )
            ]
            self.assertEqual(len(memory_only_assignments), 1, script_name)
            memory_only_value = memory_only_assignments[0].value
            self.assertIsInstance(memory_only_value, ast.Attribute, script_name)
            self.assertEqual(memory_only_value.attr, 'memory_only', script_name)

            memory_blends = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'blend_temporal_frame_scores'
                and len(node.args) == 3
                and isinstance(node.args[1], ast.Name)
                and node.args[1].id == 'memory_scores'
                and isinstance(node.args[2], ast.Attribute)
                and node.args[2].attr == 'sparse_weight'
            ]
            self.assertEqual(len(memory_blends), 1, script_name)


if __name__ == '__main__':
    unittest.main()
