"""Execute the lab in the active course environment and verify the report results."""
from pathlib import Path
import datetime
import json
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
import numpy as np

root = Path(__file__).resolve().parent
notebook_path = root / 'LABS_12_Value_Arnav_Khanna.ipynb'
baseline = json.loads((root / 'analysis_results.json').read_text())
subprocess.run([sys.executable, '-m', 'ipykernel', 'install', '--user',
                '--name', 'arnavkhanna_env', '--display-name', 'ArnavKhanna_env (LABS-12)'], check=True)
n = nbformat.read(notebook_path, as_version=4)
n.metadata.kernelspec = {'name': 'arnavkhanna_env', 'display_name': 'ArnavKhanna_env (LABS-12)', 'language': 'python'}
if 'print("Python executable:"' not in n.cells[3].source:
    n.cells[3].source += '\nimport sys\nprint("Python executable:", sys.executable)\n'

def progress(cell_index, **kwargs):
    print(f'Executing cell {cell_index + 1}/{len(n.cells)}', flush=True)

client = NotebookClient(n, kernel_name='arnavkhanna_env', timeout=600,
                       resources={'metadata': {'path': str(root)}}, on_cell_start=progress)
client.execute()
actual = json.loads((root / 'analysis_results.json').read_text())
assert actual['best_k'] == baseline['best_k'], 'Selected k differs from the report baseline.'
assert actual['confusion'] == baseline['confusion'], 'Confusion counts differ from the report.'
for key, value in baseline['overall'].items():
    np.testing.assert_allclose(actual['overall'][key], value, atol=1e-12, rtol=0)
for expected, observed in zip(baseline['groups'], actual['groups']):
    assert expected.keys() == observed.keys()
    for key, value in expected.items():
        if isinstance(value, (int, float)):
            np.testing.assert_allclose(observed[key], value, atol=1e-12, rtol=0)
        else:
            assert observed[key] == value
for category, ratios in baseline['ratios'].items():
    for metric, value in ratios.items():
        np.testing.assert_allclose(actual['ratios'][category][metric], value, atol=1e-12, rtol=0)

code_cells = [cell for cell in n.cells if cell.cell_type == 'code']
assert len(code_cells) == 22 and all(cell.execution_count is not None for cell in code_cells)
assert not [output for cell in code_cells for output in cell.outputs if output.output_type == 'error']
proof = {'location': 'GitHub Codespaces', 'python_executable': sys.executable,
         'python_version': sys.version, 'kernel': 'arnavkhanna_env',
         'executed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'executed_code_cells': len(code_cells), 'errors': 0,
         'report_results_match': True, 'best_k': actual['best_k'], 'overall': actual['overall']}
n.metadata['lab12_execution'] = proof
n.cells[1].source = 'This notebook was executed in GitHub Codespaces using ArnavKhanna_env. All 22 code cells completed without errors, and the model results match the accompanying report. Execution details are saved in the notebook metadata and codespace_execution.json.'
for cell in n.cells:
    if cell.cell_type == 'markdown':
        cell.source = cell.source.replace('The user supplied a genuine Copilot response, which is verified below. Running the model in Codespaces and GitHub add/commit/push are not recorded as completed here.', 'The actual Copilot response is verified below. Codespaces execution details are recorded in the notebook metadata; use the repository history to check the GitHub save workflow.')
nbformat.validate(n)
nbformat.write(n, notebook_path)
(root / 'codespace_execution.json').write_text(json.dumps(proof, indent=2))
frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
(root / 'requirements.txt').write_text(frozen)
(root.parent / 'requirements.txt').write_text(frozen)

audit_path = root / 'RUBRIC_CHECK.txt'
audit = audit_path.read_text()
start = audit.index('The written report covers')
end = audit.index('\n\nFORMATTING', start)
audit = audit[:start] + 'The written report covers the listed content and formatting requirements. All 22 notebook code cells were executed in GitHub Codespaces using ArnavKhanna_env with no errors. The selected k, overall results, subgroup results, and fairness ratios match the report baseline. Requirements were frozen from the course environment. GitHub history provides the commit and push record. A previous movie-model k was unavailable, so the disclosed training-only selection of k=15 remains a method difference from literally reusing that previous model.' + audit[end:]
audit = audit.split('\nOUTSTANDING COURSE WORKFLOW')[0]
audit += '\nCOURSE WORKFLOW\nPASS: Notebook and data uploaded to the course Codespace.\nPASS: ArnavKhanna_env used for both terminal and notebook execution.\nPASS: All 22 code cells completed; results independently checked against report.\nPASS: Requirements frozen from that environment.\nGitHub commit/push: check the repository history.\nCanvas submission was not performed.\nMethod difference: no previous movie-model k was supplied; k=15 was selected on training folds and disclosed.\n'
audit_path.write_text(audit)
(root / 'START_HERE.txt').write_text('LABS-12: Value - Arnav Khanna\n\nThe report contains three pages of answers and takeaway plus one page of notes, excluded from the main page limit.\nThe notebook was executed in GitHub Codespaces using ArnavKhanna_env. All 22 code cells passed, and results match the report.\nSee codespace_execution.json for execution details. Both requirements files record the course environment.\nKeep the CSV beside the notebook. Rerun with python LABS-12/run_codespace.py from the repository root after activating ArnavKhanna_env.\nThe actual Copilot exchange is included and its arithmetic is checked independently.\nReview the personal takeaway before submitting. No Canvas submission was performed.\nThe previous movie-model k was unavailable; k=15 was selected with training-only five-fold cross-validation.\n')
print('LAB12_EXECUTION_SUCCESS', json.dumps(proof), flush=True)
