QURAN GPT-5.2 FIRST 2,600 — SI-LAB VALIDATION

This clean experiment does not use GPT-OSS logs and does not align the
GPT-5.2 cohort with the old GPT-OSS cohort.

Selection:
- First 2,600 unique evaluable GPT-5.2 SPTP QA records.
- First 2,600 unique evaluable GPT-5.2 MPTP QA records.
- SPTP and MPTP are selected independently.

Main commands:

1. Setup:
.\setup_experiment.ps1 `
  -RepoPath "C:\path\quranqadataset" `
  -Destination "C:\path\quran_gpt52_silab_first2600"

2. Enter experiment folder:
cd "C:\path\quran_gpt52_silab_first2600"

3. Environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

4. Prepare:
python .\scripts\prepare_first_2600.py
python .\scripts\verify_inputs.py

5. Evaluate:
.\run_sptp.ps1
.\run_mptp.ps1

6. Check:
python .\scripts\check_results.py
