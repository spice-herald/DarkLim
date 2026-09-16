#!/bin/bash
python silicon_oi_scan.py --results_dir results/silicon_power_bkgd_1sec_370meV_phonon_massive --baseline_res_eV 0.37 --masses_GeV 1e-4 1e0 72 --elf phonon mediator massive;
python silicon_oi_scan.py --results_dir results/silicon_power_bkgd_1sec_200meV_phonon_massive --baseline_res_eV 0.20 --masses_GeV 1e-4 1e0 72 --elf phonon mediator massive;
python silicon_oi_scan.py --results_dir results/silicon_power_bkgd_1sec_100meV_phonon_massive --baseline_res_eV 0.10 --masses_GeV 1e-4 1e0 72 --elf phonon mediator massive;
python silicon_oi_scan.py --results_dir results/silicon_power_bkgd_1sec_80meV_phonon_massive --baseline_res_eV 0.08 --masses_GeV 1e-4 1e0 72 --elf phonon mediator massive;
python silicon_oi_scan.py --results_dir results/silicon_power_bkgd_1sec_50meV_phonon_massive --baseline_res_eV 0.05 --masses_GeV 1e-4 1e0 72 --elf phonon mediator massive;
python silicon_oi_scan.py --results_dir results/silicon_power_bkgd_1sec_30meV_phonon_massive --baseline_res_eV 0.03 --masses_GeV 1e-4 1e0 72 --elf phonon mediator massive;
python silicon_oi_scan.py --results_dir results/silicon_power_bkgd_1sec_20meV_phonon_massive --baseline_res_eV 0.02 --masses_GeV 1e-4 1e0 72 --elf phonon mediator massive;
