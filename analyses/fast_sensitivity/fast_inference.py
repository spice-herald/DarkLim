#class and API definition for fast inference
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from dataclasses import dataclass
from tqdm import tqdm

from darklim.utils import get_cache_path
from argparse import ArgumentParser
from copy import deepcopy
import os.path
import darklim
import time





class FastInference():
    parameters = dict()

    def __init__(self, constant_file=None, **parameter_overrides):
        """
            File to read in parameters for each fast likelihood
        """
        if constant_file is not None:
            with open(constant_file) as f:
                self.parameters.update(yaml.safe_load(f))
        self.parameters.update(parameter_overrides)


    def process_inference(self):
        """
            Class which runs the fast/simplified inference
        """
        raise NotImplemented()

    def write_inference_results(self,file_pattern,results):
        """
           Write to a numpy .csv
           file name can have any numbers of python string format statements that will be filled by the parameters,
           allowing them to be organised easily. 
        """
        filename = file_pattern.format(**self.parameters)
        np.savetxt(filename, results)

    @staticmethod
    def run_fast_inference():

        args = FastInference.read_command_line_arguments()

        constant_file = args.pop(constant_file, None)
        output_pattern = args.pop(output_pattern,None)

        fi_object = FastInference(constant_file, **args)

        results = fi_object.process_inference()

        fi_object.write_inference_results(output_pattern, results)
    @staticmethod
    def read_command_line_arguments():  
        """
            Function to perform class-dependent read-in of command line arguments. should return a dict from the
            argparse.
            Minimal arguments: must contain output_pattern
        """
        raise NotImplemented("The subclass must define available arguments to pass!")


class FastInference_GaAs(FastInference):
    """
        Fast optimum interval inference with a GaAs detector
    """
    parameters = dict(
            target = "GaAs",
            )
    def read_command_line_arguments(self):
        parser = ArgumentParser("run a GaAs fast inference run")
        
        parser.add_argument("--wimp_mass", type=float, required=True, help="wimp mass in GeV")
        parser.add_argument("--nexp", type=int, default=goal_GaAs_params.nexp)

        parser.add_argument("--LEE_improvement", type=float, default=None)
        parser.add_argument("--t_days", type=float, default=None)
        parser.add_argument("--flat_rate_DRU", type=float, default=None)



        parsed_args = dict(parser.parse_args())
        return parsed_args
    
    def process_inference(self):
        """
        Run optimum interval GaAs with required params;
        returns: median cross-section only TODO: make PR to print array to make brazil bands https://github.com/spice-herald/DarkLim/blob/f5e53627441eeb08db538574d1617436491c6769/darklim/sensitivity/_sens_est.py#L208
        """
        SE = darklim.sensitivity.SensEst(self.parameters["target_mass_kg"],
                                         self.parameters["t_days"],
                                         tm=self.parameters["target"],
                                         eff=1.,
                                         gain=1.,
                                         seed=int((time.time() + self.parameters["wimp_mass"]))
                                         )
        SE.reset_sim()
        LEE_file = get_cache_path(self.parameters["LEE_filename"])
        SE.add_lee_bkgd_from_file(LEE_file, scale_by=(1/self.parameters["LEE_improvement"]*12.))
        SE.add_flat_bkgd(self.parameters["flat_rate_DRU"])


        gaas_params = {'N_PDs': 1,
                       'E_th_PD': self.parameters["PD_energy_threshold"],
                       'E_res_PD': self.parameters["PD_energy_resolution"],
                       'E_th_GaAs': self.parameters["GaAs_energy_threshold"],
                       'E_res_GaAs': self.parameters["GaAs_energy_resolution"],
                       'collection_efficiency': self.parameters["collection_efficiency"],
                       'GaAs_gamma_energy': 1.33e-3,
                       }

        threshold_keV = gaas_params['E_th_GaAs']

        _, sigma = SE.run_sim(
                threshold_keV,
                e_high=self.parameters["e_high_keV"],
                #e_low=1e-6,
                m_dms=[self.parameters["wimp_mass"]],
                nexp=self.parameters["nexp"],
                #npts=100000,
                plot_bkgd=False,
                #res=GaAs_energy_resolution,
                verbose=True,
                sigma0=self.parameters["sigma0"],
                elf_model=self.parameters["elf_model"],
                elf_target=self.parameters["target"],
                elf_params=self.parameters["elf_params"],
                return_only_drde=False,
                gaas_params=gaas_params,
                adjust_threshold=False,
        )


        return sigma
        


class FastInference_Al2O3(FastInference):
    """
        Fast optimum interval inference with a Al2O3 detector
    """
    parameters = dict(
            target = "Al2O3",
            )
    def read_command_line_arguments(self):
        parser = ArgumentParser("run a Al2O3 fast inference run")
        
        parser.add_argument("--wimp_mass", type=float, required=True, help="wimp mass in GeV")
        parser.add_argument("--nexp", type=int, default=goal_GaAs_params.nexp)

        parser.add_argument("--LEE_improvement", type=float, default=None)
        parser.add_argument("--t_days", type=float, default=None)
        parser.add_argument("--flat_rate_DRU", type=float, default=None)



        parsed_args = dict(parser.parse_args())
        return parsed_args
    
    def process_inference(self):
        """
        Run optimum interval Al2O3 with required params;
        returns: median cross-section only TODO: make PR to print array to make brazil bands https://github.com/spice-herald/DarkLim/blob/f5e53627441eeb08db538574d1617436491c6769/darklim/sensitivity/_sens_est.py#L208
        """

        SE = darklim.sensitivity.SensEst(self.parameters["target_mass_kg"],
                                         self.parameters["t_days"],
                                         tm=self.parameters["target"],
                                         eff=1.,
                                         gain=1.,
                                         seed=int((time.time() + self.parameters["wimp_mass"]))
                                         )
        SE.reset_sim()
        LEE_file = get_cache_path(self.parameters["LEE_filename"])
        SE.add_lee_bkgd_from_file(LEE_file, scale_by=(1/self.parameters["LEE_improvement"]*12.))
        SE.add_flat_bkgd(self.parameters["flat_rate_DRU"])

        per_device_threshold_keV = self.parameters["nsigma"] * self.parameters["baseline_res_eV"] * 1e-3
        threshold_keV = args.coincidence * per_device_threshold_keV
        _, sigma = SE.run_sim(
            threshold_keV,
            e_high=self.parameters["e_high_keV"],
            #e_low=1e-6,
            m_dms=[self.parameters["mass"]],
            nexp=[self.parameters["nexp"]],
            #npts=100000,
            plot_bkgd=False,
            res=self.parameters["baseline_res_eV"]*1e-3,
            verbose=True,
            sigma0=self.parameters["sigma0"],
            elf_model=self.parameters["elf_model"],
            elf_target=self.parameters["target"],
            elf_params=self.parameters["elf_params"],
            return_only_drde=False,
            adjust_threshold=True,
#            gaas_params=None
        )
        return sigma

class FastInference_He(FastInference):
    """
        Fast optimum interval inference with a Al2O3 detector
    """
    parameters = dict(
            target = "He",
            )
    def read_command_line_arguments(self):
        parser = ArgumentParser("run a simplified Herald fast inference run")
        
        parser.add_argument("--wimp_mass", type=float, required=True, help="wimp mass in GeV")
        parser.add_argument("--nexp", type=int, default=goal_GaAs_params.nexp)

        parser.add_argument("--LEE_improvement", type=float, default=None)
        parser.add_argument("--t_days", type=float, default=None)
        parser.add_argument("--flat_rate_DRU", type=float, default=None)



        parsed_args = dict(parser.parse_args())
        return parsed_args
    
    def process_inference(self):
        """
        Run optimum interval Herald with required params;
        returns: median cross-section only TODO: make PR to print array to make brazil bands https://github.com/spice-herald/DarkLim/blob/f5e53627441eeb08db538574d1617436491c6769/darklim/sensitivity/_sens_est.py#L208
        TODO replace LEE assignment
        """

        SE = darklim.sensitivity.SensEst(self.parameters["target_mass_kg"],
                                         self.parameters["t_days"],
                                         tm=self.parameters["target"],
                                         eff=1.,
                                         gain=self.parameters["gain"],
                                         seed=int((time.time() + self.parameters["wimp_mass"]))
                                         )
        SE.reset_sim()
        #SE.add_nfold_lee_bkgd(m=args.n_sensors, n=args.coincidence, w=args.window_s, e0=0.41e-3, R=33.)
        #SE.add_nfold_lee_bkgd(m=args.n_sensors, n=args.coincidence, w=args.window_s, e0=3.81e-3, R=0.0226)
        #SE.add_power_bkgd(1.4e-8, 5.77)
        #SE.add_power_bkgd(0.107, 2.72)

        SE.add_flat_bkgd(self.parameters["flat_rate_DRU"])
        SE.add_nfold_powerlaw_lee_bkgd(m=self.parameters["n_sensors"],
                                                         n=self.parameters["coincidence"],
                                                         w=self.parameters["window_s"])
        
        per_device_threshold_keV = self.parameters["nsigma"] * self.parameters["baseline_res_eV"] * 1e-3
        threshold_keV = self.parameters["coincidence"] * per_device_threshold_keV

        _, sigma = SE.run_sim(
                threshold_keV,
                e_high=50e-3,
                #e_low=1e-6,
                m_dms=[self.parameters["wimp_mass"]],
                nexp=[self.parameters["nexp"]],
                #npts=100000,
                plot_bkgd=False,
                #res=args.baseline_res_eV*1e-3,
                res=np.sqrt(self.parameters["n_sensors"])*self.parameters["baseline_res_eV"]*1e-3,
                verbose=True,
                sigma0=self.parameters["sigma0"],
                elf_model=self.parameters["elf_model"],
                elf_target=self.parameters["target"],
                elf_params=self.parameters["elf_params"],
                return_only_drde=False,
#                gaas_params=None
        )

        return sigma
