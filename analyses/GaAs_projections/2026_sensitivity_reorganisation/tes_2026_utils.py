import numpy as np
from darklim.utils import get_cache_path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import detprocess
from glob import glob
from tqdm import tqdm
import pickle
import darklim
import scipy.stats as sps
from scipy.optimize import brentq


def get_LEE_bkgd(scale_by=1.0, LEE_file = None):
    """
        load a .txt file with LEE spectra, divides it by scale_by.
        lee_file is passed to get_cache_path to allow it to call the module data. 
    """
    fin = get_cache_path(LEE_file)
    template = np.loadtxt(fin,skiprows=1)
    #86400 = number of sec in a day
    interp_func = interp1d(template[:,0],scale_by*template[:,1] / 86400,
                            bounds_error=False,
                            #fill_value='extrapolate',
                            fill_value=0.,
                            assume_sorted=True)

    return interp_func

def extrapolate(data, mass=1e6):
    m0, s0 = data[-1]
    return np.concatenate([data,[[mass, s0*mass/m0]]], axis=0)

def minbound(curves, masses=None):
    if masses is None:
        masses = np.logspace(-3, 4, 200)
    result = np.array([np.nan]*len(masses))
    for curve in curves:
        if curve[-1][0] < masses[-1]:
            curve = extrapolate(curve, masses[-1])
        yinterp = 10**(interp1d(*(np.log10(curve).T), bounds_error=False)(np.log10(masses)))
        #yinterp = interp1d(*(curve.T), bounds_error=False)(masses)
        result = np.nanmin([result, yinterp], axis=0)
    return np.array([masses, result]).T

def log_interp(zz, xx, yy, left=None, right=None):
    logz = np.log(zz)
    logx = np.log(xx)
    logy = np.log(yy)
    return np.exp(np.interp(logz, logx, logy, left=left, right=right))


def load_and_plot_existing(leg=True, lw=1, leg_params={}, ER_model='massless', cosmic=False, migdal=False, direct=False):
    curves_dir = get_cache_path("data/ExistingLimits")
    handles = []
    fig, ax = plt.subplots(1, 1, figsize=(10,7.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    mpl.rcParams['font.size'] = 14
    
    if ER_model == 'massive':

        m_lim, x_lim = np.loadtxt(curves_dir + 'DAMIC_LBC_ER_Massive_2025.txt').transpose()
        l, = ax.plot(m_lim*1e3, x_lim, '--', color='magenta', lw=lw, label='LBC (DAMIC) 2025')
        handles.append(l)
        ax.fill_between(m_lim*1e3, x_lim, 1e-22, facecolor='whitesmoke')
        
        m_lim_1, x_lim_1 = np.loadtxt(curves_dir + 'DAMIC_ER_Massive_2023.txt').transpose()
        m_lim_2, x_lim_2 = np.loadtxt(curves_dir + 'DAMIC_ER_Massive_2024.txt').transpose()
        m_lim = np.geomspace(min(m_lim_1[0], m_lim_2[0]), max(m_lim_1[-1], m_lim_2[-1]), 150)
        x_lim_1_adj = log_interp(m_lim, m_lim_1, x_lim_1, left=np.inf, right=np.inf)
        x_lim_2_adj = log_interp(m_lim, m_lim_2, x_lim_2, left=np.inf, right=np.inf)
        x_lim = np.minimum(x_lim_1_adj, x_lim_2_adj)
        l, = ax.plot(m_lim, x_lim, '--', color='#CFC0E5', lw=lw, label='DAMIC 2023, 2024')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e-22, facecolor='whitesmoke')
        
        m_lim, x_lim = np.loadtxt(curves_dir + 'Darkside50_ER_Massive_2023.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='#CE977A', lw=lw, label='Darkside-50 2023')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e-22, facecolor='whitesmoke')

        m_lim, x_lim = np.loadtxt(curves_dir + 'SENSEI_SNOLAB_ER_Massive.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='#a4dbed', lw=lw, label='SENSEI 2025')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e-22, facecolor='whitesmoke')
        
        m_lim, x_lim = np.loadtxt(curves_dir + 'PandaX4T_ER_Massive_2023.txt').transpose()
        l, = ax.plot(m_lim*1e3, x_lim, '--', color='#CA778C', lw=lw, label='PandaX-4T 2023')
        handles.append(l)
        ax.fill_between(m_lim*1e3, x_lim, 1e-22, facecolor='whitesmoke')
        
        m_lim, x_lim = np.loadtxt(curves_dir + 'Freeze_out_Scalar_ER_Massive.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '-', color='orange', lw=lw*2, label='Freeze-out Scalar')
        handles.append(l)

        #m_lim, x_lim = np.loadtxt(curves_dir + 'XENON1T_S2Only_Solar_Reflected_ER_Massive.txt').transpose()
        #ax.plot(m_lim, x_lim, '--', lw=1.5, label='XENON1T S2only (Solar Refl.)')
        #m_lim, x_lim = np.loadtxt(curves_dir + 'SENSEI_SNOLAB_Solar_Reflection_ER_Massive.txt').transpose()
        #ax.plot(m_lim, x_lim, '--', lw=1.5, label='SENSEI SNOLAB (Solar Refl.)')

        if leg:
            leg1 = ax.legend(handles=handles, **leg_params)
            ax.add_artist(leg1)

    elif ER_model == 'massless':
        
        m_lim, x_lim = np.loadtxt(curves_dir + 'DAMIC_LBC_ER_Massless_2025.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='magenta', lw=lw, label='LBC (DAMIC) 2025', alpha=0.5)
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e-22, facecolor='whitesmoke')

        m_lim_1, x_lim_1 = np.loadtxt(curves_dir + 'DAMIC_ER_Massless_2023.txt').transpose()
        m_lim_2, x_lim_2 = np.loadtxt(curves_dir + 'DAMIC_ER_Massless_2024.txt').transpose()
        m_lim = np.geomspace(min(m_lim_1[0], m_lim_2[0]), max(m_lim_1[-1], m_lim_2[-1]), 150)
        x_lim_1_adj = log_interp(m_lim, m_lim_1, x_lim_1, left=np.inf, right=np.inf)
        x_lim_2_adj = log_interp(m_lim, m_lim_2, x_lim_2, left=np.inf, right=np.inf)
        x_lim = np.minimum(x_lim_1_adj, x_lim_2_adj)
        l, = ax.plot(m_lim, x_lim, '--', color='#CFC0E5', lw=lw, label='DAMIC 2023, 2024')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e-22, facecolor='whitesmoke')

        m_lim, x_lim = np.loadtxt(curves_dir + 'SENSEI_SNOLAB_ER_Massless.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='#a4dbed', lw=lw, label='SENSEI 2025')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e-22, facecolor='whitesmoke')

        m_lim, x_lim = np.loadtxt(curves_dir + 'Freeze_in_ER_Massless.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '-', color='#FF9039', lw=lw*2, label='Freeze-in')
        handles.append(l)

        if leg:
            leg1 = ax.legend(handles=handles, **leg_params)
            ax.add_artist(leg1)

    elif ER_model == 'absorption':

        m_lim, x_lim = np.loadtxt(curves_dir + 'DAMIC_Absorption_2025.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='magenta', lw=lw, label='LBC (DAMIC) 2025')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e0, facecolor='whitesmoke')

        m_lim, x_lim = np.loadtxt(curves_dir + 'SENSEI_Absorption_2025.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='#CFC0E5', lw=lw, label='SENSEI 2025')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e0, facecolor='whitesmoke')

        m_lim, x_lim = np.loadtxt(curves_dir + 'XENON1T_Absorption_2019.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='#a4dbed', lw=lw, label='XENON1T 2019')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e0, facecolor='whitesmoke')

        m_lim, x_lim = np.loadtxt(curves_dir + 'XENONnT_Absorption_2022.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '-', color='#FF9039', lw=lw*2, label='XENONnT 2022')
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e0, facecolor='whitesmoke')

        m_lim, x_lim = np.loadtxt(curves_dir + 'XENONnT_Absorption_2024_Lower.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='#CA778C', lw=lw, label='XENONnT 2024')
        handles.append(l)

        m_lim, x_lim = np.loadtxt(curves_dir + 'XENONnT_Absorption_2024_Upper.txt').transpose()
        l, = ax.plot(m_lim, x_lim, '--', color='#CA778C', lw=lw)
        handles.append(l)
        ax.fill_between(m_lim, x_lim, 1e0, facecolor='whitesmoke')

        if leg:
            leg1 = ax.legend(handles=handles, **leg_params)
            ax.add_artist(leg1)

    else:

        if cosmic:

            m_lim = ML.Data_Cosmic_ray[:,0]
            x_lim = ML.Data_Cosmic_ray[:,1]
            ax.plot(m_lim, x_lim, linewidth=0.5, color='gray', linestyle='-.', label="Cosmic ray scattered DM")
            ax.fill_between(m_lim, np.ones_like(x_lim)*1e-22, x_lim, color='whitesmoke')

            m_lim = ML.Data_Milky_way_satellites_2021[:,0]
            x_lim = ML.Data_Milky_way_satellites_2021[:,1]
            ax.plot(m_lim, x_lim, linewidth=0.5, color='gray', linestyle='--', label="Milky Way satellites")
            ax.fill_between(m_lim, np.ones_like(x_lim)*1e-22, x_lim, color='whitesmoke')

            m_lim = ML.Data_Ly_alpha_2022[:,0]
            x_lim = ML.Data_Ly_alpha_2022[:,1]
            ax.plot(m_lim, x_lim, linewidth=0.5, color='gray', linestyle=':', label="Ly-alpha forest")
            ax.fill_between(m_lim, np.ones_like(x_lim)*1e-22, x_lim, color='whitesmoke')
            
            m_lim = ML.Data_CMB[:,0]
            x_lim = ML.Data_CMB[:,1]
            ax.plot(m_lim, x_lim, linewidth=0.5, color='gray', linestyle='-', label="CMB")
            ax.fill_between(m_lim, np.ones_like(x_lim)*1e-22, x_lim, color='whitesmoke')
            
        if migdal:
            
            m_lim, x_lim = np.loadtxt(curves_dir + 'DAMIC_LBC_Migdal_2025.txt').transpose()
            l, = ax.plot(m_lim*1e3, x_lim, lw=lw, color='#CFC0E5', linestyle='--', label='LBC (DAMIC) Migdal 2025')
            handles.append(l)
            ax.fill_between(m_lim*1e3, x_lim, 1e-22, facecolor='#ad8150', alpha=0.1)
            
            m_lim, x_lim = np.loadtxt(curves_dir + 'SENSEI_Migdal_NR.txt').transpose()
            l, = ax.plot(m_lim, x_lim, lw=lw, color='#CE977A', linestyle='--', label='SENSEI Migdal NR')
            handles.append(l)
            ax.fill_between(m_lim, x_lim, 1e-22, facecolor='#ad8150', alpha=0.1)
            
            m_lim, x_lim = np.loadtxt(curves_dir + 'Darkside50_Migdal_2023.txt').transpose()
            l, ax.plot(m_lim*1e3, x_lim, lw=lw, color='#a4dbed', linestyle='--', label='Darkside-50 Migdal 2023')
            handles.append(l)
            ax.fill_between(m_lim*1e3, x_lim, 1e-22, facecolor='#ad8150', alpha=0.1)
            
            m_lim, x_lim = np.loadtxt(curves_dir + 'PandaX4T_Migdal_2023.txt').transpose()
            l, = ax.plot(m_lim*1e3, x_lim, lw=lw, color='#CA778C', linestyle='--', label='PandaX4T Migdal 2023')
            handles.append(l)
            ax.fill_between(m_lim*1e3, x_lim, 1e-22, facecolor='#ad8150', alpha=0.1)
            
            m_lim, x_lim = np.loadtxt(curves_dir + 'XENON1T_Migdal_2019.txt').transpose()
            l, ax.plot(m_lim*1e3, x_lim, lw=lw, color='#FF9039', linestyle='--', label='XENON1T Migdal 2019')
            handles.append(l)
            ax.fill_between(m_lim*1e3, x_lim, 1e-22, facecolor='#ad8150', alpha=0.1)

            ### SuperCDMS Migdal 2023 ###
            m_lim, x_lim = np.loadtxt(curves_dir + 'SuperCDMS_Migdal_2023.txt').transpose()
            m_lim_overburden, x_lim_overburden = np.loadtxt(curves_dir + 'SuperCDMS_Migdal_2023_Overburden.txt').transpose()
            
            m_range = np.geomspace( min(m_lim[0], m_lim_overburden[0]), max(m_lim[-1], m_lim_overburden[-1]), 150 )
            l, = ax.plot(m_range*1e3, log_interp(m_range, m_lim, x_lim), lw=lw, color='magenta', linestyle='--', label='SuperCDMS Migdal 2023')
            handles.append(l)
            ax.plot(m_range*1e3, log_interp(m_range, m_lim_overburden, x_lim_overburden), lw=lw, color='magenta', linestyle='--')
            ax.plot([m_lim[0]*1e3, m_lim[0]*1e3], [x_lim[0], x_lim_overburden[0]], lw=lw, color='magenta', linestyle='--')
            ax.fill_between(m_range*1e3,
                            log_interp(m_range, m_lim, x_lim),
                            log_interp(m_range, m_lim_overburden, x_lim_overburden),
                            facecolor='#ad8150', alpha=0.1)
            
        if direct:
        
            ### SuperCDMS CPD ###
            m_lim, x_lim = ML.cpdlimit_down[:,0], ML.cpdlimit_down[:,1]
            m_lim_overburden, x_lim_overburden = ML.cpdlimit_up[:,0], ML.cpdlimit_up[:,1]
            l, = ax.plot(m_lim*1e3, x_lim, lw=lw, color='#CE977A', linestyle='-', label='SuperCDMS CPD')
            handles.append(l)
            ax.plot(m_lim_overburden*1e3, x_lim_overburden, lw=lw, color='#CE977A', linestyle='-')
            ax.plot([m_lim[0]*1e3, m_lim[0]*1e3], [x_lim[0], x_lim_overburden[0]], lw=lw, color='#CE977A', linestyle='-')
            ax.fill_between(m_lim*1e3, x_lim, x_lim_overburden, facecolor='#dbdbdb', alpha=0.75)

            ### CRESST-III 2019 ###
            m_lim, x_lim = np.loadtxt(curves_dir + 'CRESST_III_2019.txt').transpose()
            l, = ax.plot(m_lim*1e3, x_lim, '-', color='#a4dbed', lw=lw, label='CRESST-III 2019')
            handles.append(l)
            ax.fill_between(m_lim*1e3, x_lim, 1e-10, facecolor='#dbdbdb', alpha=0.75)

            ### CRESST-III 2024 ###
            m_lim, x_lim = np.loadtxt(curves_dir + 'CRESST_III_2024_Extended.txt').transpose()
            m_lim_overburden, x_lim_overburden = np.loadtxt(curves_dir + 'CRESST_III_2024_Overburden.txt').transpose()

            # Find point of intersection between limit and overburden
            m_range = np.geomspace( max(m_lim[0], m_lim_overburden[0]), min(m_lim[-1], m_lim_overburden[-1]), 150 )
            idx_intersect = np.argmin(
                np.abs(
                    np.log(np.interp(m_range, m_lim_overburden, x_lim_overburden, left=0., right=0.)) - 
                        np.log(np.interp(m_range, m_lim, x_lim, left=0., right=0.))
                )
            )

            # Plot excluded region
            m_plot = m_range[idx_intersect:]
            l, = ax.plot(m_plot*1e3, log_interp(m_plot*1e3, m_lim*1e3, x_lim, left=0., right=0.), linestyle='-', color='#CFC0E5', lw=lw, label='CRESST-III 2024')
            handles.append(l)
            ax.plot(m_plot*1e3, log_interp(m_plot*1e3, m_lim_overburden*1e3, x_lim_overburden, left=0., right=0.), linestyle='-', color='#CFC0E5', lw=lw)
            ax.fill_between(m_plot*1e3,
                            log_interp(m_plot*1e3, m_lim*1e3, x_lim, left=0., right=0.),
                            log_interp(m_plot*1e3, m_lim_overburden*1e3, x_lim_overburden, left=0., right=0.),
                            facecolor='#dbdbdb', alpha=0.75)

            ### TESSERACT 2025 ###
            m_lim, x_lim = np.loadtxt(curves_dir + 'TESSERACT_2025_Limit.txt').transpose()
            m_lim_overburden, x_lim_overburden = np.loadtxt(curves_dir + 'TESSERACT_2025_Overburden.txt').transpose()
            m_lim_pileup, x_lim_pileup = np.loadtxt(curves_dir + 'TESSERACT_2025_Pileup.txt').transpose()
            
            # Find point of intersection between limit and pileup
            m_range = np.geomspace( min(m_lim[0], m_lim_overburden[0], m_lim_pileup[0]), min(m_lim[-1], m_lim_overburden[-1], m_lim_pileup[-1]), 150 )
            idx_intersect = np.argmin(
                np.abs(
                    np.log(np.interp(m_range, m_lim_pileup, x_lim_pileup, left=0., right=0.)) - 
                        np.log(np.interp(m_range, m_lim, x_lim, left=0., right=0.))
                )
            )
            
            # Plot excluded region
            m_plot = m_range[idx_intersect:]
            l, = ax.plot(m_plot*1e3, log_interp(m_plot*1e3, m_lim*1e3, x_lim, left=0., right=0.), linestyle='-', color='#CA778C', lw=lw, label='TESSERACT 2025')
            handles.append(l)
            ax.plot(m_plot*1e3, log_interp(m_plot*1e3, m_lim_pileup*1e3, x_lim_pileup, left=0., right=0.), linestyle='-', color='#CA778C', lw=lw)
            ax.fill_between(m_plot*1e3,
                            log_interp(m_plot*1e3, m_lim*1e3, x_lim, left=0., right=0.),
                            log_interp(m_plot*1e3, m_lim_pileup*1e3, x_lim_pileup, left=0., right=0.),
                            facecolor='#dbdbdb', alpha=0.75)

            ### LZ 2024 ###
            m_lim, x_lim = np.loadtxt(curves_dir + 'LZ_SI_2024.txt').transpose()
            l, = ax.plot(m_lim*1e3, x_lim, '-', color='pink', lw=lw, label='LZ 2024')
            handles.append(l)
            ax.fill_between(m_lim*1e3, x_lim, 1e-10, facecolor='#dbdbdb', alpha=0.75)

            ### XENONnT 2024 ###
            m_lim, x_lim = np.loadtxt(curves_dir + 'XENONnT_SI_2024.txt').transpose()
            l, = ax.plot(m_lim*1e3, x_lim, '-', color='green', lw=lw, label='XENONnT 2024')
            handles.append(l)
            ax.fill_between(m_lim*1e3, x_lim, 1e-10, facecolor='#dbdbdb', alpha=0.75)
            
            ### PandaX4T 2023 ###
            m_lim, x_lim = np.loadtxt(curves_dir + 'PandaX4T_S2_Only_2023.txt').transpose()
            l, = ax.plot(m_lim*1e3, x_lim, '-', color='orange', lw=lw, label='PandaX4T 2023')
            handles.append(l)
            ax.fill_between(m_lim*1e3, x_lim, 1e-10, facecolor='#dbdbdb', alpha=0.75)
            
        if leg:
            leg1 = ax.legend(handles=handles, **leg_params)
            ax.add_artist(leg1)
            
    return fig, ax, handles

def load_template(
    filter_file = 'filter_file.hdf',
    channel = 'Mv3025pcBigFinsLeft|Mv3025pcBigFinsRight',
    base_energy_resolution = 0.3977,  # eV; this is the resolution of the 1x1 shared template from Run 57
    GaAs_energy_resolution = 0.156,
    PD_energy_resolution = 0.089,
    E_th_PD = 1.,
    E_th_GaAs = 5.,
    ):
    """
        Function to load template and compute response/noise
    """
    ### Load template
    template = detprocess.Template(verbose=True)
    template.load_hdf5(get_cache_path(filter_file))
    
    # Load shared 2x1 template to get the energy scale right, but we only use the left channel template shape and PSD
    
    
    templates_td, t_sec_template, metadata = template.get_template(channel, return_metadata=True, tag='shared')
    templates_td = np.reshape(templates_td[0,0], (1,1,templates_td.shape[-1]))
    _, _, f_frequencies = templates_td.shape
    fs = metadata['sample_rate']
    pretrig = metadata['nb_pretrigger_samples']
    
    csd, freqs_csd, _ = template.get_csd(channel, fold=False, return_metadata=True, tag='default')
    csd = np.reshape(csd[0,0], (1,1,csd.shape[-1]))
    
    
    ### Convert to multichannel format
    
    # Rescale template so that the energy resolution per channel is some assumed value
    
    E_exposure_GaAs = E_th_GaAs # The energy cutoff to calculate effective exposure. Can be the threshold or some other value.
    E_exposure_PD = E_th_PD
    
    template_OF = np.zeros((3, 3, f_frequencies))
    template_OF[0, 0, :] = np.copy(templates_td)
    template_OF[1, 1, :] = np.copy(templates_td) 
    template_OF[2, 2, :] = np.copy(templates_td)
    
    csd_OF = np.zeros((3, 3, f_frequencies))
    csd_OF[0, 0, :] = np.copy(csd) * (GaAs_energy_resolution / base_energy_resolution)**2
    csd_OF[1, 1, :] = np.copy(csd) * (PD_energy_resolution / base_energy_resolution)**2
    csd_OF[2, 2, :] = np.copy(csd) * (PD_energy_resolution / base_energy_resolution)**2

    return template_OF, csd_OF, metadata
    

def load_simulation(
    run_length_hours = 3,
    simulation_dir = 'GatewayReview/gaas_requirement_ER/',
    E_max_PD_keV = 20e-3,
    E_max_GaAs_keV = 100e-3,
    LEE_improvement_PD = 1., # Per-volume improvement factor for the PD
    LEE_volume_scale_PD = 49 * 0.3 / 100, # Ratio of PD volume to Run 57 Si volume
    LEE_improvement_GaAs = 1., # Per-volume improvement factor for the GaAs
    LEE_volume_scale_GaAs = 49 * 0.5 / 100, # Ratio of GaAs volume to Run 57 Si volume
    sim_trace_length = 1_250_000 * 10,
    file_pattern = "my_simulation_2fold*.pkl",
    fs = 1250000,
    f_frequencies = 10,
    ):
    sim_trace_length_s = sim_trace_length / fs  # Convert to seconds
    n_simulations_per_run = int(np.floor(run_length_hours * 3600 / sim_trace_length_s))
    n_traces_per_sim = int(sim_trace_length / f_frequencies),
    E0_full = []
    E1_full = []
    E2_full = []
    sim_number_full = []
    trigger_time_full = []
    delta_delta_full = []
    detected_full = []
    n_runs = 0

    available_files = glob(simulation_dir+file_pattern)
    for fn in tqdm(available_files):
        with open(fn, 'rb') as f:
            results = pickle.load(f)

            E0_full += [e for result in results for e in result[0]]
            E1_full += [e for result in results for e in result[1]]
            E2_full += [e for result in results for e in result[2]]
            sim_number_full += [e for result in results for e in result[3]]
            trigger_time_full += [e for result in results for e in result[4]]
            delta_delta_full += [e for result in results for e in result[5]]
            detected_full += [e for result in results for e in result[6]]
            n_runs +=1
    print(f'Loaded {n_runs} runs of simulation results.')
    E0_full = np.array(E0_full)
    E1_full = np.array(E1_full)
    E2_full = np.array(E2_full)
    sim_number_full = np.array(sim_number_full)
    trigger_time_full = np.array(trigger_time_full)
    delta_delta_full = np.array(delta_delta_full)
    detected_full = np.array(detected_full)

    return E0_full, E1_full, E2_full, sim_number_full, trigger_time_full, delta_delta_full, detected_full, n_runs
    

    
def print_expectations(
    E0_full,
    E1_full,
    E_exposure_PD,
    E_exposure_GaAs,
    LEE_file,
    LEE_improvement_PD,
    LEE_improvement_GaAs,
    LEE_volume_scale_PD,
    LEE_volume_scale_GaAs,
    E_max_PD_keV,
    E_max_GaAs_keV,
    n_runs,
    ):
    ### Calculate the effective exposure in the GaAs and PD detectors, based on the observed triggers


    # Exposure in a single PD detector
    n_triggers_obs_PD1 = sum(E1_full > E_exposure_PD)
    print(f'\nNumber of triggers observed in PD 1: {n_triggers_obs_PD1}')
    
    E_above_th_PD1 = np.geomspace(E_exposure_PD * 1e-3, E_max_PD_keV, 250)
    dRdE_above_th_PD1 = get_LEE_bkgd(scale_by=1/LEE_improvement_PD, LEE_file = LEE_file)(E_above_th_PD1)
    R_above_th_PD1 = np.trapz(dRdE_above_th_PD1, E_above_th_PD1)
    print(f'Expected rate of triggers in PD 1 above threshold: {R_above_th_PD1:.2e} Hz')
    
    eff_exposure_PD1_hours = n_triggers_obs_PD1 / R_above_th_PD1 / 3600
    print(f'Effective exposure in PD: {eff_exposure_PD1_hours:.2f} hours (no volume scaling or LEE improvement)')
    
    # Exposure in the GaAs detector
    n_triggers_obs_GaAs = sum(E0_full > E_exposure_GaAs)
    print(f'\nNumber of triggers observed in GaAs: {n_triggers_obs_GaAs}')
    
    E_above_th_GaAs = np.geomspace(E_exposure_GaAs * 1e-3, E_max_GaAs_keV, 250)
    dRdE_above_th_GaAs = get_LEE_bkgd(scale_by=1/LEE_improvement_GaAs, LEE_file = LEE_file)(E_above_th_GaAs)
    R_above_th_GaAs = np.trapz(dRdE_above_th_GaAs, E_above_th_GaAs)
    print(f'Expected rate of triggers in GaAs above threshold: {R_above_th_GaAs:.2e} Hz')
    
    eff_exposure_GaAs_hours = n_triggers_obs_GaAs / R_above_th_GaAs / 3600
    print(f'Effective exposure in GaAs: {eff_exposure_GaAs_hours / 24:.2f} days (no volume scaling or LEE improvement)')
    
    # Apply the LEE and volume scaling
    print('\n##########################\n')
    eff_exposure_PD1_hours /= LEE_volume_scale_PD
    eff_exposure_per_run_PD1_hours = eff_exposure_PD1_hours / n_runs
    eff_exposure_GaAs_hours /= LEE_volume_scale_GaAs
    eff_exposure_per_run_GaAs_hours = eff_exposure_GaAs_hours / n_runs
    print(f'Effective exposure in PD after LEE and volume scaling: {eff_exposure_PD1_hours:.2f} hours ({eff_exposure_per_run_PD1_hours:.2f} hours per run)')
    print(f'Effective exposure in GaAs after LEE and volume scaling: {eff_exposure_GaAs_hours:.2f} hours ({eff_exposure_per_run_GaAs_hours:.2f} hours per run)')


# statistics/sensi results:

def counting_ul_sensitivity(expected_background, cl=0.9):
    """
        return median no-signal upper limit of a counting experiment
    """
    assert 0<=expected_background
    if expected_background>100:
        return expected_background + np.sqrt(expected_background)*sps.norm().isf(1-cl)
    else:
        n_median = sps.poisson(expected_background).median()
        f = lambda ul: sps.poisson(ul).cdf(n_median) - (1.-cl)
        return brentq(f, 0, 120)

def get_gaas_background_spectrum(
    LEE_file,
    nsigma,
    baseline_res_eV,
    volume_cm3, 
    LEE_improvement,
    t_days,
    sigma0,
    target_mass_kg
    ):
    raise NotImplementedError 

def get_gaas_absorbtion_spectrum(
    LEE_file,
    nsigma,
    baseline_res_eV,
    volume_cm3, 
    LEE_improvement,
    t_days,
    sigma0,
    target_mass_kg
    ):
    raise NotImplementedError 

    
def get_gaas_absorbtion_expectations(
    LEE_file,
    nsigma,
    volume_cm3, 
    LEE_improvement,
    t_days,
    sigma0,
    target_mass_kg,
    wimp_mass_MeV, 
    baseline_res_eV,
    ):
    """
    compute signal and background expectations 
    TODO 1) i separate spectra and expectation value calculations
    """
    LEE_filename = get_cache_path(LEE_file)
    E_arr_template, dRdE_bkgd_arr_template, _ = np.loadtxt(LEE_filename, unpack=True)
    
    # Get energy range for this mass
    E_th_keV = nsigma * baseline_res_eV * 1e-3  # threshold in keV
    E_min_sig_keV = wimp_mass_MeV * 1e6 - 3 * baseline_res_eV * 1e-3
    E_max_sig_keV = wimp_mass_MeV * 1e6 + 3 * baseline_res_eV * 1e-3
    E_th_updated_keV = max(E_th_keV, E_min_sig_keV, 0.)

    
    print("E_th_updated_keV, E_max_sig_keV")
    print(E_th_updated_keV, E_max_sig_keV)
    if E_max_sig_keV < E_th_updated_keV:
        print(f"Mass {wimp_mass_MeV*1e12:.3e} meV is below threshold for detection. Skipping...")
        return wimp_mass_MeV, [np.inf], lambda x:np.nan
    else:
        E_arr = np.linspace(E_th_updated_keV, E_max_sig_keV, 10_000)
    
    # Scale background template to exposure time and target mass
    dRdE_bkgd_arr = np.interp(E_arr, E_arr_template, dRdE_bkgd_arr_template)  # interpolate to new energy range
    dRdE_bkgd_arr *= (volume_cm3 / 0.1) # scale to 0.1 cm3 (Run 57 Si)
    dRdE_bkgd_arr /= LEE_improvement # Improvement factor for gaas
    dRdE_bkgd_arr *= t_days # scale to exposure time; new units are cts/keV

    # Load signal template for bosonic absorption
    dRdE_signal_fun = darklim.elf.get_dRdE_lambda_GaAs_absorption(mX_eV=wimp_mass_MeV*1e9, kappa=sigma0, res_eV=baseline_res_eV, suppress_darkelf_output=False)
    dRdE_signal_arr = dRdE_signal_fun(E_arr)
    dRdE_signal_arr *= (t_days * target_mass_kg) # Scale to exposure; new units are cts/keV
    
    R_signal = np.trapz(dRdE_signal_arr, E_arr)  # total signal counts
    R_bkgd = np.trapz(dRdE_bkgd_arr, E_arr)      # total background counts
    
    return R_bkgd, R_signal, dRdE_signal_fun



def get_effective_exposure(
                           E0_full, 
                           E_exposure_GaAs,  # The energy cutoff to calculate effective exposure. Can be the threshold or some other value.
                           E_max_GaAs_keV, 
                           LEE_improvement_GaAs, 
                           E1_full, 
                           E_exposure_PD,  # The energy cutoff to calculate effective exposure. Can be the threshold or some other value.
                           E_max_PD_keV, 
                           LEE_improvement_PD, 
                           LEE_file,
                           LEE_volume_scale_GaAs,
                           LEE_volume_scale_PD,
                          ):
    ### Calculate the effective exposure in the GaAs and PD detectors, based on the observed triggers
    # Exposure in a single PD detector
    
    n_triggers_obs_PD1 = sum(E1_full > E_exposure_PD)
    #print(f'\nNumber of triggers observed in PD 1: {n_triggers_obs_PD1}')
    
    E_above_th_PD1 = np.geomspace(E_exposure_PD * 1e-3, E_max_PD_keV, 250)
    dRdE_above_th_PD1 = get_LEE_bkgd(LEE_file=LEE_file,scale_by=1/LEE_improvement_PD)(E_above_th_PD1)
    R_above_th_PD1 = np.trapz(dRdE_above_th_PD1, E_above_th_PD1)
    #print(f'Expected rate of triggers in PD 1 above threshold: {R_above_th_PD1:.2e} Hz')
    
    eff_exposure_PD1_hours = n_triggers_obs_PD1 / R_above_th_PD1 / 3600
    #print(f'Effective exposure in PD: {eff_exposure_PD1_hours:.2f} hours (no volume scaling or LEE improvement)')
    
    # Exposure in the GaAs detector
    n_triggers_obs_GaAs = sum(E0_full > E_exposure_GaAs)
    #print(f'\nNumber of triggers observed in GaAs: {n_triggers_obs_GaAs}')
    
    E_above_th_GaAs = np.geomspace(E_exposure_GaAs * 1e-3, E_max_GaAs_keV, 250)
    dRdE_above_th_GaAs = get_LEE_bkgd(LEE_file=LEE_file,scale_by=1/LEE_improvement_GaAs)(E_above_th_GaAs)
    R_above_th_GaAs = np.trapz(dRdE_above_th_GaAs, E_above_th_GaAs)
    #print(f'Expected rate of triggers in GaAs above threshold: {R_above_th_GaAs:.2e} Hz')
    
    eff_exposure_GaAs_hours = n_triggers_obs_GaAs / R_above_th_GaAs / 3600
    #print(f'Effective exposure in GaAs: {eff_exposure_GaAs_hours / 24:.2f} days (no volume scaling or LEE improvement)')
    
    # Apply the LEE and volume scaling
    #print('\n##########################\n')
    eff_exposure_PD1_hours /= LEE_volume_scale_PD
    #eff_exposure_per_run_PD1_hours = eff_exposure_PD1_hours / n_runs
    eff_exposure_GaAs_hours /= LEE_volume_scale_GaAs
    #eff_exposure_per_run_GaAs_hours = eff_exposure_GaAs_hours / n_runs
    #print(f'Effective exposure in PD after LEE and volume scaling: {eff_exposure_PD1_hours:.2f} hours ({eff_exposure_per_run_PD1_hours:.2f} hours per run)')
    #print(f'Effective exposure in GaAs after LEE and volume scaling: {eff_exposure_GaAs_hours:.2f} hours ({eff_exposure_per_run_GaAs_hours:.2f} hours per run)')

    return eff_exposure_PD1_hours, eff_exposure_GaAs_hours

def plot_background_distribution(constants,
                                 E0_full, E1_full, delta_delta_full, 
                                 eff_exposure_GaAs_hours, 
                                 eff_exposure_PD_hours, 
                                n_runs,
                                run_length_hours,
                                LEE_file, 
                                LEE_improvement_GaAs,
                                dirname=".", ):
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    keV_bins = np.arange(0, 50, 0.1) * 1e-3 
    E_th_PD = constants["E_th_PD"]
    E_th_GaAs = constants["E_th_GaAs"]
    
    eff_exposure_per_run_PD_hours = eff_exposure_PD_hours / n_runs
    eff_exposure_per_run_GaAs_hours = eff_exposure_GaAs_hours / n_runs

    single_pixel = E0_full > E_th_GaAs
    duration_days = eff_exposure_GaAs_hours / 24
    ax.hist(E0_full[single_pixel]*1e-3, bins=keV_bins,
        weights=np.full(sum(single_pixel), 1/np.diff(keV_bins)[0]/duration_days),
        histtype='step', color='C0', lw=2,
        label=f'1x1 triggers in GaAs (E > {E_th_GaAs:.2f} eV)')
    hist_E_E0_single, bin_edges = np.histogram(E0_full[single_pixel]*1e-3, bins=keV_bins,
                                 weights=np.full(sum(single_pixel), 1/np.diff(keV_bins)[0]/duration_days))
    R_single_GaAs = sum(hist_E_E0_single * np.diff(bin_edges))
    print(f'Effective 1-fold exposure in GaAs: {duration_days:.2f} days')

    single_pixel = E1_full > E_th_PD
    duration_days = eff_exposure_PD_hours / 24
    ax.hist(E1_full[single_pixel]*1e-3, bins=keV_bins,
        weights=np.full(sum(single_pixel), 1/np.diff(keV_bins)[0]/duration_days),
        histtype='step', color='C1', lw=2,
        label=f'1x1 triggers in PD 1 (E > {E_th_PD:.2f} eV)')
    hist_E_E1_single, bin_edges = np.histogram(E1_full[single_pixel]*1e-3, bins=keV_bins,
                                 weights=np.full(sum(single_pixel), 1/np.diff(keV_bins)[0]/duration_days))
    R_single_PD_A = sum(hist_E_E1_single * np.diff(bin_edges))
    print(f'Effective 1-fold exposure in PD 1: {duration_days:.2f} days')
    plt.xlabel("Energy [keV]")
    plt.ylabel("Events [/keV/d]")

    detected = (E0_full > E_th_GaAs) * (E1_full > E_th_PD) * (delta_delta_full < 9)
    detected_overall = np.copy(detected)
    duration_days = eff_exposure_per_run_PD_hours * eff_exposure_per_run_GaAs_hours / run_length_hours / 24 * n_runs
    ax.hist(E0_full[detected]*1e-3, bins=keV_bins,
            weights=np.full(sum(detected), 1/np.diff(keV_bins)[0]/duration_days),
            histtype='step', color='C3', lw=2,
            label='2-fold coincident Triggers, GaAs')
    ax.hist(E1_full[detected]*1e-3, bins=keV_bins,
            weights=np.full(sum(detected), 1/np.diff(keV_bins)[0]/duration_days),
            histtype='step', color='C4', lw=2,
            label='2-fold coincident Triggers, PD')
    
    hist_E_2fold_PD, bin_edges = np.histogram(E1_full[detected]*1e-3, bins=keV_bins,
                                     weights=np.full(sum(detected), 1/np.diff(keV_bins)[0]/duration_days))
    hist_E_2fold_GaAs, bin_edges = np.histogram(E0_full[detected]*1e-3, bins=keV_bins,
                                     weights=np.full(sum(detected), 1/np.diff(keV_bins)[0]/duration_days))
    R_coincident_2fold = sum(hist_E_2fold_GaAs * np.diff(bin_edges))
    
    print(f'Effective 2-fold PD exposure: {duration_days:.2f} days')
    
    # Export to a file
    E_save = 0.5 * (keV_bins[1:] + keV_bins[:-1])
    np.savetxt(dirname + f'LEE_2fold_PD_{E_th_PD:.2f}_GaAs_{E_th_GaAs:.1f}.txt',
               np.column_stack((E_save, hist_E_2fold_GaAs)),
               header='Energy [keV]     dRdE GaAs [1/keV/day]',
               fmt='%.4e')

    E_arr = np.linspace(0.1, 50, 200) * 1e-3
    dRdE_arr = get_LEE_bkgd(LEE_file=LEE_file,scale_by=1/LEE_improvement_GaAs)(E_arr)
    ax.plot(E_arr, dRdE_arr * 86400, color='k', label='Single-pixel LEE Template', lw=2)
    
    ax.set_yscale('log')
    ax.set_xlabel('Energy in Pixel [keV]')
    ax.set_ylabel('Rate [1/keV/day]')
    ax.set_ylim([1e-2, 1e8])
    ax.set_xlim([0, 30e-3])
    
    ax.legend(loc='upper right', fontsize=14)
    fig.tight_layout()
    
    print(f'\nRates [1/day/module]:')
    print(f'2-fold in GaAs, with either PD: {R_coincident_2fold:.2e}')
    print(f'Number of coincident events: {sum(detected):,}')
    

def get_gaas_signal(
    m_DM_eV, 
    sigma, 
    mediator, #one of NR, massive, massless (ER), absorption
    E_th_PD,
    E_res_PD,
    E_th_GaAs,
    E_res_GaAs,
    collection_efficiency,
    baseline_res_eV=None,
    n_PDs=1 # number of photodetectors viewing the xtal
    ):
    """
    Use DarkELF to compute signal expectation, just a wrapper for now to alert that absorption is a special case.
    returns energy and rate _in the GaAs_ 
    """
    if mediator in ["NR","massive","massless"]:
        energies, dRdE = darklim.detector.DM_spectrum_GaAs(
            m_DM_eV=m_DM_eV, 
            sigma=sigma,
            E_th_PD = E_th_PD,
            E_res_PD = E_res_PD,
            E_th_GaAs = E_th_GaAs,
            E_res_GaAs = E_res_GaAs,
            collection_efficiency=collection_efficiency,
        )
    elif mediator=="absorption":
        print("hello")
        assert baseline_res_eV is not None
        dRdE_signal_fun =  darklim.elf.get_dRdE_lambda_GaAs_absorption(mX_eV=m_DM_eV, kappa=sigma, res_eV=baseline_res_eV, suppress_darkelf_output=False)

        energies = m_DM_eV*1e-3 + baseline_res_eV*1e-3*np.linspace(-5,5,101)
        dRdE = dRdE_signal_fun(energies)
    else:
        raise NotImplementedError("this is not an implemented mediator")
    
    return energies, dRdE




    

    