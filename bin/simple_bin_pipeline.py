#!/usr/bin/env python

from pathlib import Path
import numpy as np
from pprint import pprint
from datetime import datetime
import matplotlib.pyplot as plt

import spikeextractors as se
import spiketoolkit as st
import spikesorters as ss
import spikecomparison as sc
import spikewidgets as sw

from mease_lab_to_nwb.convert_ced.cednwbconverter import quick_write

bin_file = Path(r'../../data/m365_pt1_590-1190secs.bin')
sampling_frequency = 3.003003003003003e+04;
data_type = "int16"
numChan = 64;
# ced_file = Path('/Users/abuccino/Documents/Data/catalyst/heidelberg/ced/m365_pt1_590-1190secs-001.smrx')
# ced_file = Path('D:/CED_example_data/Other example/m365_pt1_590-1190secs-001.smrx')
#probe_file = '../probe_files/cambridge_neurotech_H3.prb'
spikeinterface_folder = bin_file.parent
spikeinterface_folder.mkdir(parents=True, exist_ok=True)


# In[ ]:


recording_prb = '../../mease-lab-to-nwb/probe_files/cambridge_neurotech_H3.prb'


# In[ ]:


# get_ipython().run_line_magic('pinfo', 'se.BinDatRecordingExtractor')


# In[ ]:


# Rhd channels should have already been selected thanks to smrx2bin
recording = se.BinDatRecordingExtractor(bin_file, sampling_frequency, numChan, data_type)


# In[ ]:


recording = se.load_probe_file(recording, recording_prb)


# In[ ]:


property_names = recording.get_shared_channel_property_names()
print(type(property_names))
print(property_names)
print(property_names[2])


# In[ ]:


recording.get_channel_ids();
print(recording.get_channel_ids())


# In[ ]:


recording.get_channel_groups()


# In[ ]:


recording.get_sampling_frequency()


# In[ ]:


recording.get_num_frames()


# In[ ]:


recording.get_channel_groups()


# In[ ]:


tr = recording.get_traces(end_frame=30000, return_scaled=False)


# In[ ]:


plt.figure()
plt.plot(tr[0])
plt.savefig('tr0_1.png')


# In[ ]:


sw.plot_electrode_geometry(recording)


# In[ ]:


print(f"Num channels: {recording.get_num_channels()}")
print(f"Channel ids: {recording.get_channel_ids()}")
print(f"Sampling rate: {recording.get_sampling_frequency()}")
print(f"Duration (s): {recording.get_num_frames() / recording.get_sampling_frequency()}")


# # Get LFPs from MultiChannel Data

# In[ ]:


recording_lfp = recording


# In[ ]:


recording_lfp = st.preprocessing.resample(recording_lfp, resample_rate=1000)


# In[ ]:


apply_filter = True
freq_min_hp = 0.1
freq_max_hp = 300


# In[ ]:


if apply_filter:
    recording_lfp = st.preprocessing.bandpass_filter(recording_lfp, freq_min=freq_min_hp, freq_max=freq_max_hp)
else:
    recording_lfp = recording_lfp


# In[ ]:


tr = recording_lfp.get_traces(end_frame=3000, return_scaled=False)


# In[ ]:


plt.figure()
plt.plot(tr[0])
plt.savefig('tr0_2.png')


# ## Inspect signals

# In[ ]:


w_ts_ap = sw.plot_timeseries(recording, trange=[4, 8])


# ## 2) Pre-processing

# In[ ]:


apply_filter = True
apply_cmr = True
freq_min_hp = 600
freq_max_hp = 3000


# In[ ]:


#get_ipython().run_line_magic('pinfo', 'st.preprocessing.common_reference')


# In[ ]:



st.preprocessing.common_reference(
    recording,
    reference='median',
    groups=None,
    ref_channels=None,
    dtype=None,
    verbose=False,
)


# In[ ]:


if apply_filter:
    recording_processed = st.preprocessing.bandpass_filter(recording, freq_min=freq_min_hp, freq_max=freq_max_hp)
else:
    recording_processed = recording
    
if apply_cmr:
    recording_processed = st.preprocessing.common_reference(recording_processed)


# In[ ]:


# Stub recording for fast testing; set to False for running processing pipeline on entire data
stub_test = False
nsec_stub = 30

if stub_test:
    recording_processed = se.SubRecordingExtractor(
        parent_recording=recording_processed, 
        end_frame=int(nsec_stub*recording_processed.get_sampling_frequency())
    )
    recording_lfp = se.SubRecordingExtractor(
        recording_lfp, 
        end_frame=int(nsec_stub*recording_lfp.get_sampling_frequency())
    )
    
print(f"Original signal length: {recording.get_num_frames()}")
print(f"Processed signal length: {recording_processed.get_num_frames()}")


# In[ ]:


num_frames = recording_processed.get_num_frames()
print(num_frames)


# In[ ]:


w_ts_ap = sw.plot_timeseries(recording_processed, trange=[4, 8])


# ## 3) Run spike sorters

# In[ ]:


#ss.Kilosort3Sorter.set_kilosort3_path("/export/home/lkeegan/CTC/Kilosort")
ss.Kilosort2_5Sorter.set_kilosort2_5_path("/export/home/lkeegan/CTC/Kilosort-2.5")
#ss.Kilosort2Sorter.set_kilosort2_path("/export/home/lkeegan/CTC/Kilosort-2.0")
#ss.installed_sorters()
# ss.IronClustSorter.set_ironclust_path("D:/GitHub/ironclust")


# In[ ]:


sorter_list = [
    "kilosort2_5"
]


# In[ ]:


# Inspect sorter-specific parameters and defaults
for sorter in sorter_list:
    print(f"\n\n{sorter} params description:")
    pprint(ss.get_params_description(sorter))
    print("Default params:")
    pprint(ss.get_default_params(sorter))    


# In[ ]:


# user-specific parameters

sorter_params = dict(kilosort2_5 = dict({'NT': None,
 'car': True,
 'chunk_mb': 500,
 'detect_threshold': 6,
 'freq_min': 150,
 'keep_good_only': False,
 'minFR': 0.1,
 'minfr_goodchannels': 0.1,
 'nPCs': 3,
 'n_jobs_bin': 1,
 'nblocks': 5,
 'nfilt_factor': 4,
 'ntbuff': 64,
 'preclust_threshold': 8,
 'projection_threshold': [10, 4],
 'sig': 20,
 'sigmaMask': 30}))



    #kilosort3 = dict(NT= None,
 #car= True,
 #chunk_mb= 500,
 #detect_threshold= 3.5,
 #freq_min= 600,
 #keep_good_only= False,
 #minFR= 0.1,
 #minfr_goodchannels= 0.1,
 #nPCs= 3,
 #n_jobs_bin= 1,
 #nfilt_factor= 4,
 #ntbuff= 64,
 #preclust_threshold= 8,
 #projection_threshold= [10, 2],
 #sigmaMask= 30), 


#kilosort2_5 = dict(NT= None,
 #car= True,
 #chunk_mb= 250,
 #detect_threshold= 3.5,
 #freq_min= 600,
 #keep_good_only= False,
 #minFR= 0.1,
 #minfr_goodchannels= 0.1,
 #nPCs= 3,
 #n_jobs_bin = 24,
 #nfilt_factor= 4,
 #ntbuff= 64,
 #preclust_threshold= 8,
 #projection_threshold= [10, 2],
 #sigmaMask= 30))
    #sorter_params = {"kilosort3": {}}
#sorter_params = {"kilosort3": {"n_jobs_bin": 8, "chunk_mb": 8000}}
#sorter_params = dict('ironclust': {'detect_threshold': 6}, 'klusta': {}, "herdingspikes": {})


# In[ ]:


ss.available_sorters()


# In[ ]:


#get_ipython().run_line_magic('pinfo', 'ss.run_sorters')


# In[ ]:


sorting_outputs = ss.run_sorters(
    sorter_list=sorter_list, 
    working_folder=spikeinterface_folder / 'ced_si_output',
    recording_dict_or_list=dict(rec0=recording_processed), 
    sorter_params=sorter_params,
    mode="overwrite", # "overwrite" to overwrite # change to "keep" to avoid repeating the spike sorting
    verbose=True,
)


# The `sorting_outputs` is a dictionary with ("rec_name", "sorter_name") as keys.

# In[ ]:


for result_name, sorting in sorting_outputs.items():
    rec_name, sorter = result_name
    print(f"{sorter} found {len(sorting.get_unit_ids())} units")


# ## 4) Post-processing: extract waveforms, templates, quality metrics, extracellular features

# ### Set postprocessing parameters

# In[ ]:


# Post-processing params
postprocessing_params = st.postprocessing.get_common_params()
postprocessing_params["verbose"] = True
postprocessing_params["recompute_info"] = True
pprint(postprocessing_params)


# In[ ]:


# (optional) change parameters
postprocessing_params['memmap'] = False
pprint(postprocessing_params)
#postprocessing_params['max_spikes_per_unit'] = 1000  # with None, all waveforms are extracted


# **Important note for Windows**: on Windows, we currently have some problems with the `memmap` argument. While we fix it, we recommend to set it to `False`.

# ### Set quality metric list

# In[ ]:


# Quality metrics
qc_list = st.validation.get_quality_metrics_list()
print(f"Available quality metrics: {qc_list}")


# In[ ]:


# (optional) define subset of qc
qc_list = ['num_spikes', 'firing_rate', 'presence_ratio', 'isi_violation', 'amplitude_cutoff', 'snr', 'max_drift', 'cumulative_drift', 'silhouette_score', 'isolation_distance', 'l_ratio', 'noise_overlap', 'nn_hit_rate', 'nn_miss_rate']


# ### Set extracellular features

# In[ ]:


# Extracellular features
ec_list = st.postprocessing.get_template_features_list()
print(f"Available EC features: {ec_list}")


# In[ ]:


#get_ipython().run_line_magic('pinfo', 'st.postprocessing')


# ### Postprocess all sorting outputs

# In[ ]:


for result_name, sorting in sorting_outputs.items():
    rec_name, sorter = result_name
    print(f"Postprocessing recording {rec_name} sorted with {sorter}")
    tmp_folder = Path('tmp_ced') / sorter
    tmp_folder.mkdir(parents=True, exist_ok=True)
    
    # set local tmp folder
    sorting.set_tmp_folder(tmp_folder)
     
        
    postprocessing_params['memmap'] = False
    # pprint(postprocessing_params)
    
    
    # compute waveforms
    waveforms = st.postprocessing.get_unit_waveforms(recording_processed, sorting, 
                                                     n_jobs=24, chunk_mb=2000, **postprocessing_params)
    
    # compute templates
    templates = st.postprocessing.get_unit_templates(recording_processed, sorting, n_jobs=24, chunk_mb=2000, **postprocessing_params)
    
    # comput EC features
    ec = st.postprocessing.compute_unit_template_features(recording_processed, sorting, n_jobs=24, chunk_mb=2000,
                                                          feature_names=ec_list, as_dataframe=True, memmap = False)
    ## compute QCs
    #qc = st.validation.compute_quality_metrics(sorting, recording=recording_processed, 
    #                                           metric_names=qc_list, as_dataframe=True, memmap = False)
    
    # export to phy example
    # pprint(postprocessing_params)
    if sorter == "kilosort2_5":
        postprocessing_params['memmap'] = False
       # pprint(postprocessing_params)
        recompute_info = True
       # pprint(postprocessing_params)
        postprocessing_params['memmap'] = False
       # pprint(postprocessing_params)
        phy_folder = spikeinterface_folder / 'phy' / sorter
        phy_folder.mkdir(parents=True, exist_ok=True)
        print("Exporting to phy")
        postprocessing_params['memmap'] = False
        # pprint(postprocessing_params)
        st.postprocessing.export_to_phy(recording_processed, sorting, phy_folder, verbose=True, memmap = False, recompute_info = True, n_jobs=24)
        #st.postprocessing.export_to_phy(recording_processed, sorting, phy_folder, verbose=True, compute_pc_features=False, compute_amplitudes=False, memmap = False, recompute_info = True, n_jobs=24)


# In[ ]:


sorting_kilosort = sorting_outputs[('rec0', 'kilosort2_5')]
print(f"Properties: {sorting_kilosort.get_shared_unit_property_names()}")
print(f"Spikefeatures: {sorting_kilosort.get_shared_unit_spike_feature_names()}")


# ### Load Phy-curated data back to SI

# In[ ]:


get_ipython().system('phy template-gui Z:\\PainData\\m365\\10min\\phy\\kilosort2_5\\params.py')


# In[ ]:


phy_folder = r'Z:\PainData\m365\10min\phy\kilosort2_5'
recording_phy = se.PhyRecordingExtractor(phy_folder)
sorting_curated = se.PhySortingExtractor(phy_folder)
sorting_phy = se.PhySortingExtractor(phy_folder, exclude_cluster_groups=["noise"])
print(f"Units after manual curation: {len(sorting_curated.get_unit_ids())}")


# In[ ]:


good_units = []
for u in sorting_phy.get_unit_ids():
    if sorting_phy.get_unit_property(u, 'quality') == 'good':
        good_units.append(u)
sorting_good = se.SubSortingExtractor(sorting_phy, unit_ids=good_units)
print(good_units)


# In[ ]:


#get_ipython().run_line_magic('pinfo', 'st.curation.threshold_num_spikes')


# In[ ]:


sorting_curated = st.curation.threshold_num_spikes(sorting_curated,
                                                   threshold=50, threshold_sign='less')
print(f"Units after num spikes curation: {len(sorting_curated.get_unit_ids())}")


# In[ ]:


tr_phy = recording_phy.get_traces(end_frame=30000)


# In[ ]:


plt.figure()
plt.plot(tr_phy[0])


# # 7) Quick save to NWB; writes only the spikes and lfp

# ## To complete the full conversion for other types of data, either
# ###    1) Run the external conversion script before this notebook, and append to it by setting overwrite=False below
# ###    2) Run the external conversion script after this notebook, which will append the NWBFile you make here so long as overwrite=False in the external script

# In[ ]:


sorting_outputs


# In[ ]:


# Name your NWBFile and decide where you want it saved
nwbfile_path = r"Z:\PainData\m365\10min\phy\kilosort2_5\m365.nwb"

# Enter Session and Subject information here
session_description = "m365 spikes + LFPs 10min test"

# Manually insert the session start time
session_start = datetime(2020,10, 8)  # (Year, Month, Day)

# Choose the sorting extractor from the notebook environment you would like to write to NWB
# chosen_sorting_extractor = sorting_outputs[('rec0', 'ironclust')]
# chosen_sorting_extractor = sorting_ensemble

#quick_write(
 #   ced_file_path=bin_file,
  #  session_description=session_description,
   # session_start=session_start,
    #save_path=nwbfile_path,
    #sorting=sorting_curated,
    #recording_lfp=None,
    #overwrite=True
#)

#se.NwbRecordingExtractor.write_recording(recording_lfp, 'LFPs.nw')

quick_write(
    ced_file_path=bin_file,
    session_description=session_description,
    session_start=session_start,
    save_path=nwbfile_path,
    sorting = sorting_phy,
    recording_lfp = recording_lfp,
    overwrite=True
)


# In[ ]:





# In[ ]:




