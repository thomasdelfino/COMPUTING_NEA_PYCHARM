# Importing libraries and the data set

# General libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage


# Potential Approach 1
from sklearn.ensemble import RandomForestClassifier



# This is the dataset to train the model
from mne.datasets.sleep_physionet.age import fetch_data


# This turns all unnecessary outputs off to make code easier to read.
mne.set_log_level("warning")


# -------------------------------------
# Binary awake vs asleep classification

# Constants

# Group by Awake, Light Sleep (2 and 3), Deep Sleep (3 and 4), and REM (R)
annotation_event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

# We want to combine stages 3 and 4, and 1 and 2, to add a layer of abstraction for the user, and make classification easier

event_id = {
    "Awake": 1,
    "Light Sleep": 2,
    "Deep Sleep": 3,
    "REM Sleep": 4,
}

frequency_bands = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "beta": [8.5, 11.5],
    "alpha": [11.5, 15.5],
    "sigma": [15.5, 30.5],
}

# Subroutines

def loading_data(subject_num, subject_index):
    [subject_files] = fetch_data(subjects=[subject_num], recording=[2])  # Not sure what recording=[1] means

    raw_file = mne.io.read_raw_edf(
        subject_files[subject_index],  # 0th file, can change
        stim_channel="Event marker",
        infer_types=True,
        preload=True,
        verbose="error"
    )

    # Reading and setting the annotations
    annotations_local = mne.read_annotations(subject_files[1])
    raw_file.set_annotations(annotations_local)

    # Not all channels can be available to the user e.g. rectal temperature
    raw_file.drop_channels(['horizontal'])
    raw_file.drop_channels(['oro-nasal'])
    raw_file.drop_channels(['submental'])
    raw_file.drop_channels(['rectal'])

    return raw_file, annotations_local


def crop_annotations(annotations, raw):
    # Cropping 30 minutes before and after sleeping
    # This may need to change depending on

    sleep_start = annotations[1]["onset"] - 30 * 60
    sleep_end = annotations[-2]["onset"] + 30 * 60

    annotations.crop(sleep_start, sleep_end)
    raw.set_annotations(annotations)


def create_events(raw_file):
    # Normally, mne.events_from_annotations returns a tuple so the underscore in "events, _" ensures that only the events is saved to the variable
    events, _ = mne.events_from_annotations(raw_file, event_id=annotation_event_id, chunk_duration=30.0)
    event_id = dict(awake=1, light_sleep=2, deep_sleep=3, rem=4)
    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')

    return events, event_id, picks



def create_epochs(raw_file, events, event_id, picks):

    epochs = mne.Epochs(
        raw_file,
        events = events,
        event_id = event_id,
        tmin = 0.5, # TEMPORARY CHANGE
        tmax = 30.0,
        proj = True,
        picks = picks,
        baseline = None,
        preload = True,
    )

    epochs_train = epochs.copy().crop() # figure out when to crop
    labels = epochs.events[0:len(epochs), -1]
    print(labels)



# This is used to visualise the PSD for each stage
def psd_plot(raw):
    plt.close("all")
    raw.compute_psd(fmin=0.5, fmax=30).plot()
    plt.show()


# Main program
raw, _ = loading_data(1, 0)
events, event_id, picks = create_events(raw)
create_epochs(raw, events, event_id, picks)

psd_plot(raw)

# Plotting raw file





# Saving the model
""" 
import joblib

joblib.dump(model, 'sleep_eeg_model.joblib')
"""


# INFORMATION
# Code copied from PycharmCE as it is an old version
