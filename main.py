# Importing libraries and the data set

# General libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage


# This is the dataset to train the model
from mne.datasets.sleep_physionet.age import fetch_data


# This turns all unnecessary outputs off to make code easier to read.
mne.set_log_level("warning")


# -------------------------------------
# Binary awake vs asleep classification

# Constants
annotation_event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

# We want to combine stages 3 and 4 as they are too similar to distinguish
# NEED TO RENAME FIND OUT HOW

event_id = {
    "Awake": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3/4": 4,
    "REM Sleep": 5,
}

frequency_bands = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "beta": [8.5, 11.5],
    "alpha": [11.5, 15.5],
    "sigma": [15.5, 30.5],
}

# Subroutines

# LOAD THE SAMPLE DATASET
# Testing first sample


def loading_data(subject_num):
    [subject_files] = fetch_data(subjects=[subject_num], recording=[1])

    raw_local = mne.io.read_raw_edf(
        subject_files[0], # 0th file, can change
        stim_channel="Event marker",
        infer_types=True,
        preload=True,
        verbose="error"
        )

    # Reading and setting the annotations
    annotations_local = mne.read_annotations(subject_files[1])
    raw_local.set_annotations(annotations_local)

    return raw_local, [subject_files], annotations_local


def crop_annotations(annotations, raw):
    # Cropping 30 minutes before and after sleeping

    sleep_start = annotations[1]["onset"] - 30 * 60
    sleep_end = annotations[-2]["onset"] + 30 * 60

    annotations.crop(sleep_start, sleep_end)
    raw.set_annotations(annotations)


def create_events(raw_local):
    events_local, _ = mne.events_from_annotations(raw_local, event_id=annotation_event_id, chunk_duration=30.0)
    return events_local

def create_epochs(raw_local):
    return ""



# Main program
raw, subject1, annotations = loading_data(0)
crop_annotations(annotations, raw)
events = create_events(raw)

# Plotting raw file
raw.plot(start=60, duration=60)
plt.show()


# Plotting by sleep stage
plot = mne.viz.plot_events(
    events,
    event_id=event_id,
    sfreq=raw.info["sfreq"],
    first_samp=events[0, 0],
)


# Lowest delta is 0.5 and highest sigma is 30.5
raw.filter(l_freq=0.5, h_freq=30.5, fir_design="firwin", skip_by_annotation="edge")


# NEED to customise figure via matplot commands


# Saving the model
""" 
import joblib

joblib.dump(model, 'sleep_eeg_model.joblib')
"""


# INFORMATION
# Code copied from PycharmCE as it is an old version
