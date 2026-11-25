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

# tmin and tmax represent when epochs should start / finish
tmin = 0.0
tmax = 30.0 # By default, I am setting tmax as 30.0 but this may need to be changed to take into account the total size of files as it may not be cleanly divisible by 30, which might create problems

# Subroutines

def load_data(subject_index, night):
    # Each participant has a subject number, and recording_number (indicating which night is shown)
    # Subject number indicates which subject out of the 74 is selected
    # All variables have the suffix local so they don't shadow global variable

    [subject_files] = fetch_data(subjects=[subject_index], recording=[night]) # Recording represents the night / session that is being loaded, 1st or 2nd

    raw_file_local = mne.io.read_raw_edf(
        subject_files[0], # 0 for the raw file, 1 for the histogram of the participant
        stim_channel="Event marker",
        infer_types=True,
        exclude=['horizontal', 'oro-nasal', 'submental', 'rectal'], # Not all channels in the data can be available to the user e.g. rectal temperature
        preload=True,
        verbose="error"
        )

    # Remove frequencies <0.5 and >30.0 as these are likely not from the participant / will not help the model
    raw_file_local.filter(l_freq=0.5, h_freq=30.0, method="iir") # Basic preprocessing

    # Reading and setting the annotations
    annotations_local = mne.read_annotations(subject_files[night])

    # Crop annotations subroutine
    sleep_start = annotations_local[1]["onset"] - 30 * 4 * 60
    sleep_end = annotations_local[-2]["onset"] + 30 * 4 * 60
    annotations_local.crop(sleep_start, sleep_end)

    raw_file_local.set_annotations(annotations_local)


    return raw_file_local

def crop_annotations(annotations, raw):
    # Cropping 30 minutes before and after sleeping
    # This may need to change depending on

    sleep_start = annotations[1]["onset"] - 30 * 60
    sleep_end = annotations[-2]["onset"] + 30 * 60

    annotations.crop(sleep_start, sleep_end)
    raw.set_annotations(annotations)


def create_events(raw_file_local):
    # Normally, mne.events_from_annotations returns a tuple so the underscore in "events, _" ensures that only the events is saved to the variable
    events_local, _ = mne.events_from_annotations(raw_file_local, event_id=annotation_event_id, chunk_duration=30.0)

    # Create a dictionary linking the number of events to a label
    event_id_local = dict(awake=1, light_sleep=2, deep_sleep=3, rem=4)

    return events_local, event_id_local


def create_epochs(raw_file_local, events_local, event_id_local):

    epochs_local = mne.Epochs(
        raw_file_local,
        events = events_local,
        event_id = event_id_local,
        tmin = tmin, # Set as 0.0
        tmax = tmax, # Set as 30.0
        proj = True,
        picks=['Fpz-Cz', 'Pz-Oz'],  # Event marker not included otherwise the model may be fed the true sleep stage
        baseline = None,
        preload = True,
    )


    epochs_train_local = epochs_local.copy().crop(tmin=0.0, tmax=30.0) # Would need to find out if 0 - 30 is an appropriate crop, and no crop may be needed at all

    labels_local = epochs_local.events[0:len(epochs_local), -1]
    print("Event labels array:")
    print(labels_local)

    return epochs_local, epochs_train_local, labels_local



# This is used to visualise the PSD for each stage
def psd_plot(samples, samples_name):
    print(samples)

    plt.close("all")

    psd_of_samples = samples.compute_psd(method='multitaper', fmin=0.5, fmax=50)
    psd_of_samples.plot(show=False, spatial_colors=False)

    #psd_of_samples.title(samples_name)
    #psd_of_samples.xlim(0, 50)
    #psd_of_samples.ylim(-40, 25)


    plt.show()


# Main program
raw_file= load_data(1, 1)
print("Raw file loaded")

# Insert crop annotations subroutine

events, event_id = create_events(raw_file)
print("Events created from raw file")

epochs, epochs_train, labels = create_epochs(raw_file, events, event_id)
print("Epochs created")


raw_file.info

"""
# Plotting epochs

psd_of_epochs = epochs.compute_psd(method='multitaper', fmin=0.5, fmax=30.0)
psd_of_epochs.plot()
print(psd_of_epochs)

psds, psd_frequencies = psd_of_epochs.get_data(return_freqs=True)
print("Shape of PSDs")
print("(30 second sections, channels, frequencies)")
print(psds.shape)

print("Shape of PSDs frequencies")
print(psd_frequencies.shape)

psd_plot(epochs["deep_sleep"].average(), "Deep sleep")

psd_plot(epochs["awake"].average(), "Awake")
psd_plot(epochs["rem"].average(), "REM")
psd_plot(epochs["light_sleep"].average(), "Light sleep")


# INFORMATION
# Code copied from PycharmCE as it is an old version
"""