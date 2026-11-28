from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
import joblib
import matplotlib.pyplot as plt

mne.set_log_level("warning")

# --------------------- INFO ------------------------

# This code is NOT a part of my NEA just a proof of concept to show that LDA works well with this code





# --------------------- INFO ------------------------




# Group by Awake, Light Sleep (2 and 3), Deep Sleep (3 and 4), and REM (R)
# LET THESE JUST BE AWAKE AND ASLEEP FOR NOW
annotation_event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 2,
    "Sleep stage 3": 2,
    "Sleep stage 4": 2,
    "Sleep stage R": 2,
}

# We want to combine stages 3 and 4, and 1 and 2, to add a layer of abstraction for the user, and make classification easier

event_id = {
    "Awake": 1,
    "Sleep": 2,
}

frequency_bands = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30.5],
}

# tmin and tmax represent when epochs should start / finish
tmin = 0.0
# Changed
# tmax = 30.0 - 1.0 / raw_edf.info["sfreq"]

num_participants = 10

# MY OWN CODE
# 0 - 82 participants BUT 39, 68, 69, 78, 79 are NOT available so my for loops round these numbers must be ammended
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
    sleep_start = annotations_local[1]["onset"] - 30 * 6 * 60 # Change to more than 4 hours before onset?? I don't think the classes are balanced
    sleep_end = annotations_local[-2]["onset"] + 30 * 6 * 60
    annotations_local.crop(sleep_start, sleep_end)

    raw_file_local.set_annotations(annotations_local)




    return raw_file_local

def create_events(raw_file_local):
    # Normally, mne.events_from_annotations returns a tuple so the underscore in "events, _" ensures that only the events is saved to the variable
    events_local, _ = mne.events_from_annotations(raw_file_local, event_id=annotation_event_id, chunk_duration=30.0)

    # Create a dictionary linking the number of events to a label
    event_id_local = dict(awake=1, sleep=2)

    return events_local, event_id_local

def create_epochs(raw_file_local, events_local, event_id_local):
    tmax = 30.0 - 1.0 / raw_file_local.info["sfreq"]
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


    epochs_train_local = epochs_local.copy().crop(tmin=0.0, tmax=tmax) # Would need to find out if 0 - 30 is an appropriate crop, and no crop may be needed at all

    labels_local = epochs_local.events[0:len(epochs_local), -1]
    print("Event labels array:")
    print(labels_local)

    return epochs_local, epochs_train_local, labels_local

def eeg(epochs, frequency_bands=frequency_bands):

    psd = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0) # Multitaper
    psds, frequencies = psd.get_data(return_freqs=True)

    # Normalization WHAT IS THIS
    psds = psds / np.sum(psds, axis=-1, keepdims=True)

    # SLICE AND average EACH band to get
    # End data should look like ()epoch, 2 * bands number

    X = []
    for min_frequency, max_frequency in frequency_bands.values(): # Split into two for loops!!

        # og line
        # psds_band = psds[0:, 0:, (frequencies >= fmax) & (freqs < fmax).mean(axis=-1)]
        freq_mask = (frequencies >= min_frequency) & (frequencies < max_frequency)
        psds_band = psds[:, :, freq_mask].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), - 1))

    return np.concatenate(X, axis=1)



# Loading data from all of the participants

all_epochs = []
for subject_index in range(3):
    print("Loading participant", str(subject_index))

    raw_file = load_data(subject_index, 1)
    events, event_id = create_events(raw_file)

    epochs, _, _ = create_epochs(raw_file, events, event_id)
    all_epochs.append(epochs)

print(all_epochs)




# Feature engineering !!!

X = []
y = []
groups = []

for group_id, epochs in enumerate(all_epochs):
    print("Processing participants", str(group_id))

    X_epoch = eeg(epochs) # Change these names to be more informative
    y_epoch = epochs.events[:, 2]
    group_epoch = [group_id] * len(y_epoch)

    X.append(X_epoch)
    y.append(y_epoch)
    groups.append(group_epoch)

# Change to proper size for sklearn models
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
groups = np.concatenate(groups, axis=0)

# Create pipeline with CSP, FunctionTransformer - might cause overfitting, and LDA


# LeaveOneGroupOut / Train / test split
group_id = 0
for train, test in LeaveOneGroupOut().split(X, y, groups=groups):
    print("Processing participants", str(group_id))
    group_id += 1 # Increment

    # Training
    model = make_pipeline(LinearDiscriminantAnalysis())
    X_train = X[train]
    y_train = y[train]
    model.fit(X_train, y_train)

    # Testing
    X_test = X[test]
    y_test = y[test]

    y_pred = model.predict(X_test)

    print(y_pred)


    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy}")

    #Create a confusion matix and report on all the metrics
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)


    display.plot()
    print(classification_report(y_test, y_pred, target_names=event_id.keys()))
    print(event_id)


model.fit(X, y)
joblib.dump(model, "test_model.joblib")


# Trainig the model on NEW data

print("NEW data tested on model to see performance -----------------")

raw_file = load_data(4, 1)
events, event_id = create_events(raw_file)

epochs, _, _ = create_epochs(raw_file, events, event_id)

X = []

X_epoch = eeg(epochs) # Change these names to be more informative

X.append(X_epoch)

X_new = np.concatenate(X, axis=0)


# Warning!!! AI code just used for proof of concept
new_model = joblib.load("test_model.joblib")
preds = new_model.predict(X_new)


# Adjust to match YOUR label encoding
stage_names = {1: "Wake", 2: "Sleep"}

# Plot order: top = Wake, bottom = N3
plot_order = ["Wake", "Sleep"]
level = {name: i for i, name in enumerate(reversed(plot_order))}  # N3=0 ... W=4

y = np.array([level[stage_names[p]] for p in preds])

epoch_seconds = 30  # standard sleep epoch length
t_hours = np.arange(len(preds)) * epoch_seconds / 3600

fig, ax = plt.subplots(figsize=(12, 3))
ax.step(t_hours, y, where="post", linewidth=1.5)
ax.set_yticks(list(level.values()))
ax.set_yticklabels(list(level.keys()))
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Sleep stage")
ax.set_title("Predicted hypnogram")
ax.set_xlim(0, t_hours[-1])
plt.tight_layout()
plt.show()


