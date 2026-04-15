import streamlit as st
import numpy as np
import joblib
import neurokit2 as nk
import matplotlib.pyplot as plt
import wfdb
import tempfile
import os

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('ecg_model.pkl')
    return model

model = load_model()

# Page setup
st.title('ECG Arrhythmia Detector')
st.write('Upload an ECG signal file to detect abnormal heartbeats')
st.warning('⚠️ This tool is for educational purposes only and is not a substitute for professional medical diagnosis')

# File uploader
uploaded_file = st.file_uploader(
    'Upload ECG record (.dat file from MIT-BIH format)',
    type=['dat']
)

if uploaded_file is not None:
    st.info('Processing ECG signal...')
    
    # Save uploaded file temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        dat_path = os.path.join(tmpdir, '100.dat')
        with open(dat_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Check if header file was uploaded too
        st.write('Note: MIT-BIH records require both .dat and .hea files')
        
    # For demo purposes show analysis of a fixed record
    st.subheader('Demo Analysis')
    st.write('Showing analysis of MIT-BIH record 100 as demonstration')
    
    # Load demo record
    try:
        record = wfdb.rdrecord('data/mit-bih-arrhythmia-database-1.0.0/100')
        signal = record.p_signal[:, 0]
        fs = record.fs
        
        # Clean signal
        signal_clean = nk.ecg_clean(signal, sampling_rate=fs)
        
        # Detect R peaks
        peaks, info = nk.ecg_peaks(signal_clean, sampling_rate=fs)
        r_peaks = info['ECG_R_Peaks']
        
        # Extract beat segments
        window = 100
        beats = []
        valid_peaks = []
        
        for pos in r_peaks:
            if pos - window >= 0 and pos + window < len(signal_clean):
                beat = signal_clean[pos-window:pos+window]
                beats.append(beat)
                valid_peaks.append(pos)
        
        beats = np.array(beats)
        valid_peaks = np.array(valid_peaks)
        
        # Make predictions
        predictions = model.predict(beats)
        
        # Count results
        normal_count = np.sum(predictions == 1)
        abnormal_count = np.sum(predictions == 0)
        total = len(predictions)
        
        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Beats', total)
        with col2:
            st.metric('Normal Beats', normal_count)
        with col3:
            st.metric('Abnormal Beats', abnormal_count,
                     delta=f'{abnormal_count/total*100:.1f}% of total',
                     delta_color='inverse')
        
        # Plot ECG with annotations
        st.subheader('ECG Signal with Detected Beats')
        seconds = 30
        samples = seconds * fs
        time = np.arange(samples) / fs
        
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time, signal_clean[:samples], 
                color='steelblue', linewidth=0.8, label='ECG Signal')
        
        # Mark normal and abnormal peaks
        for peak, pred in zip(valid_peaks, predictions):
            if peak < samples:
                if pred == 1:  # normal
                    ax.scatter(peak/fs, signal_clean[peak],
                             color='green', s=30, zorder=5)
                else:  # abnormal
                    ax.scatter(peak/fs, signal_clean[peak],
                             color='red', s=50, zorder=5, marker='x')
        
        # Add legend
        ax.scatter([], [], color='green', s=30, label='Normal beat')
        ax.scatter([], [], color='red', s=50, marker='x', label='Abnormal beat')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title('ECG Signal - Green=Normal, Red=Abnormal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # HRV metrics
        st.subheader('HRV Analysis')
        rr_intervals = np.diff(valid_peaks) / fs * 1000
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Average Heart Rate', 
                     f'{60000/rr_intervals.mean():.1f} BPM')
        with col2:
            st.metric('Average RR Interval',
                     f'{rr_intervals.mean():.1f} ms')
        with col3:
            st.metric('HRV (SDNN)',
                     f'{rr_intervals.std():.1f} ms')
        
        # RR interval plot
        fig2, ax2 = plt.subplots(figsize=(14, 3))
        ax2.plot(rr_intervals, color='steelblue', linewidth=1)
        ax2.axhline(rr_intervals.mean(), color='red', 
                   linestyle='--', label=f'Mean: {rr_intervals.mean():.1f} ms')
        ax2.set_xlabel('Beat number')
        ax2.set_ylabel('RR Interval (ms)')
        ax2.set_title('RR Intervals Over Time (HRV)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f'Error processing record: {e}')