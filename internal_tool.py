import os
import streamlit as st
import pandas as pd
import re
import librosa
import spacy
import numpy as np
import soundfile as sf
import numpy as np
import tempfile
from nltk import ngrams
import string
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk import bigrams
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
import librosa.display
import joblib
import tensorflow as tf
import io
import librosa
import numpy as np
from pydub import AudioSegment
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import wave
import contextlib
import librosa
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

model = joblib.load("./background_noise_detector.pkl")
nlp = spacy.load("en_core_web_sm")
st.set_page_config(layout="wide")
df_combined = pd.read_csv("./merged_preprocessing_lvl_combined_segmentation_with_call_duration.csv")
df_split = pd.read_csv("merged_preprocessing_lvl_split_segmentation.csv")
combined_unique_urls = df_combined['url'].unique()
split_unique_urls = df_split['url'].unique()
fail_check = st.checkbox("Fail")
pass_check = st.checkbox("Pass")
def get_url(url):
    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
    cleaned_url = re.sub(pattern, '',url)
    return cleaned_url
def count_empathy_phrases(sentence, empathy_words):
    if isinstance(sentence, str):
        doc = nlp(sentence.lower())
        tokens = [token.text for token in doc]
        count = 0
        word_string=[]
        # Iterate through n-grams of different lengths
        for n in range(2, 3):  # Considering n-grams from unigram to 4-gram
            n_grams = ngrams(tokens, n)
            for gram in n_grams:
                gram_text = ' '.join(gram)
                if gram_text in empathy_words:
                    count += 1
                    word_string.append(gram_text) 
    else:
        count = 0
        word_string=[]
    return count , word_string
def create_mel_spectrogram(file_path, n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(file_path, sr=None)
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    spect = librosa.power_to_db(spect, ref=np.max)
    max_length = 500
    if spect.shape[1] > max_length:
        spect = spect[:, :max_length]
    else:
        spect = np.pad(spect, ((0,0),(0,max_length-spect.shape[1])), mode='constant')
    print(spect.shape)
    return spect
def extract_features(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512, n_mfcc=13):
    y,sr=librosa.load(audio_path)
    max_length = 600
    # Extract mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    if spectrogram.shape[1] > max_length:
        spectrogram = spectrogram[:, :max_length]
    else:
        spectrogram = np.pad(spectrogram, ((0,0),(0,max_length-spectrogram.shape[1])), mode='constant')

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mels=n_mels)
    if mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    else:
        mfccs = np.pad(mfccs, ((0,0),(0,max_length-mfccs.shape[1])), mode='constant')
        # print(mfccs)

    # Extract other features
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Combine features into a single array
    rest_features_arr = np.array([zero_crossing_rate, rms, spectral_centroid, spectral_rolloff, chroma_stft, spectral_bandwidth], dtype=np.float32)
    dummy_zeros = np.zeros(594)
    new_array = np.concatenate((rest_features_arr, dummy_zeros))

    # Concatenate all arrays
    result = np.concatenate((mfccs, spectrogram, new_array.reshape(1, -1)), axis=0)
    # print(result)
    print(result.shape)
    return result
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
def lowercase_text(text):
    return str(text.lower())
def generate_word_pairs(text):
    words = word_tokenize(text)
    word_pairs = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
    return word_pairs
def calculate_wpm(row):
    if row['duration'] > 0:
        return row['transcript_count'] / (row['duration']/60)
    else:
        return 0
    

def get_bag_of_words(document, vocab):
    bag_of_words = {word: 0 for word in vocab}
    words = document.lower().split()
    for word in words:
        if word in vocab:
            bag_of_words[word] += 1
    return bag_of_words

def extract_spectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    max_length = 700

    # Ensure the spectrogram width is exactly max_length
    if spectrogram.shape[1] > max_length:
        spectrogram = spectrogram[:, :max_length]
    else:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, max_length - spectrogram.shape[1])), mode='constant')

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')

    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    rest_features_arr = [zero_crossing_rate, rms, spectral_centroid, spectral_rolloff, chroma_stft, spectral_bandwidth]
    dummy_zeros = [0] * 694
    new_array = np.array(rest_features_arr + dummy_zeros, dtype=np.float32)

    # Concatenate new_array, mfccs, and spectrogram
    features = np.concatenate((new_array[np.newaxis, :], mfccs, spectrogram), axis=0)

    # Ensure the features height is exactly 149
    expected_height = 149
    if features.shape[0] > expected_height:
        features = features[:expected_height, :]
    else:
        features = np.pad(features, ((0, expected_height - features.shape[0]), (0, 0)), mode='constant')

    return features
def human_disturbance():
    st.subheader('Human Disturbance')
    with st.expander("Audio Quality", expanded=False):
        if st.button('Run Algorithm',key="human disturbance"):
            with st.spinner():
                item1=[]
                item2=[]
                item3=[]
                item4=[]
                item5=[]
                for url in split_unique_urls:
                    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                    cleaned_url = re.sub(pattern, '',url)
                    df=df_split[df_split['url']==url]
                    df_left=df[df['channel']=='Mono_Left']
                    df_right=df[df['channel']=='Mono_Right']
                    call_duration=df['audio_duration'].iloc[0]
                    overlapping_info_1 = pd.DataFrame(columns=['url','start', 'end', 'paramater_label','date','call_duration', 'overlapping_df'])
                    audio =(f'audio_files/{cleaned_url}_left.mp3')
   
                    y,sr=librosa.load(audio)
                    y = librosa.util.normalize(y)

                    
                  
                    for index_left, row_left in df_left.iterrows():
                        start=row_left['start']
                        end=row_left['end']
                        start_sample = int(start *sr)
                        end_sample = int(end *sr)
                        clip=y[start_sample:end_sample]
                        segment_duration =end-start
                        if segment_duration > 3:
                            # Load the trained model
                            
                            # Extract spectrogram from the new audio file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                                temp_path = temp_file.name
                                sf.write(temp_path, clip, sr)
                                

                            clip_y, clip_sr = librosa.load(temp_path)

                            zero_crossings = librosa.zero_crossings(clip_y, pad=False)
                            spectral_centroid = librosa.feature.spectral_centroid(y=clip_y, sr=clip_sr)
                            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=clip_y, sr=clip_sr)
                            spectral_rolloff = librosa.feature.spectral_rolloff(y=clip_y, sr=clip_sr)
                            rms = librosa.feature.rms(y=clip_y)

                            features= {
                                'zero_crossings': np.sum(zero_crossings),
                                'spectral_centroid': np.mean(spectral_centroid),
                                'spectral_bandwidth': np.mean(spectral_bandwidth),
                                'spectral_rolloff': np.mean(spectral_rolloff),
                                'rms': np.median(rms)
                            }
                            # dullness_thresholds = {
                            #     "zero_crossings":21876,
                            #     "spectral_rolloff":1573,


                            #     'spectral_centroid': 891,  # Hz
                            #     'spectral_bandwidth': 762,  # Hz
                            #     'rms': 0.05
                            # }

                            #10 % less
                            dullness_thresholds ={"zero_crossings": 21876.4,
                                                "spectral_rolloff": 1573.7,
                                                "spectral_centroid": 891.9,  # Hz
                                                "spectral_bandwidth": 762.8,  # Hz
                                                "rms": 0.08}

                            if (
                                # features['zero_crossings'] < dullness_thresholds['zero_crossings'] and
                                # features['spectral_rolloff'] < dullness_thresholds['spectral_rolloff'] and
                                # features['spectral_centroid'] < dullness_thresholds['spectral_centroid'] and
                                # features['spectral_bandwidth'] < dullness_thresholds['spectral_bandwidth'] and
                                features['rms'] < dullness_thresholds['rms']):
                            
                                # Basic statistics
                                rms_energy_np = np.array(rms)
                                mean_energy = np.mean(rms_energy_np)
                                std_energy = np.std(rms_energy_np)
                                max_energy = np.max(rms_energy_np)
                                min_energy = np.min(rms_energy_np)
                                st.write(rms)

                                st.write(f"Mean RMS Energy: {mean_energy}")
                                st.write(f"Standard Deviation of RMS Energy: {std_energy}")
                                st.write(f"Max RMS Energy: {max_energy}")
                                st.write(f"Min RMS Energy: {min_energy}")

                                # Identify segments of silence
                                silence_threshold = 0.01
                                silence_segments = np.where(rms_energy_np < silence_threshold)[0]
                                st.write(f"Silence Segments: {silence_segments}")

                                # Identify segments of speech
                                speech_threshold = 0.05
                                speech_segments = np.where(rms_energy_np > speech_threshold)[0]
                                st.write(f"Speech Segments: {speech_segments}")

                                # Identify abrupt changes
                                abrupt_changes = np.where(np.abs(np.diff(rms_energy_np)) > (2 * std_energy))[0]
                                st.write(f"Abrupt Changes: {abrupt_changes}")
                                fig, ax = plt.subplots(figsize=(15, 5))
                                ax.plot(rms)
                                ax.axhline(y=silence_threshold, color='r', linestyle='--', label='Silence Threshold')
                                ax.axhline(y=speech_threshold, color='g', linestyle='--', label='Speech Threshold')
                                ax.set_title('RMS Energy of Audio Signal')
                                ax.set_xlabel('Sample Index')
                                ax.set_ylabel('RMS Energy')
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                st.audio(clip_y,sample_rate=clip_sr)
                                                                                            
                            # else:
                            #     st.write('Is Clear')
                            #     st.audio(clip_y,sample_rate=clip_sr)
                            
                            item1.append(features['zero_crossings'])
                            item2.append(features['spectral_centroid'])
                            item3.append(features['spectral_bandwidth'])
                            item4.append(features['spectral_rolloff'])
                            item5.append(features['rms'])
                st.write(item1)
                st.write(sum(item1)/len(item1))
                st.write(item2)
                st.write(sum(item2)/len(item2))
                st.write(item3)
                st.write(sum(item3)/len(item3))
                st.write(item4)
                st.write(sum(item4)/len(item4))
                st.write(item5)
                st.write(sum(item5)/len(item5))


    # Accumulate sums
        

    # Calculate averages
                # average_features = {key: value / avg_count for key, value in accumulated_features.items()}
                # st.write(average_features)
                # print(average_features)
                    

                            
def dull_1():
    st.subheader('Dull ')
    with st.expander("Audio Quality", expanded=False):
        if st.button('Run Algorithm',key="dull"):
            with st.spinner():
                for url in split_unique_urls:
                    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                    cleaned_url = re.sub(pattern, '',url)
                    df=df_split[df_split['url']==url]
                    df_left=df[df['channel']=='Mono_Left']
                    df_right=df[df['channel']=='Mono_Right']
                    call_duration=df['audio_duration'].iloc[0]
                    overlapping_info_1 = pd.DataFrame(columns=['url','start', 'end', 'paramater_label','date','call_duration', 'overlapping_df'])
                    audio =(f'audio_files/{cleaned_url}_left.mp3')
                    y, sr = librosa.load(audio)
                    current_duration = librosa.get_duration(y=y, sr=sr)

                    # Calculate the speed factor to adjust to 10 seconds
                    speed_factor = 10.0 / current_duration

                    # Adjust the audio duration
                    y_adjusted = librosa.effects.time_stretch(y, rate=speed_factor)

                    # If the adjusted audio is shorter than 10 seconds, pad it
                    if len(y_adjusted) < sr * 10:
                        padding = sr * 10 - len(y_adjusted)
                        y_adjusted = np.pad(y_adjusted, (0, padding), 'constant')

                    # Create a temporary directory
                    temp_dir = tempfile.mkdtemp()

                    # Save the modified audio to a temporary file
                    output_path = os.path.join(temp_dir, "output_audio_10s.wav")
                    sf.write(output_path, y_adjusted, sr)

                    # Print the path where the audio is saved
                    print(f"Audio has been adjusted to 10 seconds and saved to {output_path}")

                    # Display the audio (assuming st.audio() is Streamlit's function)
                    st.audio(y, sample_rate=sr)
           
                    # return output_path

                    # segment_duration = end - start
                    # clip = y[start_sample:end_sample]

                    # with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    #     temp_path = temp_file.name
                    #     sf.write(temp_path, clip, sr)

                    # clip_y, clip_sr = librosa.load(temp_path)


                    #############
                    # target_duration=20
                    # sr=22050
                    # n_mels=128
                    # length = target_duration * sr
                    # # if segment_duration >5:
                    # if len(y) < length:
                    #     y = np.pad(y, (0, length - len(y)))
                    # else:
                    #     y = y[:length]
                    # mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                    # mel_spectrogram = mel_spectrogram.reshape((*mel_spectrogram.shape, 1))

                    # mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
                    # model = load_model('./speech_volume_detection_model_50_epoch.h5')

                    # prediction = model.predict(mel_spectrogram)

                    # if  prediction[0][0] > 0.5 :
                    #     if fail_check:
                    #         st.write("Clear Clips")
                    #         st.audio(y, sample_rate=sr)
                    # else:
                    #     if pass_check:
                                
                    #         st.write("Reduced")
                    #         st.audio(y, sample_rate=sr)
                            

def dull():
    st.subheader('Dull ')
    with st.expander("Audio Quality", expanded=False):
        if st.button('Run Algorithm',key="dull"):
            with st.spinner():
                for url in split_unique_urls:
                    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                    cleaned_url = re.sub(pattern, '',url)
                    df=df_split[df_split['url']==url]
                    df_left=df[df['channel']=='Mono_Left']
                    df_right=df[df['channel']=='Mono_Right']
                    call_duration=df['audio_duration'].iloc[0]
                    overlapping_info_1 = pd.DataFrame(columns=['url','start', 'end', 'paramater_label','date','call_duration', 'overlapping_df'])
                    audio =(f'audio_files/{cleaned_url}_left.mp3')
                    y, sr = librosa.load(audio)
                    for index_left, row_left in df_left.iterrows():
                        start = row_left['start']
                        end = row_left['end']
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        # clip=y[start_sample:end_sample]
                        # end_sample = sr * 10  # 10 seconds worth of samples

                        # Clip the audio to the first 10 seconds
                        segment_duration = end - start
                        # if segment_duration > 5:
                        clip = y[start_sample:end_sample]

                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                            temp_path = temp_file.name
                            sf.write(temp_path, clip, sr)
                            # st.write(sr)

                        clip_y, clip_sr = librosa.load(temp_path)
                        # y, sr = librosa.load(audio_path, sr=None)
                        target_duration=10
                        sr=22050
                        n_mels=128
                        # Load the audio file
                        length = target_duration * clip_sr
                        if segment_duration >5:
                            # st.write(clip_sr)
                            # st.write(length)

                            # Pad or trim the audio to the target duration
                            if len(clip_y) < length:
                                clip_y = np.pad(clip_y, (0, length - len(clip_y)))
                            else:
                                clip_y = clip_y[:length]

                            # Extract the Mel spectrogram
                            mel_spectrogram = librosa.feature.melspectrogram(y=clip_y, sr=sr, n_mels=n_mels)
                            
                            # Reshape to add a channel dimension
                            mel_spectrogram = mel_spectrogram.reshape((*mel_spectrogram.shape, 1))

                            # Expand dimensions to match the input shape of the model (batch size, height, width, channels)
                            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
                            model = load_model('./speech_volume_detection_model_50_epoch.h5')

                            # Make prediction
                            prediction = model.predict(mel_spectrogram)

                            # Label the prediction
                            # label = "reduced" if prediction > 0.5 else "clean"
                            # print(label,prediction)
                            # n_mels=128
                            # max_len=128
                            # mel_spec = librosa.feature.melspectrogram(y=clip_y, sr=clip_sr, n_mels=n_mels)
                            # log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                            
                            # # Ensure all mel spectrograms have the same length by padding or truncating
                            # if log_mel_spec.shape[1] > max_len:
                            #     log_mel_spec = log_mel_spec[:, :max_len]
                            # else:
                            #     pad_width = max_len - log_mel_spec.shape[1]
                            #     log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
                            
                            # rms = librosa.feature.rms(y=clip_y).mean()
                            # decibels = librosa.amplitude_to_db(np.abs(clip_y)).mean()
                            # mel_spec = np.expand_dims(log_mel_spec, axis=-1)  # Add channel dimension
                            

            
                            # Predict using the model
                            # prediction = model.predict([np.array([mel_spec]), np.array([[rms, decibels]])])
                            if  prediction[0][0] > 0.5 :
                                if fail_check:
                                    st.write("Clear Clips")
                                    st.audio(clip_y, sample_rate=sr)
                            else:
                                if pass_check:
                                        
                                    st.write("Reduced")
                                    st.audio(clip_y, sample_rate=sr)
                            
                            # n_mels = 128
                            # fmax = 8000
                            # n_mfcc = 13
                            # S = librosa.feature.melspectrogram(y=clip_y, sr=clip_sr, n_mels=n_mels, fmax=fmax)
                            # S_DB = librosa.power_to_db(S, ref=np.max)

                            # # RMS Energy
                            # rms = librosa.feature.rms(y=clip_y)[0]

                            # # Decibels
                            # db = librosa.amplitude_to_db(S, ref=np.max)
                            # size = (128, 128)

                            # # MFCCs
                            # mfccs = librosa.feature.mfcc(y=clip_y, sr=clip_sr, n_mfcc=n_mfcc)
                            # resized_S_DB = cv2.resize(S_DB, size)
                            # resized_rms = cv2.resize(rms.reshape(1, -1), size)
                            # resized_db = cv2.resize(db, size)
                            # resized_mfccs = [cv2.resize(mfcc, size) for mfcc in mfccs]

                            # # Stack the features along the channel dimension
                            # combined_features = np.stack([resized_S_DB, resized_rms, resized_db, *resized_mfccs], axis=-1)
                            # combined_features = combined_features.reshape(1, *combined_features.shape)  # Add batch dimension
                            # best_model = load_model('./dull_classification_model.h5')

                            # prediction = best_model.predict(combined_features)

                            # # Example usage
                            # if prediction > 0.5:
                            #     st.write('The audio is dull and not audible.')
                            #     st.audio(clip_y, sample_rate=clip_sr)
                            # else:
                            #     print('The audio is clear and audible.')
                        #         st.write("Noisy")
                        #         st.audio(clip, sample_rate=sr) 
                        #     else:
                        #         st.write("Clean")
                        #         st.audio(clip, sample_rate=sr) 

                    # features = extract_features(audio)


                    # audio_segment = AudioSegment.from_mp3(io.BytesIO(audio))
                    # buffer = io.BytesIO()
                    # audio_segment.export(buffer, format='wav')
                    # buffer.seek(0)

                    # # Extract features using librosa
                    # waveform, sr = librosa.load(buffer, sr=None)
                    # features = extract_features(waveform, sr)
                    # features = features[np.newaxis, ..., np.newaxis]
                    # y,sr=librosa.load(audio)


    
                    # # # Add a channel dimension
                    
                    # # # Predict
                    # prediction = model_background.predict(features)
                    
                    # if prediction[0][0] > 0.5 :
                    #     st.write("Noisy")
                    #     st.audio(y, sample_rate=sr) 


                    # for index_left, row_left in df_left.iterrows():
                    #     start=row_left['start']
                    #     end=row_left['end']
                    #     start_sample = int(start *sr)
                    #     end_sample = int(end *sr)
                    #     clip=y[start_sample:end_sample]
                    #     segment_duration =end-start
                    #     if segment_duration > 10:
                    #         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    #             temp_path = temp_file.name
                    #             sf.write(temp_path, clip, sr)

                    #         clip_y, clip_sr = librosa.load(temp_path)
                    #         spectrogram = extract_spectrogram(temp_path)
    
                    #         # Add a channel dimension
                    #         spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
                            
                    #         # Predict
                    #         prediction = model_background.predict(spectrogram)
                            
                    #         if prediction[0][0] > 0.2 :
                    #             st.write (prediction)
                    #             st.write("Noisy")
                    #             st.audio(clip_y, sample_rate=clip_sr) 
                    
                    # else :
                    #     st.write('Clean')

                    # y, sr = librosa.load(audio,sr=22050)
                    # n_mels=128
                    # n_fft=2048
                    # hop_length=512
                    # n_mfcc=13
                    

                    # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                    # spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
                    # spectrogram = np.array(spectrogram)
                    # max_length=700
                    # if spectrogram.shape[1] > max_length:
                    #     spectrogram = spectrogram[:, :max_length]
                    # else:
                    #     spectrogram = np.pad(spectrogram, ((0,0),(0,max_length-spectrogram.shape[1])), mode='constant')
                    # mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mels=n_mels)
                    # if mfccs.shape[1] > max_length:
                    #     mfccs = mfccs[:, :max_length]
                    # else:
                    #     mfccs = np.pad(mfccs, ((0,0),(0,max_length-mfccs.shape[1])), mode='constant')
                    # zero_crossing_rate=(np.mean(librosa.feature.zero_crossing_rate(y)))
                    # rms=(np.mean(librosa.feature.rms(y=y)))
                    # spectral_centroid=(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                    # spectral_rolloff=(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
                    # chroma_stft=(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
                    # spectral_bandwidth=(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
                    # rest_features_arr = [zero_crossing_rate, rms, spectral_centroid, spectral_rolloff, chroma_stft, spectral_bandwidth]#, spectrogram, mfccs]
                    # rest_features_arr = rest_features_arr
                    # dummy_zeros = [0] * 694
                    # new_array = np.array([rest_features_arr + dummy_zeros], dtype=np.float32)
                    # result = np.concatenate((mfccs, spectrogram), axis=0)
                    # result=np.concatenate((new_array,result),axis=0)

                    # result=np.array(result, dtype=np.float32)
                    # st.write(result)
                    # result = result[..., np.newaxis]

                    # st.write(result.shape)
                    # prediction = model_background.predict(result)
                    # st.write(prediction)

                    ##################### working
                    # y, sr=librosa.load(audio)
                    # # sr=22050
                    # n_mels=128
                    # n_fft=2048
                    # hop_length=512
                    # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                    # spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
                    # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                    # spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
                    # # Extract MFCCs
                    # mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mels=n_mels)
                    
                    # # Combine features (you can choose to concatenate or use them separately)
                    # spectrogram = np.concatenate((spectrogram, mfccs), axis=0)
                    # max_length = 128
                    # if spectrogram.shape[1] > max_length:
                    #     spectrogram = spectrogram[:, :max_length]
                    # else:
                    #     spectrogram = np.pad(spectrogram, ((0,0),(0,max_length-spectrogram.shape[1])), mode='constant')
                    # spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
                    # model_path='./background_noise_detector_mfcc_spectogram_cnn.h5'
                    # model_background = tf.keras.models.load_model(model_path)
                    # prediction = model_background.predict(spectrogram)
                    # if round(prediction[0][0],2) > 0.5:
                    #     st.write(round(np.float64(prediction[0][0]),2))
                    #     st.write("Noisy")
                    #     st.audio(y, sample_rate=sr) 
########################################working
            
                    # for index_left, row_left in df_left.iterrows():
                    #     start=row_left['start']
                    #     end=row_left['end']
                    #     start_sample = int(start *sr)
                    #     end_sample = int(end *sr)
                    #     segment_duration =end-start
                    #     if segment_duration > 10:
                    #         # with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    #         #     temp_path = temp_file.name
                    #         #     sf.write(temp_path, clip, sr)

                    #         # Load the clip again with librosa
                    #         # clip_y, clip_sr = librosa.load(temp_path)
                    #         clip, sr = librosa.load(audio, offset=start, duration=end - start)


                
                    #         n_mels=128
                    #         n_fft=2048
                    #         hop_length=512
                    #         spectrogram = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                    #         spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
                    #         # Extract MFCCs
                    #         mfccs = librosa.feature.mfcc(y=clip, sr=sr,n_mels=n_mels)
                            
                    #         # Combine features (you can choose to concatenate or use them separately)
                    #         spectrogram = np.concatenate((spectrogram, mfccs), axis=0)
                    #         # Extract spectrogram from the new audio file
                    #         max_length = 128
                    #         if spectrogram.shape[1] > max_length:
                    #             spectrogram = spectrogram[:, :max_length]
                    #         else:
                    #             spectrogram = np.pad(spectrogram, ((0,0),(0,max_length-spectrogram.shape[1])), mode='constant')
                    #         spectrogram = spectrogram[np.newaxis, ..., np.newaxis]


                    #         prediction = model_background.predict(spectrogram)
                            
                    #         # # prediction = model.predict(spectrogram.reshape(-1, spectrogram.shape[1]))

                    #         # # return 'Noisy' if prediction[0][0] > 0.5 else 'Clean'
                    #         if round(prediction[0][0],2) > 0.5:
                    #             st.write(round(np.float64(prediction[0][0]),2))

                    #             st.write("Noisy")
                    #             st.audio(clip, sample_rate=sr) 
                    ##############################################
                    # else:
                    #     st.write("Clean")
                    #     st.audio(y, sample_rate=sr) 
                    # for index_left, row_left in df_left.iterrows():
                    #     start=row_left['start']
                    #     end=row_left['end']
                    #     start_sample = int(start *sr)
                    #     end_sample = int(end *sr)
                    #     clip = y[start_sample:end_sample]
                    #     segment_duration =end-start
                        

                        # if segment_duration >5:
                        #     features = []
                        #     features.append(np.mean(librosa.feature.zero_crossing_rate(clip)))
                        #     features.append(np.mean(librosa.feature.rms(y=clip)))
                        #     features.append(np.mean(librosa.feature.spectral_centroid(y=clip, sr=sr)))
                        #     features.append(np.mean(librosa.feature.spectral_rolloff(y=clip, sr=sr)))
                        #     mfcc = librosa.feature.mfcc(y=clip, sr=sr)
                        #     for coeff in mfcc:
                        #         features.append(np.mean(coeff))
                        #     features.append(np.mean(librosa.feature.chroma_stft(y=clip, sr=sr)))
                        #     features.append(np.mean(librosa.feature.spectral_bandwidth(y=clip, sr=sr)))
                        #     X_new = np.array([features])
                        #     prediction = model.predict(X_new)
                            
                        #     if prediction[0] == 1:
                        #         st.write("Noisy")
                        #         st.audio(clip, sample_rate=sr) 
                        #     else:
                        #         st.write("Clean")
                        #         st.audio(clip, sample_rate=sr) 

        

     
        
                        # spectrogram = librosa.amplitude_to_db(abs(librosa.stft(clip)), ref=np.max)
                        # librosa.display.specshow(spectrogram, y_axis='log', x_axis='time')
                        # if np.mean(spectrogram) < -78 and segment_duration<1:  # Adjust this threshold as needed
                        #     with st.container():
                        #         st.audio(clip, sample_rate=sr)
                        #     # Create a Streamlit figure
                        #     fig = plt.figure(figsize=(5, 2))
                        #     ax = fig.add_subplot(111)
                        #     img = ax.imshow(spectrogram, cmap='inferno', origin='lower')
                        #     fig.colorbar(img, format='%+2.0f dB')
                        #     plt.title('Spectrogram')
                        #     plt.tight_layout()

                        #     # Display the spectrogram using Streamlit
                        #     st.pyplot(fig)
                        #     st.write(np.mean(spectrogram))
                        #     st.write("The audio is dull.")
  

                        
                        

                    #     for index_right, row_right in df_right.iterrows():
                    #         overlap_start = max(row_left['start'], row_right['start'])
                    #         overlap_end = min(row_left['end'], row_right['end'])
                    #         overlap_duration = overlap_end - overlap_start
                    #         if overlap_start < overlap_end and overlap_duration >= 2:
                    #             if row_left['start'] <= row_right['start']:
                    #                 overlapping_df = 'Mono_Right'
                    #             else:
                    #                 overlapping_df = 'Mono_Left'
                    #             data_to_append = {
                    #                 'start': overlap_start,
                    #                 'end': overlap_end,
                    #                 'call_duration':call_duration,
                    #                 'overlapping_df': overlapping_df
                    #             }
                    #             df_to_append = pd.DataFrame([data_to_append])
                    #             #non_empty_columns = df_to_append.columns[df_to_append.notna().any()]
                    #             overlapping_info_1 = pd.concat([overlapping_info_1, df_to_append], ignore_index=True)
                    # agent_overlaps=overlapping_info_1

                    # if not agent_overlaps.empty:
                    #     # st.markdown("### Entire Call")
                    #     band_length = agent_overlaps["call_duration"].iloc[0]
                    #     fig, ax = plt.subplots(figsize=(10, 0.5))
                    #     ymargin = 0.05 * band_length
                    #     ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                    #     ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                    #     ax.set_ylim(-0.9, 0.5)
                    #     ax.axis('off')

                        
                    #     for index,rows in agent_overlaps.iterrows():
                    #         st.markdown(f"###### From [{rows['start']}] To [{rows['end']}] seconds Overlap of [{round(rows['end']-rows['start'],2)}] seconds")
                    #         ax.barh(0, rows['end']-rows['start'], left=rows['start'], color='red', edgecolor='none')
                    #     st.pyplot(fig)
                    #     audio =(f'./audio_files/{cleaned_url}.mp3')
                    #     y, sr=librosa.load(audio)
                    #     with st.container():
                    #         st.audio(y, sample_rate=sr)
                    #     spectrogram = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
                    #     librosa.display.specshow(spectrogram, y_axis='log', x_axis='time')

                    #     # Create a Streamlit figure
                    #     fig = plt.figure(figsize=(10, 4))
                    #     ax = fig.add_subplot(111)
                    #     img = ax.imshow(spectrogram, cmap='inferno', origin='lower')
                    #     fig.colorbar(img, format='%+2.0f dB')
                    #     plt.title('Spectrogram')
                    #     plt.tight_layout()

                    #     # Display the spectrogram using Streamlit
                    #     st.pyplot(fig)
                    #     st.write(np.mean(spectrogram))
                    #     if np.mean(spectrogram) > 20:  # Adjust this threshold as needed
                    #         st.write("The audio is dull.")
                    #     else:
                    #         st.write("The audio is not dull.")


def background():
    st.subheader('Avoiding interrupting the customer during conversation')
    with st.expander("Audio Quality", expanded=False):
        if st.button('Run Algorithm',key="background"):
            with st.spinner():
                for url in split_unique_urls[:10]:
                    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                    cleaned_url = re.sub(pattern, '',url)
                    df=df_split[df_split['url']==url]
                    df_left=df[df['channel']=='Mono_Left']
                    df_right=df[df['channel']=='Mono_Right']
                    call_duration=df['audio_duration'].iloc[0]
                    overlapping_info_1 = pd.DataFrame(columns=['url','start', 'end', 'paramater_label','date','call_duration', 'overlapping_df'])
                    for index_left, row_left in df_left.iterrows():
                        for index_right, row_right in df_right.iterrows():
                            overlap_start = max(row_left['start'], row_right['start'])
                            overlap_end = min(row_left['end'], row_right['end'])
                            overlap_duration = overlap_end - overlap_start
                            if overlap_start < overlap_end and overlap_duration >= 2:
                                if row_left['start'] <= row_right['start']:
                                    overlapping_df = 'Mono_Right'
                                else:
                                    overlapping_df = 'Mono_Left'
                                data_to_append = {
                                    'start': overlap_start,
                                    'end': overlap_end,
                                    'call_duration':call_duration,
                                    'overlapping_df': overlapping_df
                                }
                                df_to_append = pd.DataFrame([data_to_append])
                                #non_empty_columns = df_to_append.columns[df_to_append.notna().any()]
                                overlapping_info_1 = pd.concat([overlapping_info_1, df_to_append], ignore_index=True)
                    agent_overlaps=overlapping_info_1

                    if not agent_overlaps.empty:
                        # st.markdown("### Entire Call")
                        band_length = agent_overlaps["call_duration"].iloc[0]
                        fig, ax = plt.subplots(figsize=(10, 0.5))
                        ymargin = 0.05 * band_length
                        ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                        ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                        ax.set_ylim(-0.9, 0.5)
                        ax.axis('off')

                        
                        for index,rows in agent_overlaps.iterrows():
                            st.markdown(f"###### From [{rows['start']}] To [{rows['end']}] seconds Overlap of [{round(rows['end']-rows['start'],2)}] seconds")
                            ax.barh(0, rows['end']-rows['start'], left=rows['start'], color='red', edgecolor='none')
                        st.pyplot(fig)
                        audio =(f'./audio_files/{cleaned_url}.mp3')
                        y, sr=librosa.load(audio)
                        with st.container():
                            st.audio(y, sample_rate=sr)
                        spectrogram = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
                        librosa.display.specshow(spectrogram, y_axis='log', x_axis='time')

                        # Create a Streamlit figure
                        fig = plt.figure(figsize=(10, 4))
                        ax = fig.add_subplot(111)
                        img = ax.imshow(spectrogram, cmap='inferno', origin='lower')
                        fig.colorbar(img, format='%+2.0f dB')
                        plt.title('Spectrogram')
                        plt.tight_layout()

                        # Display the spectrogram using Streamlit
                        st.pyplot(fig)
                        st.write(np.mean(spectrogram))
                        if np.mean(spectrogram) > 20:  # Adjust this threshold as needed
                            st.write("The audio is dull.")
                        else:
                            st.write("The audio is not dull.")


def open_with_standard_script():
    st.subheader('\nOpening should be using a standard script\n')
    with st.expander("Opening", expanded=False):
        if st.button('Run Algorithm',key="opening_script"):
            with st.spinner():
                vocab=['my name','calling from','dot com','मैं देख','can see','मेरा नाम','बात कर','confirm नहीं','confirm नहीं','पहले shopping']#,'my name','place कर','order place','पहले shopping',
                #vocab=['रही हूं', 'कर रही', 'dot com', 'सकती हूं', 'से बात', 'बात कर', 'क्या मैं', 'order place', 'मेरा नाम', 'है और', 'और मैं', 'देख सकती','हूं कि', 'मैं देख', 'com से', 'hello क्या', 'कर रहे', 'मैं आपको', 'place कर', 'नहीं किया', 'call कर', 'रहे थे', 'confirm नहीं', 'order को', 'कर सकती', 'को confirm', 'मैं dot', 'मदद कर', 'के लिए', 'place करने', 'हूं मैं', 'कि आप', 'में मदद', 'हूं maam', 'करने में', 'क्या मेरी', 'मेरी बात', 'हो रही', 'से हो', 'आपका order', 'है मैं', 'किया है', 'हमारे साथ', 'रही है', 'बारे में', 'के बारे', 'आपको order','नाम आयशा', 'है क्या', 'आपने order', 'से call', 'कि आपका', 'मैं आपका', 'आयशा है', 'एक order', 'com की', 'ओर से', 'दुकान dot', 'सकते हैं', 'आपने ना', 'की ओर', 'साथ पहले', 'order आप', 'आप place', 'दीवाली offer', 'maam मेरा', 'पहले shopping', 'shopping की', 'का order', 'of   ffer के', 'पर आपने', 'लिए call', 'का एक', 'की है', 'बताने के', 'में बताने', 'आपका एक', 'ठीक है', 'हूं मेरा', 'नाम है', 'थे पर',  'आपको हमारे', 'maam मैं', 'रहे हैं', 'placing order', 'किया क्या', 'इस order', 'करने का', 'आपने हमारे', 'हूं आपने', 'थे आपने', 'मैं दुकान', 'कर सकते', 'हमारे दीवाली', 'ना हमारे', 'यह सही', 'good evening', 'तो मैं']
                df = pd.read_csv('./merged_preprocessing_lvl_split_segmentation_with_call_duration.csv')
                left_df=df[df['channel']=="Mono_Left"]
                left_df = left_df.dropna(subset="transcript")  
                left_df['transcript'] = left_df['transcript'].apply(lambda x: remove_punctuation(x) if isinstance(x, str) else x)
                left_df['transcript'] = left_df['transcript'].apply(lowercase_text)
                count=0
                unique_url=df['url'].unique()
                data_1 = []
                count=0
                for url in unique_url[:50]:
                    filtered_df=left_df[left_df['url'] == url]
                    filtered_df=(filtered_df.head(5))
                    result = filtered_df['transcript'].str.cat(sep=' ')
                    if len(result.split()) > 10:
                        tokens = generate_word_pairs(result)
                        presence_array = [1 if token in tokens else 0 for token in vocab]
                        nonzero_count = np.count_nonzero(presence_array)
                        if nonzero_count >=2:
                            #st.write(result)
                            words = word_tokenize(result)
                            # Generate bigrams from the tokenized words
                            bigrams = list(ngrams(words, 2))
                            # Convert bigrams to string format
                            bigram_strings = [' '.join(bigram) for bigram in bigrams]
                            transcript_count=len(result.split())
                            match_count=0
                            # Highlight bigrams in the original text
                            for phrase in vocab:
                                if phrase in bigram_strings:
                                    match_count+=1
                                    # text = re.sub(f'({re.escape(phrase)})', r'<span style="background-color: yellow">\1</span>', result)
                            # Display the highlighted text using Streamlit's markdown function

                            st.markdown("#### Transcript")
                            st.markdown(result, unsafe_allow_html=True)
                            # st.write(nonzero_count)
                            row_1 = {
                                "index": "index",
                                'transcript': result}
                            data_1.append(row_1)
                            count=count+1
                            start=filtered_df['start'].iloc[0]
                            end=filtered_df['end'].iloc[-1]
                            pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                            cleaned_url = re.sub(pattern, '',url)
                            audio =(f'./audio_files/{cleaned_url}_left.mp3')
                            y, sr=librosa.load(audio)
                            start_sample = int(start *sr)
                            end_sample = int(end *sr)
                            clip = y[start_sample:end_sample]
                            st.markdown(f"##### Match {round(match_count/(len(vocab))*100,1)} %")

                            st.markdown("#### Audio")

                            with st.container():
                                st.audio(clip, sample_rate=sr)  
                            st.divider() 

                        else:
                            st.markdown("#### Transcript")
                            st.markdown(f"<p style='color:red;'>{result}</p>", unsafe_allow_html=True)
                            words = word_tokenize(result)
                            # Generate bigrams from the tokenized words
                            bigrams = list(ngrams(words, 2))
                            # Convert bigrams to string format
                            bigram_strings = [' '.join(bigram) for bigram in bigrams]
                            match_count_1=0
                            for phrase in vocab:
                                if phrase in bigram_strings:
                                    match_count_1+=1
                            st.markdown(f"##### Match {round(match_count_1/(len(vocab))*100,1)} %")
                            st.markdown("#### Audio")
                            with st.container():
                                st.audio(clip, sample_rate=sr)  
                    else:
                            st.markdown(f"<p style='color:green;'>{result}</p>", unsafe_allow_html=True)

                # fake_df = pd.DataFrame(data_1)
                # st.write(fake_df)
                # fake_df['transcript'] = fake_df['transcript'].apply(lambda x: remove_punctuation(x) if isinstance(x, str) else x)
                # fake_df['transcript'] = fake_df['transcript'].apply(lowercase_text)
                # stop_words = set(stopwords.words('english'))
                # fake_df['words'] = fake_df['transcript'].apply(nltk.word_tokenize)


                # fake_df['words'] = fake_df['words'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

                # fake_df['bigrams'] = fake_df['words'].apply(lambda x: list(bigrams(x)))
                # all_bigrams = [bigram for sublist in fake_df['bigrams'] for bigram in sublist]

                # # Count the frequency of each bigram
                # bigram_freq = Counter(all_bigrams)
                # sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1],reverse=True)
                # # print(sorted_bigrams)
                # top_10_bigrams = sorted_bigrams[:100]

                # # Create a list of the top 10 bigrams in string format
                # top_10_bigrams_list = [f"{bigram[0]} {bigram[1]}" for bigram, _ in top_10_bigrams]
                # st.write(top_10_bigrams_list)
                # print(top_10_bigrams)
                # print(top_10_bigrams_list)


                    # for i,row in filtered_df.iterrows():
                    #     if len(row['transcript'].split()) < 15:
                    #         tokens = generate_word_pairs(row['transcript'])
                    #         presence_array = [1 if token in tokens else 0 for token in vocab]
                    #         nonzero_count = np.count_nonzero(presence_array)
                    #         if nonzero_count >=1:
                    #             st.write(row['transcript'])
                    #             st.write(nonzero_count)
                    #             start = pd.to_numeric(row['start'])
                    #             end = pd.to_numeric(row['end'])
                    #             pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                    #             cleaned_url = re.sub(pattern, '',url)
                    #             audio =(f'./audio_files/{cleaned_url}_left.mp3')
                    #             y, sr=librosa.load(audio)
                    #             start_sample = int(start *sr)
                    #             end_sample = int(end *sr)
                    #             clip = y[start_sample:end_sample]
                    #             with st.container():
                    #                 st.audio(clip, sample_rate=sr)    
                                # count+=1
                st.write(count)    
def open_within_5sec():
    st.subheader('\nOpening should be given within a 5-second threshold\n')
    with st.expander("Opening", expanded=False):
        df = pd.read_csv('./merged_preprocessing_lvl_split_segmentation_with_call_duration.csv')
        agent_start_threshold = st.text_input('Opening should be given within [x] seconds threshold', '5')
        call_duration_threshold = st.text_input('Call duration should be more than [x] seconds', '10')
        if st.button('Run Algorithm',key="Opening in 5 seconds"):
            with st.spinner():
                if (agent_start_threshold != ""):
                    st.divider()
                    st.markdown("## Output")
                    for i in combined_unique_urls:
                        pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                        cleaned_url = re.sub(pattern, '',i)
                        dataframes=df[df['url']==i]
                        print(dataframes)
                        mono_left_df = dataframes[dataframes['channel'] == 'Mono_Left']
                        if not mono_left_df.empty:
                            first_start_timestamp_mono_left = pd.to_numeric(mono_left_df['start'].iloc[0])
                            call_duration = pd.to_numeric(mono_left_df['call_duration'].iloc[0])               
                            if first_start_timestamp_mono_left>int(agent_start_threshold) and call_duration > int(call_duration_threshold):
                                st.markdown(f"###### Call ID - {cleaned_url}")
                                st.markdown(f"<div class='custom-red-background'>id_1_1_Fail<br>Reason - (call is over {call_duration_threshold} sec and Agent Started Speaking after {agent_start_threshold} sec)<br>Agent Started At {first_start_timestamp_mono_left}<br> Call duration {call_duration}</div>", unsafe_allow_html=True)
                                play_from=0
                                play_till=15
                                pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                cleaned_url = re.sub(pattern, '',i)
                                audio =(f'./audio_files/{cleaned_url}_left.mp3')
                                st.write(f"<div class='custom-red-background'>Playing Start 15 sec of Audio ...<br>Agent Clip</div>", unsafe_allow_html=True)
                                y, sr=librosa.load(audio)
                                start_sample = int(play_from *sr)
                                end_sample = int(play_till *sr)
                                clip = y[start_sample:end_sample]
                                if mono_left_df['call_duration'].iloc[0] < 15:
                                    play_till=mono_left_df['call_duration'].iloc[0]
                                band_length = play_till-play_from
                                ymargin = 0.05 * band_length
                                fig, ax = plt.subplots(figsize=(10, 0.5))
                                ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                                ax.barh(0, first_start_timestamp_mono_left, left=0, color='red', edgecolor='none')
                                ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                                ax.set_ylim(-0.9, 0.5)
                                ax.axis('off')
                                st.pyplot(fig)
                                st.audio(clip, sample_rate=sr)
                                # st.write(f"<div class='custom-red-background'>Orignal Clip</div>", unsafe_allow_html=True)
                                # audio =(f'./audio_files/{cleaned_url}.mp3')
                                # y, sr=librosa.load(audio)
                                # start_sample = int(play_from *sr)
                                # end_sample = int(play_till *sr)
                                # clip = y[start_sample:end_sample]
                                # st.audio(clip, sample_rate=sr)

def consent_to_record():
    st.subheader('\nObtain customers consent to record call, using a standard script\n')
    with st.expander("Opening", expanded=False):
       st.write("No Calls")
def obtain_customer_name():
    st.subheader('\nObtain the customers name in a respectful manner, using a standard script\n')
    with st.expander("Opening", expanded=False):
        if st.button('Run Algorithm',key="customers name"):
            with st.spinner():
                vocab=['hello क्या', 'से बात', 'बात कर', 'क्या मैं', 'कर रही', 'रही हूं', 'क्या मेरी', 'मेरी बात', 'हो रही', 'से हो', 'रही है']
                df = pd.read_csv('./merged_preprocessing_lvl_split_segmentation_with_call_duration.csv')
                left_df=df[df['channel']=="Mono_Left"]
                left_df = left_df.dropna(subset="transcript")  
                left_df['transcript'] = left_df['transcript'].apply(lambda x: remove_punctuation(x) if isinstance(x, str) else x)
                left_df['transcript'] = left_df['transcript'].apply(lowercase_text)
                unique_url=df['url'].unique()
                for url in unique_url:
                    filtered_df=left_df[left_df['url'] == url]
                    filtered_df=(filtered_df.head(5))
                    for i,row in filtered_df.iterrows():
                        if len(row['transcript'].split()) < 15:
                            tokens = generate_word_pairs(row['transcript'])
                            presence_array = [1 if token in tokens else 0 for token in vocab]
                            nonzero_count = np.count_nonzero(presence_array)
                            if nonzero_count >=3:
                                st.write(row['transcript'])
                                start = pd.to_numeric(row['start'])
                                end = pd.to_numeric(row['end'])
                                pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                cleaned_url = re.sub(pattern, '',url)
                                audio =(f'./audio_files/{cleaned_url}_left.mp3')
                                y, sr=librosa.load(audio)
                                start_sample = int(start *sr)
                                end_sample = int(end *sr)
                                clip = y[start_sample:end_sample]
                                with st.container():
                                    st.audio(clip, sample_rate=sr)



        # # print(bigram_freq)
        # # # Print the most common bigrams

        # # two_word_list = [words for (words, _) in bigram_freq]
        # # st.write(two_word_list)
        # data_1 = []
        # # print(two_word_list)
        # unique_url=df['url'].unique()
        # for url in unique_url:
        #     filtered_df=(df[(df['url'] == url) & (df['channel'] == 'Mono_Left')])
        #     filtered_df=(filtered_df.head(5))
        #     for i,row in filtered_df.iterrows():
                
        #         tokens = generate_word_pairs(row['trnscript'])
        #         presence_array = [1 if token in tokens else 0 for token in vocab]
        #         row['presence_array'] = presence_array
        #         nonzero_count = np.count_nonzero(row['presence_array'])
        #         if nonzero_count >=2:
        #             print(row['transcript_before_deadair'])
        #             print( row['presence_array'])




        #         bag_of_words = get_bag_of_words(row['transcript'], vocab)
        #         values_list = list(bag_of_words.values())
        #         data = np.array(values_list)
        #         count = np.count_nonzero(data)
        #         if count>4  and len(row['transcript'].split())<14:
        #             st.write(row['transcript'])
        #             row_1 = {
        #             "index": i,
        #             'transcript': row['transcript']}
        #             data_1.append(row_1)
        # fake_df = pd.DataFrame(data_1)
        # st.write(fake_df)
        # fake_df['transcript'] = fake_df['transcript'].apply(lambda x: remove_punctuation(x) if isinstance(x, str) else x)
        # fake_df['transcript'] = fake_df['transcript'].apply(lowercase_text)
        # stop_words = set(stopwords.words('english'))
        # fake_df['words'] = fake_df['transcript'].apply(nltk.word_tokenize)


        # fake_df['words'] = fake_df['words'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

        # fake_df['bigrams'] = fake_df['words'].apply(lambda x: list(bigrams(x)))
        # all_bigrams = [bigram for sublist in fake_df['bigrams'] for bigram in sublist]

        # # Count the frequency of each bigram
        # bigram_freq = Counter(all_bigrams)
        # sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1],reverse=True)
        # top_10_bigrams = sorted_bigrams[:20]

        # # Create a list of the top 10 bigrams in string format
        # top_10_bigrams_list = [f"{bigram[0]} {bigram[1]}" for bigram, _ in top_10_bigrams]
        # print(top_10_bigrams_list)

        # # If you want to display the sorted bigrams
        # for bigram, freq in sorted_bigrams:
        #     st.write(f"{bigram}: {freq}")
        #             start = pd.to_numeric(row['start'])
        #             end = pd.to_numeric(row['end'])
        #             pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
        #             cleaned_url = re.sub(pattern, '',url)
        #             audio =(f'./audio_files/{cleaned_url}_left.mp3')
        #             st.write(f"<div class='custom-red-background'>Agent Clip</div>", unsafe_allow_html=True)
        #             y, sr=librosa.load(audio)
        #             start_sample = int(start *sr)
        #             end_sample = int(end *sr)
        #             clip = y[start_sample:end_sample]
        #             with st.container():
        #                 st.audio(clip, sample_rate=sr)

        st.write("Pass")      
def precall_silence():
    st.subheader('\nIf the customers voice is inaudible from the beginning of the call, disconnect after 10 seconds\n')
    with st.expander("Opening", expanded=False):
        st.write("Pass")  
def apologize_for_redial_calls():
    st.subheader('Apologize for redial calls if the call got disconnected, using a standard script')
    with st.expander("Redial Calls", expanded=False):
        if st.button('Run Algorithm',key="redial call"):
            with st.spinner():
                vocab = ['कुछ दिन','last time']
                df=df_split
                left_df=df[df['channel']=="Mono_Left"]
                left_df = left_df.dropna(subset="transcript")  
                left_df['transcript'] = left_df['transcript'].apply(lambda x: remove_punctuation(x) if isinstance(x, str) else x)
                left_df['transcript'] = left_df['transcript'].apply(lowercase_text)
                                # # Vocabulary
                # vocab = ['ok क्या', 'एक minute','minute दीजिए','wait कर','2 minutes','hold करिए','minute मुझे','minute हां']
                # # Function to tokenize text into pairs of two consecutive words
                # def generate_word_pairs(text):
                #     words = word_tokenize(text)
                #     word_pairs = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
                #     return word_pairs
                # Function to create array indicating presence of vocabulary items
                for index,row in left_df.iterrows():
                    tokens = generate_word_pairs(row['transcript'])
                    presence_array = [1 if token in tokens else 0 for token in vocab]
                    row['presence_array'] = presence_array
                    nonzero_count = np.count_nonzero(row['presence_array'])
                    if nonzero_count >=1:
                        st.write("Redial Sentence")
                        st.write(row['transcript'])
                        play_from=row['start']
                        play_till=row['end']
                        pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                        cleaned_url = re.sub(pattern, '',row['url'])
                        audio =(f'./audio_files/{cleaned_url}_left.mp3')
                        y, sr=librosa.load(audio)

                        start_sample = int(play_from *sr)
                        end_sample = int(play_till *sr)
                        clip = y[start_sample:end_sample]
                        with st.container():
                            st.audio(clip, sample_rate=sr)   
    
                    
                # Apply the function to each row


def intro_for_redial_calls():
    st.subheader('Give a short introduction, for redial calls, using a standard script')
    with st.expander("Redial Calls", expanded=False):
        if st.button('Run Algorithm',key="redial call + intro"):
            with st.spinner():
                vocab = ['कुछ दिन','last time']
                df=df_split
                left_df=df[df['channel']=="Mono_Left"]
                left_df = left_df.dropna(subset="transcript")  
                left_df['transcript'] = left_df['transcript'].apply(lambda x: remove_punctuation(x) if isinstance(x, str) else x)
                left_df['transcript'] = left_df['transcript'].apply(lowercase_text)
                                # # Vocabulary
                # vocab = ['ok क्या', 'एक minute','minute दीजिए','wait कर','2 minutes','hold करिए','minute मुझे','minute हां']
                # # Function to tokenize text into pairs of two consecutive words
                # def generate_word_pairs(text):
                #     words = word_tokenize(text)
                #     word_pairs = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
                #     return word_pairs
                # Function to create array indicating presence of vocabulary items
                for index,row in left_df.iterrows():
                    tokens = generate_word_pairs(row['transcript'])
                    presence_array = [1 if token in tokens else 0 for token in vocab]
                    row['presence_array'] = presence_array
                    nonzero_count = np.count_nonzero(row['presence_array'])
                    regex = r'dot com'
                    if nonzero_count >=1:
                        sentence = " ".join(left_df[left_df['channel'] == 'Mono_Left']['transcript'].head(3))

                        match = re.search(regex, sentence)
                        if match:
                            print("Redial Sentence + Intro")
                            print(sentence)
                            st.write(left_df[left_df['channel'] == 'Mono_Left']['transcript'].head(3))
                            print(True) 
                            print("-"*50)
                            print()
                            st.write("Redial Sentence")
                            st.write(row['transcript'])
                            play_from=row['start']
                            play_till=row['end']
                            pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                            cleaned_url = re.sub(pattern, '',row['url'])
                            audio =(f'./audio_files/{cleaned_url}_left.mp3')
                            y, sr=librosa.load(audio)

                            start_sample = int(play_from *sr)
                            end_sample = int(play_till *sr)
                            clip = y[start_sample:end_sample]
                            with st.container():
                                st.audio(clip, sample_rate=sr) 


def interruption():
    st.subheader('Avoiding interrupting the customer during conversation')
    with st.expander("Behaviour", expanded=False):
        if st.button('Run Algorithm',key="interruption"):
            with st.spinner():
                for url in split_unique_urls:
                    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                    cleaned_url = re.sub(pattern, '',url)
                    df=df_split[df_split['url']==url]
                    df_left=df[df['channel']=='Mono_Left']
                    df_right=df[df['channel']=='Mono_Right']
                    call_duration=df['audio_duration'].iloc[0]
                    overlapping_info_1 = pd.DataFrame(columns=['url','start', 'end', 'paramater_label','date','call_duration', 'overlapping_df'])
                    for index_left, row_left in df_left.iterrows():
                        for index_right, row_right in df_right.iterrows():
                            overlap_start = max(row_left['start'], row_right['start'])
                            overlap_end = min(row_left['end'], row_right['end'])
                            overlap_duration = overlap_end - overlap_start
                            if overlap_start < overlap_end and overlap_duration >= 4:
                                if row_left['start'] <= row_right['start']:
                                    overlapping_df = 'Mono_Right'
                                else:
                                    overlapping_df = 'Mono_Left'
                                data_to_append = {
                                    'start': overlap_start,
                                    'end': overlap_end,
                                    'call_duration':call_duration,
                                    'overlapping_df': overlapping_df
                                }
                                df_to_append = pd.DataFrame([data_to_append])
                                #non_empty_columns = df_to_append.columns[df_to_append.notna().any()]
                                overlapping_info_1 = pd.concat([overlapping_info_1, df_to_append], ignore_index=True)
                    agent_overlaps=overlapping_info_1[overlapping_info_1['overlapping_df']=='Mono_Left']

                    if not agent_overlaps.empty:
                        # st.markdown("### Entire Call")
                        band_length = agent_overlaps["call_duration"].iloc[0]
                        fig, ax = plt.subplots(figsize=(10, 0.5))
                        ymargin = 0.05 * band_length
                        ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                        ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                        ax.set_ylim(-0.9, 0.5)
                        ax.axis('off')

                        
                        for index,rows in agent_overlaps.iterrows():
                            st.markdown(f"###### From [{rows['start']}] To [{rows['end']}] seconds Overlap of [{round(rows['end']-rows['start'],2)}] seconds")
                            ax.barh(0, rows['end']-rows['start'], left=rows['start'], color='red', edgecolor='none')
                        st.pyplot(fig)
                        audio =(f'./audio_files/{cleaned_url}.mp3')
                        y, sr=librosa.load(audio)
                        with st.container():
                            st.audio(y, sample_rate=sr)


def proactive_cancellation():
    st.subheader('No cutting the call while the customer is speaking')
    with st.expander("Behaviour", expanded=False):
        min_value = st.number_input('Minimum value',value=-0.006)
        st.write('Minimum value is ', min_value)
        max_value = st.number_input('Maximum value',value=0.006)
        st.write('Maximum value is', max_value)
        selected_channel=st.selectbox("Channel", ["Customer", "Agent"])
        if st.button('Process Matric 3_2'):
            for i in split_unique_urls:
                dataframes=df_split[df_split['url']==i]
                mono_left_df = dataframes[dataframes['channel'] == 'Mono_Left']
                mono_right_df = dataframes[dataframes['channel'] == 'Mono_Right']
                #st.dataframe(mono_left_df.iloc[-1])
                pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                cleaned_url = re.sub(pattern, '', i)
                if selected_channel=="Agent":
                    audio =(f'./audio_files/{cleaned_url}_left.mp3')
                elif selected_channel=="Customer":
                    audio =(f'./audio_files/{cleaned_url}_right.mp3')
                y, sr=librosa.load(audio)
                smoothed_y = np.convolve(y, np.ones(100)/100, mode='valid')
                if not (min_value <= smoothed_y[-1] <= max_value):
                    st.write(smoothed_y[-1], f'is between {min_value} and {max_value}')
                    st.markdown(f"<div class='custom-green-background'>Pass</div>", unsafe_allow_html=True)
                    st.header(f'(no of seconds x {sr})  Frames')
                    st.header('Last 1000 Frames')
                    plt.figure(figsize=(10, 8))
                    plt.plot(smoothed_y[-100:])
                    plt.yticks([min_value, 0, max_value])
                    plt.tick_params(axis='y', labelright=True, right=True)
                    plt.plot([0, 100], [min_value, min_value], color='red')
                    plt.plot([0, 100], [max_value, max_value], color='red')
                    st.pyplot(plt)
                    with st.container():
                    #st.markdown(f"<div style='background-color:#aaf683; padding-top: 10px;'>", unsafe_allow_html=True)
                        st.audio(y, sample_rate=sr)
                else:
                    pass_state = False
         
def regular_changeover():
    st.subheader('Engaging in a responsive manner with regular changeovers')
    with st.expander("Behaviour", expanded=False):
        if st.button('Run Algorithm',key="regular_changeover"):
            with st.spinner():
                for url in split_unique_urls:
                    deadair_df=df_split[df_split['url']==url]
                    deadair_df['start'] = pd.to_numeric(deadair_df['start'], errors='coerce')
                    deadair_df['end'] = pd.to_numeric(deadair_df['end'], errors='coerce')
                    deadair_df=deadair_df.sort_values(by='start')
                    left_starts=[]
                    right_starts=[]
                    st.markdown("#### Changover")
                    for i in range(0, len(deadair_df)):
                        if deadair_df['channel'].iloc[i] != deadair_df['channel'].iloc[i-1]:  # Check if the current element is different from the previous one
                            st.write(f"Turn {i} [{deadair_df['channel'].iloc[i]}] From [{deadair_df['start'].iloc[i]}]")
                            if deadair_df['channel'].iloc[i]=="Mono_Left":
                                left_starts.append(deadair_df['start'].iloc[i])
                            if deadair_df['channel'].iloc[i]=="Mono_Right":
                                right_starts.append(deadair_df['start'].iloc[i])
                    if len(left_starts) !=0 or len(right_starts) !=0:
                        max_val = max(max(left_starts), max(left_starts))
                        num_groups = int(np.ceil(max_val / 60))
                        groups = [(k * 60 + 1, (k + 1) * 60) for k in range(num_groups)]
                        df_freq = pd.DataFrame(index=range(num_groups), columns=['group', 'x_freq', 'y_freq'])
                        for l, (start, end) in enumerate(groups):
                            x_freq = sum(1 for val in left_starts if start <= val <= end)
                            y_freq = sum(1 for val in right_starts if start <= val <= end)
                            df_freq.loc[l] = [(start, end), x_freq, y_freq]
                        x_mean = df_freq['x_freq'].mean()
                        y_mean = df_freq['y_freq'].mean()
                        x_std = df_freq['x_freq'].std()
                        y_std = df_freq['y_freq'].std()
                        st.markdown("#### Frequency Distribution")
                        df_freq['x_label'] = np.where(df_freq['x_freq'] < 3, 'inactive', 'active')
                        df_freq['y_label'] = np.where(df_freq['y_freq'] < 3, 'inactive', 'active')
                        st.write(df_freq)

def moderate_pace():
    st.subheader('Maintain a moderate pace throughout the call')
    with st.expander("Behaviour", expanded=False):
        wpm_threshold = st.text_input('pace should be less than [x] wpm', '200')
        if st.button('Run Algorithm',key="moderate_pace"):
            with st.spinner():
                moderate_pace=df_split[df_split['channel']=="Mono_Left"]
                moderate_pace = moderate_pace.dropna(subset=['transcript'])
                for url in split_unique_urls:
                    df=moderate_pace[moderate_pace['url']==url]
                    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                    cleaned_url = re.sub(pattern, '',url)
                    for index,row in df.iterrows():
                        transcript=str(row['transcript'])
                        start=pd.to_numeric(row['start'])
                        end=pd.to_numeric(row['end'])
                        duration=end-start
                        wpm=(len(transcript.split()) / (duration/60))
                        if wpm > pd.to_numeric(wpm_threshold) and duration > 10:
                            st.markdown(f"##### WPM [{wpm}]")
                            play_from=row['start']
                            play_till=row['end']
                            audio =(f'./audio_files/{cleaned_url}_left.mp3')
                            y, sr=librosa.load(audio)
                            st.markdown("### Fast Pace Audio")
                            start_sample = int(play_from *sr)
                            end_sample = int(play_till *sr)
                            clip = y[start_sample:end_sample]
                            with st.container():
                                st.audio(clip, sample_rate=sr)


def empathy():
    st.subheader('Use Empathy/Pleasantries/Reassurance throughout the call such as "Thank You", "Sorry", "Please do not worry"')
    with st.expander("Behaviour", expanded=False):
        if st.button('Run Algorithm',key="empathy"):
            with st.spinner():
                for url in split_unique_urls[:20]:
                    
                    st.markdown(f"#### Call Id [{get_url(url)}]")
                    deadair_df=df_split[(df_split['url']==url) & (df_split['channel']=="Mono_Left")]
                    if not deadair_df.empty:
                        left_starts=[]
                        deadair_df['start'] = pd.to_numeric(deadair_df['start'], errors='coerce')
                        deadair_df['end'] = pd.to_numeric(deadair_df['end'], errors='coerce')
                        deadair_df=deadair_df.sort_values(by='start')
                        empathy_words_file = "./input_csv/empathy_words.csv"  # Replace "empathy_words.csv" with your file path
                        empathy_words_df = pd.read_csv(empathy_words_file)
                        empathy_words = set(empathy_words_df['transcript'].str.lower())
                        deadair_df['empathy_count'] = deadair_df['transcript'].apply(lambda x: count_empathy_phrases(x, empathy_words)[0])
                        deadair_df['empathy_words'] = deadair_df['transcript'].apply(lambda x: count_empathy_phrases(x, empathy_words)[1])
                        st.write(deadair_df[['start','end','transcript','empathy_words','empathy_count']])
                        x=deadair_df['start'].to_list()
                        max_val = max(x)
                        num_groups = int(np.ceil(max_val / 60))
                        groups = [(i * 60 + 1, (i + 1) * 60) for i in range(num_groups)]
                        df_freq = pd.DataFrame(index=range(num_groups), columns=['per_min_distriution', 'words_used', 'word_count'])
                        st.markdown("#### Empathy Words Used per Min Frequency")
                        for l, (start, end) in enumerate(groups):
                            per_minute_df=deadair_df[(deadair_df['start'] >= start) & (deadair_df['end'] <= end)]   
                            counts = per_minute_df['empathy_count']
                            count_list = counts.tolist()
                            words = per_minute_df['empathy_words']
                            words_list = words.tolist()
                            result = []
                            for sublist in words_list:
                                if sublist:
                                    result.extend(sublist)
                            df_freq.loc[l] = [(start, end), sum(count_list),result]
                        st.write(df_freq)
                        st.divider()

def language_switch_as_per_customer_response():
    st.subheader('Agent needs to switch the language as per the response provided by the customer')
    with st.expander("Language", expanded=False):
        language_switch_threshold = st.text_input('Language change threshold [x] times', '5')
        if st.button('Run Algorithm',key="language_switch"):
            with st.spinner():
                for url in split_unique_urls:
                    df=df_split[(df_split['url']==url)]
                    df=df.sort_values(by='start')
                    df = df.reset_index(drop=True)
                    df_left = df[df['channel']=='Mono_Left']
                    df_right = df[df['channel']=='Mono_Right']
                    if len(df_left) !=0 and len(df_right) !=0:
                        switches = []
                        for index,row in df.iterrows():
                            current_speaker = row['channel']
                            if index > 0:
                                prev_row = df.iloc[index - 1]
                                prev_lang=(prev_row['language-detection_roberta-base-model'])
                                current_lang=row['language-detection_roberta-base-model']
                                transcript=str(row['transcript'])
                                prev_transcript = str(prev_row['transcript'])
                                previous_start = prev_row['start']
                                current_end=  row['end']
                                previous_speaker=prev_row['channel']
                                if prev_lang != current_lang:
                                    switches.append((row['start'], previous_speaker,current_speaker,previous_start,current_end,prev_lang,prev_transcript, current_lang, transcript))
                        if len(switches)>pd.to_numeric(language_switch_threshold):
                            band_length = df["audio_duration"].iloc[0]
                            bar_added = False
                            # Loop to check the condition first
                            for start_time, previous_speaker, current_speaker, previous_start, current_end, prev_lang, prev_transcript, current_lang, transcript in switches:
                                if current_speaker == "Mono_Left" and len(prev_transcript.strip().split()) > 5 and len(transcript.strip().split()) > 5:
                                    bar_added = True
                                    break

                            # Create plot only if bar is added
                            if bar_added:
                                fig, ax = plt.subplots(figsize=(10, 0.5))
                                ymargin = 0.05 * band_length
                                ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                                ax.set_xlim(0 - (0.07 * band_length), band_length + ymargin)
                                ax.set_ylim(-0.9, 0.5)
                                ax.axis('off')
                                
                                # Loop again to add bars
                                for start_time, previous_speaker, current_speaker, previous_start, current_end, prev_lang, prev_transcript, current_lang, transcript in switches:
                                    if current_speaker == "Mono_Left" and len(prev_transcript.strip().split()) > 5 and len(transcript.strip().split()) > 5:
                                        ax.barh(0, current_end - previous_start, left=previous_start, color='red', edgecolor='none')
                                st.divider()
                                pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                cleaned_url = re.sub(pattern, '',url)
                                st.markdown(f"#### {cleaned_url}")

                                st.pyplot(fig)
                                
                                audio =(f'./audio_files/{cleaned_url}.mp3')
                                y, sr=librosa.load(audio)
        
                                with st.container():
                                    st.audio(y, sample_rate=sr)



                            for start_time, previous_speaker,current_speaker, previous_start,current_end, prev_lang,prev_transcript,current_lang ,transcript in switches:
                                if current_speaker=="Mono_Left" and len(prev_transcript.strip().split()) > 5 and len(transcript.strip().split())>5:
                                    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                    cleaned_url = re.sub(pattern, '',url)

                                    st.write(len(prev_transcript.strip().split()))
                                    st.markdown(f"#### Language switch detected at {round(start_time,2)} ") 
                                    st.markdown(f"##### [{prev_lang}] [{previous_speaker}] -> {prev_transcript}")
                                    st.markdown(f"##### [{current_lang}] [{current_speaker}] -> {transcript}")
                                    play_from=previous_start
                                    play_till=current_end
                                    audio =(f'./audio_files/{cleaned_url}.mp3')
                                    y, sr=librosa.load(audio)
                                    start_sample = int(play_from *sr)
                                    end_sample = int(play_till *sr)
                                    clip = y[start_sample:end_sample]
                                    with st.container():
                                        st.audio(clip, sample_rate=sr)
def language_switchs_the_language_against_entire_call():
    st.subheader('Agent should not switch language to a different language than what the customer is using')
    with st.expander("Language", expanded=False):
        language_switch_threshold = st.text_input('Language switch threshold [x] times', '1')
        vocab = ['ok क्या', 'एक minute','minute दीजिए','wait कर','2 minutes','hold करिए','minute मुझे','minute हां']
        if st.button('Run Algorithm',key="language_switch_2"):
            with st.spinner():
                for url in split_unique_urls:
                    df=df_split[(df_split['url']==url)]
                    df=df.sort_values(by='start')
                    df = df.reset_index(drop=True)
                    df_left = df[df['channel']=='Mono_Left']
                    df_right = df[df['channel']=='Mono_Right']
                    if len(df_left) !=0 and len(df_right) !=0:
                        switches = []
                        for index,row in df.iterrows():
                            current_speaker = row['channel']
                            if index > 0:
                                prev_row = df.iloc[index - 1]
                                prev_lang=(prev_row['language-detection_roberta-base-model'])
                                current_lang=row['language-detection_roberta-base-model']
                                transcript=str(row['transcript'])
                                prev_transcript = str(prev_row['transcript'])
                                previous_start = prev_row['start']
                                current_end=  row['end']
                                previous_speaker=prev_row['channel']
                                if prev_lang != current_lang:
                                    switches.append((row['start'], previous_speaker,current_speaker,previous_start,current_end,prev_lang,prev_transcript, current_lang, transcript))
                        if len(switches)>pd.to_numeric(language_switch_threshold):
                            for start_time, previous_speaker,current_speaker, previous_start,current_end, prev_lang,prev_transcript,current_lang ,transcript in switches:
                                if current_speaker=="Mono_Left" and previous_speaker=="Mono_Left":
                                    if current_speaker=="Mono_Left" and len(prev_transcript.strip().split()) > 5 and len(transcript.strip().split())>5:
                                        pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                        cleaned_url = re.sub(pattern, '',url)
                                        st.markdown(f"#### {cleaned_url}")
                                        st.write(len(prev_transcript.strip().split()))
                                        st.markdown(f"#### Language switch detected at {round(start_time,2)} ") 
                                        st.markdown(f"##### [{prev_lang}] [{previous_speaker}] -> {prev_transcript}")
                                        st.markdown(f"##### [{current_lang}] [{current_speaker}] -> {transcript}")
                                        play_from=previous_start
                                        play_till=current_end
                                        audio =(f'./audio_files/{cleaned_url}.mp3')
                                        y, sr=librosa.load(audio)
                                        start_sample = int(play_from *sr)
                                        end_sample = int(play_till *sr)
                                        clip = y[start_sample:end_sample]
                                        with st.container():
                                            st.audio(clip, sample_rate=sr)
def count_hold_keywords(sentence):
    hold_words=['hold','minute','wait','minutes']
    doc = nlp(sentence.lower())
    #tokens_no_punct = [token.text for token in doc if not token.is_punct]
    # st.write(doc)
    # tokens = [token.text for token in doc]
    # count = (True for token in doc if token in hold_words)
    found_hold = any(token.text in hold_words for token in doc)
    if found_hold:
        st.write("Hold keyword found!")
        return True
    else:
        st.write("No hold keyword found.")
        return False
def hold():
    st.subheader('Confirm with the Customer before placing the call on hold, using a standard script')
    with st.expander("Process", expanded=False):
        # hold_threshold = st.text_input('Hold threshold', '5')
        vocab = ['ok क्या', 'एक minute','minute दीजिए','wait कर','2 minutes','hold करिए','minute मुझे','minute हां']
        if st.button('Run Algorithm',key="Hold"):
            with st.spinner():
                df=df_combined[df_combined['channel']=="Merged"]
                # df_no_null = df.dropna(subset=['transcript_before_deadair'])
                combined_unique_urls = df['url'].unique()
                for url in combined_unique_urls:
                    hold_df=df[df['url']==url]
                    # st.write(hold_df)
                    pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                    cleaned_url = re.sub(pattern, '',url)
                    for index,row in hold_df.iterrows():
                        if str(row['transcript_before_deadair']) != 'nan':
                            if index != 0:
                                        previous_index = index - 1
                                        # Check if previous_index exists in hold_df's index
                                        if previous_index in hold_df.index:
                                            tokens = generate_word_pairs(row['transcript_before_deadair'])
                                            # st.write(tokens)
                                            presence_array = [1 if token in tokens else 0 for token in vocab]
                                            row['presence_array'] = presence_array
                                            nonzero_count = np.count_nonzero(row['presence_array'])
                                            if nonzero_count >=2:
                                                st.markdown(f"### Hold Duration [{row['dead_air']}]")
                                                previous_start=(hold_df.loc[previous_index]['start'])
                                                previous_end=(hold_df.loc[previous_index]['end'])
                                                play_from=previous_start#-row['dead_air']-5
                                                play_till=row['end']
                                                audio =(f'./audio_files/{cleaned_url}_left.mp3')
                                                y, sr=librosa.load(audio)
                                                start_sample = int(play_from *sr)
                                                end_sample = int(play_till *sr)
                                                clip = y[start_sample:end_sample]
                                                band_length = play_till-play_from
                                                ymargin = 0.05 * band_length
                                                fig, ax = plt.subplots(figsize=(10, 0.5))
                                                ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                                                ax.barh(0, row['dead_air'], left=previous_end-previous_start, color='red', edgecolor='none')
                                                ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                                                ax.set_ylim(-0.9, 0.5)
                                                ax.axis('off')

                                                st.pyplot(fig)
                                                with st.container():
                                                    st.audio(clip, sample_rate=sr)
                                        else:
                                            st.write(f"Previous index {previous_index} does not exist in the DataFrame")

def dead_air_more_than_5():
    st.subheader('No dead air on the call for more than 5 seconds, at any point')
    with st.expander("Audio Quality", expanded=False):
        dead_air_threshold = st.text_input('Dead air threshold', '5')
        if st.button('Run Algorithm',key="Dead Air"):
            with st.spinner():
                input_contents = []  # let the user input all the data
                if (dead_air_threshold != ""):
                    st.divider()
                    input_contents.append(str(dead_air_threshold))
                    st.markdown("## Output")
                    for i in combined_unique_urls:
                        deadair_df=df_combined[df_combined['url']==i]
                        for index, row in deadair_df.iterrows():
                            if row['dead_air'] > float(dead_air_threshold):
                                pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                cleaned_url = re.sub(pattern, '',i)
                                st.markdown(f"#### Call ID {cleaned_url}")
                                st.markdown(f"##### Dead Air of [{round(row['dead_air'],2)}] seconds")
                                st.markdown(f"##### From [{row['start'] - row['dead_air']}] To [{row['start']}] seconds ")
                                # st.markdown("### Entire Call")
                                play_from=row['start'] - row['dead_air'] - 5
                                play_till=row['end']
                                pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                cleaned_url = re.sub(pattern, '',row['url'])
                                audio =(f'./audio_files/{cleaned_url}.mp3')
                                y, sr=librosa.load(audio)
                                # with st.container():
                                #     st.audio(y, sample_rate=sr)
                                st.markdown("### Dead Air Audio")
                                start_sample = int(play_from *sr)
                                end_sample = int(play_till *sr)
                                clip = y[start_sample:end_sample]
                                band_length = play_till-play_from
                                ymargin = 0.05 * band_length

                                fig, ax = plt.subplots(figsize=(10, 0.5))
                                ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                                ax.barh(0, row['dead_air'], left=5, color='red', edgecolor='none')
                                ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                                ax.set_ylim(-0.9, 0.5)
                                ax.axis('off')

                                st.pyplot(fig)
                                with st.container():
                                    st.audio(clip, sample_rate=sr)

                               
                                st.divider()
def sum_of_dead_air_more_than_40():
    st.subheader('No total dead air of more than 40 seconds in the call')
    with st.expander("Audio Quality", expanded=False):
        dead_air_threshold = st.text_input('Sum of dead air Threshold', '40')
        if st.button('Run Algorithm',key="Sum of Dead Air"):
            with st.spinner():
                input_contents = []
                if (dead_air_threshold != ""):
                    st.divider()
                    input_contents.append(str(dead_air_threshold))
                    st.markdown("## Output")
                    for i in combined_unique_urls:
                        pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                        cleaned_url = re.sub(pattern, '',i)
                        deadair_df=df_combined[df_combined['url']==i]
                        deadair_df['dead_air_2'] = np.where(deadair_df['dead_air'] >= 6, deadair_df['dead_air'], 0)
                        if deadair_df['dead_air_2'].sum() > 40:
                            deadair_filtered=deadair_df[deadair_df['dead_air_2']!=0]
                            st.markdown(f"#### Call ID {cleaned_url}")
                            st.markdown(f"##### Dead Air was taken at Multiple instances")
                            band_length = deadair_df['call_duration'].iloc[0]
                            fig, ax = plt.subplots(figsize=(10, 0.5))
                            ymargin = 0.05 * band_length
                            ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                            ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                            ax.set_ylim(-0.9, 0.5)
                            ax.axis('off')
                            st.markdown("### Entire Call")
                            for index,rows in deadair_filtered.iterrows():
                                st.markdown(f"###### From [{rows['start']}] To [{rows['end']}] seconds  Dead air of [{round(rows['dead_air'],2)}] seconds")
                                ax.barh(0, rows['dead_air_2'], left=rows['start']-rows['dead_air_2'], color='red', edgecolor='none')
                            st.pyplot(fig)
                            audio =(f'./audio_files/{cleaned_url}.mp3')
                            y, sr=librosa.load(audio)
                            with st.container():
                                st.audio(y, sample_rate=sr)
                            #For Playing Clip wise
                            ###################
                            # st.markdown("### Dead Air Clips")
                            # for index,row in deadair_filtered.iterrows():
                            #     play_from=row['start'] - row['dead_air'] - 5
                            #     play_till=row['end']
                            #     pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                            #     cleaned_url = re.sub(pattern, '',row['url'])
                            #     start_sample = int(play_from *sr)
                            #     end_sample = int(play_till *sr)
                            #     clip = y[start_sample:end_sample]
                            #     with st.container():
                            #         st.audio(clip, sample_rate=sr)    
                            ############################    

def network_issue():
    st.subheader('If the customers voice is not audible due to network problems, check for 20 seconds and disconnect the call.')
    with st.expander("Audio Quality", expanded=False):
        network_df = pd.read_csv("./network_dataset.csv")
        if st.button('Run Algorithm',key="network issue"):
            with st.spinner():
                count_vectorizer = CountVectorizer(max_features=1000)
                X_intro = count_vectorizer.fit_transform(network_df['transcript'])  
                def predict(sentence):
                    X_sentence = count_vectorizer.transform([sentence])
                    similarities = cosine_similarity(X_sentence, X_intro)
                    max_similarity_index = similarities.argmax()
                    if similarities[0, max_similarity_index] > 0.87:  # Adjust threshold as needed
                        return 1  # Introduction
                    else:
                        return 0  # Not an introduction
                df=df_split
                df.dropna(subset=['transcript'], inplace=True)
                for url in split_unique_urls:
                    filtered_df = df[df['url'] == url]
                    filtered_df.sort_values(by='start')
                    sorted_df=filtered_df.sort_values(by='start')
                    previous_right_timestamp=0
                    for i, row in sorted_df.iterrows():
                        sentence = row['transcript']
                        start = row['start']
                        end = row['end']
                        channel = row['channel']
                        # print(channel)
                        # print(round(end,2))
                        if channel == 'Mono_Right':
                            previous_right_timestamp = end  # Update timestamp
                        if channel == 'Mono_Left':
                            prediction = predict(sentence)

                            if prediction==1:
                                st.write(sentence)
                                st.write(f"Last Response of customer {round(previous_right_timestamp,2)}")
                                st.write(f"Current Sentence Timestamp {round(start,2)}  {round(end,2)}")
                                st.write(f"Difference {round(start,2)- round(previous_right_timestamp,2)}")
                                play_from=start
                                play_till=end
                                pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                cleaned_url = re.sub(pattern, '',row['url'])
                                audio =(f'./audio_files/{cleaned_url}_left.mp3')
                                y, sr=librosa.load(audio)

                                start_sample = int(play_from *sr)
                                end_sample = int(play_till *sr)
                                clip = y[start_sample:end_sample]
                                with st.container():
                                    st.audio(clip, sample_rate=sr)    



                
def post_call_silence():
    st.subheader('Disconnect the call within 10 seconds of conversation being completed')
    with st.expander("Audio Quality", expanded=False):
        post_call_silence_more_than_threshold = st.text_input('opening should be given within [x] seconds threshold', '5')
        call_duration_threshold = st.text_input('call duration should be more than [x] seconds', '30')
        if st.button('Run Algorithm',key="Closing within 10"):
            with st.spinner():
                if (post_call_silence_more_than_threshold != ""):
                    st.divider()
                    st.markdown("## Output")
                    for i in combined_unique_urls:
                        pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                        cleaned_url = re.sub(pattern, '',i)
                        dataframes=df_combined[df_combined['url']==i]
                        #mono_left_df = dataframes[dataframes['channel'] == 'Mono_Left']
                        if not dataframes.empty:
                            last_speech_end_timestamp = pd.to_numeric(dataframes['end'].iloc[-1])

                            call_duration = pd.to_numeric(dataframes['call_duration'].iloc[0])               
                            if (call_duration-last_speech_end_timestamp) > int(post_call_silence_more_than_threshold) and call_duration > pd.to_numeric(call_duration_threshold):
                                st.markdown(f"<div class='custom-red-background'>3_1_Fail<br>Reason - (Call Was Not disconnected post call after  Even After {post_call_silence_more_than_threshold} sec when Agent and client stopped talking)<br>Last Spoken Time Stamp (Agent or Customer){last_speech_end_timestamp}<br>Customer First Response {last_speech_end_timestamp}<br>Call disconnected at {dataframes['call_duration'].iloc[0]}<br>Playing Last 15 seconds of the call</div>", unsafe_allow_html=True)
                               
                                pattern = re.compile(r'https://s3-ap-southeast-1\.amazonaws\.com/exotelrecordings/futwork1/|\.mp3$')
                                cleaned_url = re.sub(pattern, '',i)
                                audio =(f'./audio_files/{cleaned_url}.mp3')
                                st.write(f"<div class='custom-red-background'>Agent Clip</div>", unsafe_allow_html=True)
                                y, sr=librosa.load(audio)
                                band_length =call_duration
                                st.write(call_duration)
                                st.write(call_duration-last_speech_end_timestamp)
                                st.write(last_speech_end_timestamp)

                                fig, ax = plt.subplots(figsize=(10, 0.5))
                                ymargin = 0.05 * band_length
                                ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                                ax.barh(0, call_duration-last_speech_end_timestamp, left=last_speech_end_timestamp, color='red', edgecolor='none')
                                ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                                ax.set_ylim(-0.9, 0.5)
                                ax.axis('off')
                                st.markdown("### Entire Call")
                                st.pyplot(fig)
                                with st.container():
                                    st.audio(y, sample_rate=sr)

                                # band_length = play_till-play_from
                                # ymargin = 0.05 * band_length
                                # fig, ax = plt.subplots(figsize=(10, 0.5))
                                # ax.barh(0, band_length, color='#94FFD8', edgecolor='none')
                                # ax.barh(0, first_start_timestamp_mono_left, left=0, color='red', edgecolor='none')
                                # ax.set_xlim(0 - (0.07* band_length), band_length + ymargin)
                                # ax.set_ylim(-0.9, 0.5)
                                # ax.axis('off')
                                # st.pyplot(fig)
                               

if __name__ == "__main__":
    # human_disturbance()
    # background()
    # dull_1()
    open_with_standard_script()
    open_within_5sec()
    consent_to_record()
    obtain_customer_name()
    precall_silence()
    apologize_for_redial_calls()
    intro_for_redial_calls()
    interruption()
    proactive_cancellation()
    regular_changeover()
    moderate_pace()
    empathy()
    language_switch_as_per_customer_response()
    language_switchs_the_language_against_entire_call()
    hold()
    dead_air_more_than_5()
    sum_of_dead_air_more_than_40()
    network_issue()
    post_call_silence()




