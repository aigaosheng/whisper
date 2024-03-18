import whisper

model = whisper.load_model("base") #"large-v3")

audio = whisper.load_audio("/home/gs/Documents/zhaozhongxiang.wav")#eng.mp3")#
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels = 80).to(model.device)

# # detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions(beam_size=5)#, best_of=5)
# options = dict(language=language, beam_size=5, best_of=5)

result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

# emb = model.embed_audio(mel[None,...])