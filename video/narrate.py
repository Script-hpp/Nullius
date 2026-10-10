"""Local English speech and timed subtitles; no online TTS or paid requests."""
import json,subprocess,wave
from pathlib import Path
out=Path(__file__).resolve().parent/'out'
rows=json.loads((out/'timeline.json').read_text());sr=22050;audio=bytearray(150*sr*2)
def timestamp(s):
 ms=round(s*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
srt=[];n=0
for i,r in enumerate(rows):
 voice=out/f'voice-{i:02}.wav'
 subprocess.run(['espeak-ng','-v','en-us','-s','180','-w',str(voice),r['voice']],check=True)
 with wave.open(str(voice)) as w:
  assert w.getframerate()==sr and w.getnchannels()==1 and w.getsampwidth()==2
  data=w.readframes(w.getnframes())
 duration=len(data)/2/sr
 if duration>r['duration']-.3:raise RuntimeError(f'Shot {i}: speech {duration:.1f}s exceeds slot {r["duration"]}s')
 start=round((r['start']+.15)*sr)*2;audio[start:start+len(data)]=data
 words=r['voice'].split();chunks=[words[j:j+11] for j in range(0,len(words),11)]
 for j,chunk in enumerate(chunks):
  n+=1;a=r['start']+.15+duration*j/len(chunks);b=r['start']+.15+duration*(j+1)/len(chunks)
  srt.append(f'{n}\n{timestamp(a)} --> {timestamp(b)}\n'+ ' '.join(chunk)+'\n')
 print(i,round(duration,2),'/',r['duration'])
with wave.open(str(out/'narration.wav'),'wb') as w:
 w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(audio)
(out/'film-v9.en.srt').write_text('\n'.join(srt))
