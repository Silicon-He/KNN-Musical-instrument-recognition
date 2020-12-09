#-*- coding = utf-8 -*-
#@Author:何欣泽
#@Time:2020/7/13 22:06
#@File:music_cut.py
#@Software:PyCharm

from pydub import AudioSegment
AudioSegment.converter = r"D:\python\ffmpeg\bin\ffmpeg.exe"
# filename = 'sax (1).mp3'
song = AudioSegment.from_mp3(r'D:\piano (1).mp3')
miaoshu = len(song)/1000
print(miaoshu)
duanshu = int(miaoshu // 30)
print(duanshu)
for i in range(0,duanshu):
    filename = 'piano' + str(i) + '.mp3'
    song_cut = song[(30 * i) *1000 : (30 * (i + 1)) * 1000]
    song_cut.export(filename,format='mp3')