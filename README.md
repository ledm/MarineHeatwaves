

Marine Heatwaves - the first cartopy music video
=================================================

This is the code used to make the 
Marine Heatwaves video:  https://youtu.be/VEKVwpnKX5w

The analysis and the video frames are generated ing the 
`calculate_timeseries.py` script.

Upon succesful completition of the analysis, an
ffmpeg  command is output, which can be copied and 
run to produce a mp4 video file. 



The Audio is generated though EarthSystemMusic2:
https://github.com/ledm/earthsystemmusic2


The `esm2.py` script passes an input yml file to 
earthsystemmusic2 which generates a MIDI output.

The MIDI is then converted into audio using a series of
VSTs in a DAW. 


The model data used is from the Mission Atlantic Project.




