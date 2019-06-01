function [] = spectro(data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
figure;
spectrogram(data,128,125,128,256,'yaxis','center');
end