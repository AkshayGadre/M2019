
BW = 1.25e5;
SF = 8;
chirp_size=512;
chirpsize=chirp_size;

extra_sampling_factor = 1;
Fs = BW;
symbol_length=chirp_size;
symbol_length_upsampled = extra_sampling_factor*chirp_size;
freq_shift_per_sample =  Fs/symbol_length; % How each frequency bin maps to a difference in frequency
Ts = 1/freq_shift_per_sample; % Symbol Duration
f = linspace(-BW/2,BW/2-freq_shift_per_sample,symbol_length); % The X-Axis
reset_freq = -BW/2; % The initial frequency of the base chirp
final_freq = (BW/2)-freq_shift_per_sample; % The final frequency
[up,down] = my_create_chirpspecial1(extra_sampling_factor*Fs,Ts,reset_freq,final_freq,chirp_size);
upfft=((fft(up)));
up250fft=[upfft(1:length(up)/2); zeros(length(upfft),1);upfft(length(up)/2+1:length(up))];
up250=(ifft(up250fft));
spectro(double(sample(6238:6238+512*100-1)).*repmat(conj(up250),100,1))