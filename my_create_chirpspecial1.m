function [base_chirp,base_chirp_conj] = my_create_chirpspecial1(Fs,Ts,reset_freq,final_freq,symbol_length)
t=0:Ts/(symbol_length):Ts-(Ts/(symbol_length)); 
base_chirp=chirp(t,reset_freq,t(end),final_freq,'linear',90)+...
    1i*chirp(t,reset_freq,t(end),final_freq,'linear'); 
base_chirp=reshape(base_chirp,length(base_chirp),1);
base_chirp_conj = conj(base_chirp);
end