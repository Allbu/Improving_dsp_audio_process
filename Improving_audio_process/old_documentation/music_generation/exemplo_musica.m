% Programa musica1.m
% Preparado por S. Scott Moor, IPFW  August 2004
% Baseado em sugest�es de Shreekanth Mandayam, 
% Departamento de Engenharias El�trica e da Computa��o, Rowan University
%   veja http://users.rowan.edu/~shreek/networks1/music.html
% Esse programa cria ondas sen�ides para s�ries de notas padr�es.
%  Cada nota � configurada para ter uma dura��o de 0.5 segundos 
% e taxa de amostragem de 8000 Hz.  As notas s�o reunidas em uma
% m�sica que ent�o � reproduzido pelo computador.
% vari�veis usadas: 
%	fa	= frequencia de amostragem (amostragens/seg)
%	t	= vetor tempo (seg.)
%	a, b, cs, d, e, & fs = a amplitude de uma s�rie de notas usadas na m�sica. 
% 	linha1, linha2, linha3 = a s�rie de amplitudes usadas para cada linha da m�sica.
%	song 	= the amplitude series for the entire song. 

% configura��o das s�ries de tempo.
fa= 16000; %frequencia de amostragem
Ta=1/fa; %periodo de amostragem
t = [0:Ta:0.5]; %eixo do tempo, discretizado, total: 0.5 s

% defina cada nota individualmente
a=sin(2*pi*440*t);
b=sin(2*pi*493.88*t);
cs=sin(2*pi*554.37*t);
d=sin(2*pi*587.33*t);
e=sin(2*pi*659.26*t);
fs=sin(2*pi*739.99*t);

% montando as notas em uma m�sica.
linha1 = [a,a,e,e,fs,fs,e,e];
linha2 = [d,d,cs,cs,b,b,a,a];
linha3 = [e,e,d,d,cs,cs,b,b];
musica = [linha1,linha2,linha3,linha3,linha1,linha2];

% agora tocando a m�sica
soundsc(musica, fa)
filterBWInHz=80; %em Hz
windowShiftInms=10; %deslocamento da janela em ms
thresholdIndB=120; %limiar, abaixo disso, nao mostra
ak_specgram(musica,filterBWInHz,fa,windowShiftInms,thresholdIndB)
colorbar