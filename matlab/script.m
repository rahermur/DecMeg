load mecg1.dat
load fecg1.dat
load noise1.dat

%Suma de las tres señales
suma=fecg1+mecg1+noise1;
%%
t=(1:1:2560)/256;
%%
subplot(4,1,1)
plot(t,mecg1)
subplot(4,1,2)
plot(t,fecg1)
subplot(4,1,3)
plot(t,noise1)
subplot(4,1,4)
plot(t,suma)

%%
%Espectro de potencia

psd_m=pwelch(mecg1);
psd_f=pwelch(fecg1);
psd_n=pwelch(noise1);
%%
subplot(3,1,1)
plot(psd_m)
axis([0 100 0 6])
title('Materna')
subplot(3,1,2)
plot(psd_f)
axis([0 200 0 6])
title('Fetal')
subplot(3,1,3)
plot(psd_n)
axis([0 100 0 30])
title('Ruido')

%%
subplot(3,1,1)
hist(mecg1)
title('Materno')
subplot(3,1,2)
hist(fecg1)
title('Fetal')
subplot(3,1,3)
hist(noise1)
title('Ruido')
hist(noise1)

%%
%FILTRO WIENER
[yhat, H] = wienerFilter(fecg1,suma);

save  yhat yhat
save H H


%%
%SEPARACIÓN DE LA SEÑAL UTILIZANDO SVD

load ('X.dat')
plot3ch(X)

% U,S,V
[U,S,V]=svd(X);

autoval1=S(1,1);
autoval2=S(2,2);
autoval3=S(3,3); 
plot3dv(V(:,1),autoval1,'r')
plot3dv(V(:,2),autoval2,'b')
plot3dv(V(:,3),autoval3,'y')

%dimensionalidad

U(:,4:end)=0;
save V V

%%
%autovalor del feto
Sfetal=S;
Sfetal(1,1)=0;
Sfetal(3,3)=0;

%%
Y=U*Sfetal*V';
%señal reconstruida
subplot(3,1,1)
plot(t,Y(:,1))
subplot(3,1,2)
plot(t,Y(:,2))
subplot(3,1,3)
plot(t,Y(:,3))

%%
subplot(3,1,1)
plot(t,U(:,1))
subplot(3,1,2)
plot(t,U(:,2))
subplot(3,1,3)
plot(t,U(:,3))
%%
stem(t,U(:,1))
stem(t,U(:,2))
stem(t,U(:,3))

save Y Y
%%
%Separación de señales utilizando ICA

[W,Z]=ica(X');
save W W
save Z Z
W_inv=inv(W);
save W_inv W_inv

plot3ch(X)

for i=1:3 
plot3dv(W_inv(:,i))
end

W_fecg=W_inv; %columna de W inversa representativa del fecg
W_fecg(:,1)=0;
W_fecg(:,3)=0;

%%
%Representar Z
subplot(3,1,1)
plot(t,Z(1,:))
subplot(3,1,2)
plot(t,Z(2,:))
subplot(3,1,3)
plot(t,Z(3,:))

%%
subplot(3,1,1)
plot(t,mecg1,'r')
hold on
plot(t,Z(1,:))
subplot(3,1,2)
plot(t,noise1,'y')
hold on
plot(t,Z(2,:))
subplot(3,1,3)
plot(t,fecg1,'g')
hold on
plot(t,Z(3,:))
%%
Z_prueba=Z;
Z_prueba(1,:)=fliplr(Z_prueba(1,:));
Z_prueba(2,:)=fliplr(Z_prueba(2,:));
Z_prueba(2,:)=-Z_prueba(2,:);
Z_prueba(3,:)=-Z(3,:);

subplot(3,1,1)
plot(t,mecg1,'r')
hold on
plot(t,Z_prueba(1,:))
subplot(3,1,2)
plot(t,noise1,'y')
hold on
plot(t,Z_prueba(2,:))
subplot(3,1,3)
plot(t,fecg1,'g')
hold on
plot(t,Z_prueba(3,:))


%%
%COMPARACIÓN

%SVD
alfa1=acos(dot(V(:,1),V(:,2))/norm(V(:,1),2)*norm(V(:,2),2));
alfa1=alfa1/pi
alfa2=acos(dot(V(:,1),V(:,3))/norm(V(:,1),2)*norm(V(:,3),2));
alfa2=alfa2/pi
alfa3=acos(dot(V(:,2),V(:,3))/norm(V(:,2),2)*norm(V(:,3),2));
alfa3=alfa3/pi

normV1=norm(V(:,1))
normV2=norm(V(:,2))
normV3=norm(V(:,3))

%%
%ICA

alfa1=acos(dot(W_inv(:,1),W_inv(:,2))/(norm(W_inv(:,1),2)*norm(W_inv(:,2),2)));
alfa1=alfa1/pi
alfa2=acos(dot(W_inv(:,1),W_inv(:,3))/(norm(W_inv(:,1),2)*norm(W_inv(:,3),2)));
alfa2=alfa2/pi
alfa3=acos(dot(W_inv(:,2),W_inv(:,3))/(norm(W_inv(:,2),2)*norm(W_inv(:,3),2)));
alfa3=alfa3/pi

normV1=norm(W_inv(:,1))
normV2=norm(W_inv(:,2))
normV3=norm(W_inv(:,3))

%%
%Comparación
load fecg2.dat
subplot(4,1,1)
plot(t,fecg2,'r'), title('Señal original')
subplot(4,1,2)
plot(t,yhat,'y'), title('Wiener')
subplot(4,1,3)
plot(t,Y(:,2),'g'), title('SVD')
subplot(4,1,4)
plot(t,Z_prueba(3,:))

%%
%corrcoef

C1=corrcoef(fecg2,yhat)
C2=corrcoef(fecg2,Y(:,2))
C3=corrcoef(fecg2,Z_prueba(3,:))