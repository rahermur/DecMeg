subject = 1; 

path = './data/';  % Specify absolute path
filename = sprintf(strcat(path,'train_subject%02d.mat'),subject);
disp(strcat('Loading ',filename));
data = load(filename);
XX = data.X;
yy = data.y;
sfreq = data.sfreq;

tmin = 0;
tmax = 0.5;
tmin_original = data.tmin;
beginning = (tmin - tmin_original) * sfreq+1;
e = (tmax - tmin_original) * sfreq;
XX = XX(:,:,beginning:e); 
%%
pruebas = zeros(306,4); 
abs_err = zeros(306,125); 

for sensor = 1:306

trail = double(reshape(XX(:,sensor,:),594,125));

%trail = (trail - ones(594)*mean(trail,1))/( ones(594)*std(trail,1)); 

faces = mean(trail(double(yy)==1,:));
scramble_faces = mean(trail(double(yy)==0,:)); 
abs_err(sensor,:) = abs(faces-scramble_faces); 

%%
figure(1)
plot(faces,'b')
hold on
plot(scramble_faces,'r')
plot(abs_err(sensor,:),'g')

pruebas(sensor,1) = prueba_orden(faces-scramble_faces);
pruebas(sensor,2) = prueba_acf(faces-scramble_faces,0);
pruebas(sensor,3) = prueba_cambio(faces-scramble_faces);
pruebas(sensor,4) = prueba_portmanteau(faces-scramble_faces);

disp(sensor)
% pause;
% close all
end


malos = find(mean(abs_err,2)<1e-13)