pruebas = zeros(306,4,17); 
abs_err = zeros(306,125,17); 


for k = 1:16
    
subject = k;
disp(['subject ',num2str(k)])

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


    for sensor = 1:306

    trail = double(reshape(XX(:,sensor,:),size(XX,1),125));

    trail = (trail - ones(size(XX,1),1)*mean(trail,1)); 
    %%
    disp('Features Normalization.');
    coeffs = ones(1,3)/3;
    
    for i = 1 : size(trail,2)
        trail(:,i) = filter(coeffs, 1, trail(:,i));
        trail(:,i) = trail(:,i)-mean(trail(:,i));
        trail(:,i) = trail(:,i)./std(trail(:,i));
    end
%%
    faces = mean(trail(double(yy)==1,:));
    scramble_faces = mean(trail(double(yy)==0,:)); 
    abs_err(sensor,:,k) = abs(faces-scramble_faces); 

    %%
    figure(k)
    plot(faces,'b')
    hold on
    plot(scramble_faces,'r')
    plot(abs_err(sensor,:,k),'g')
    hold off

    pruebas(sensor,1) = prueba_orden(faces-scramble_faces);
    pruebas(sensor,2) = prueba_acf(faces-scramble_faces,0);
    pruebas(sensor,3) = prueba_cambio(faces-scramble_faces);
    pruebas(sensor,4) = prueba_portmanteau(faces-scramble_faces);

    disp(sensor)
    % pause;
    % close all
    end

end



%%
for j = 1:16
plot(mean(abs_err(:,:,j),2))
hold all
end
%%

yyy = mean(abs_err(:,:,:),3); 

plot(mean(yyy,2))
