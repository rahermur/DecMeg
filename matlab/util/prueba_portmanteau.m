function iid = prueba_portmanteau(data)

aut=xcorr(data);
alfa=.05;
% stem(1:length(aut),aut);
% hold on
% plot(1:length(aut),1.96/sqrt(length(data))*ones(length(aut),1),'r-');

[maxaut idx]=max(aut);
Q=length(data)*sum((aut(idx+1:end)./maxaut).^2);

if Q>chi2inv(1-alfa,length(data))
    iid=0;
else
    iid=1;
end
end