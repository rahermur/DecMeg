function iid = prueba_acf(data,verbose)
aut=xcorr(data);
[maxaut idx]=max(aut);
if verbose==1
    figure
    stem(1:length(aut(idx+1:end)),aut(idx+1:end)./maxaut);
    hold on
    plot(1:length(aut(idx+1:end)),1.96/sqrt(length(data))*ones(length(aut(idx+1:end)),1),'r-',1:length(aut(idx+1:end)),-1.96/sqrt(length(data))*ones(length(aut(idx+1:end)),1),'r-');

    figure
    bar(aut(idx+1:end)./maxaut);
end

if (sum(abs(aut(idx+1:end)./maxaut)>1.96/sqrt(length(data)))<=.05*length(aut(idx+1:end)))
    iid=1;
else
    iid=0;
end
end