function iid = prueba_cambio(data)

cont=0;
alfa=.05;
for n=2:length(data)-1
    p1=data(n)>data(n-1);
    p2=data(n)>data(n+1);
    if p1==p2
        cont=cont+1;
    end
end
stand_cont=abs(cont-(2/3*(length(data)-2)))/((16*length(data)-29)/90);
if stand_cont>norminv(1-alfa/2,0,1)
    iid=0;
else
    iid=1;
end
end