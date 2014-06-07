function iid = prueba_orden(data)
cont=0;
alfa=.05;
for j=2:length(data)
    for i=j-1:-1:1
        if (data(i)<data(j))
            cont=cont+1;
        end
    end
end

stand_cont=abs(cont-(.25*length(data)*(length(data)-1)))/(length(data)*(length(data)-1)*(2*length(data)+5)/8);
if stand_cont>norminv(1-alfa/2,0,1)
    iid=0;
else
    iid=1;
end
end