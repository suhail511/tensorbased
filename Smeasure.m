function out = Smeasure(Fg,videolength,nrow,ncol,alpha)

m = size(alpha,2);
checking = zeros(m,1);
for t = 1 : m;
    checking(t) = ((2*t+1)^3)*0.5;
end
out = Fg;
for t=m+1:videolength-m
    for i = m+1 : ncol-m
        for j = m+1 : nrow-m
            if(Fg(i+(j-1)*nrow,t) ~= 0 )
                validity =zeros(m);
                final =0;
                for x = 1:m
                    validity(x) = layersum(x,Fg,t,i,j,nrow).*alpha(x);
                    if(validity(x) > checking(x))
                        final = 1;
                    end
                end
                if(final)
                    out(i+(j-1)*nrow,t)=0;
                end
            end
        end
    end
end

function sum = layersum(m,Fg,t,i,j,nrow)

sum =-1;
for k = 0 : ((2*m+1)^3 - 1)
    i1 = rem( floor(k / (2*m+1)^0) , (2*m+1)) - m;
    j1 = rem( floor(k / (2*m+1)^1) , (2*m+1)) - m;
    t1 = rem( floor(k / (2*m+1)^2) , (2*m+1)) - m;
    if(Fg(i+i1 + (j+j1-1) *nrow,t+t1) == 0)
        sum = sum + 1;
    end
end