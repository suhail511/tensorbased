function xhat = subLS(Pt,T,y)

%A = Pt(T,:);

It = speye(size(Pt,1));
It = It(:,T);
C = It - Pt*Pt(T,:)';

[xhat,~] = lsqr(C,y);