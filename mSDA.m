function [allhx, Ws] = mSDA(xx, noise, layers, A_n)

% xx : dxn input
% noise: corruption level
% layers: number of layers to stack
% A_n: d*d structure

% allhx: (layers*d)xn stacked hidden representations

lambda = 1e-5;
disp('stacking hidden layers...')
prevhx = xx;
allhx = [];
Ws={};
for layer = 1:layers
    disp(['layer:',num2str(layer)])
	tic
	[newhx, W] = mDA(prevhx,noise,lambda,A_n);
	Ws{layer} = W;
	toc
	allhx = [allhx; newhx];
	prevhx = newhx;
end
