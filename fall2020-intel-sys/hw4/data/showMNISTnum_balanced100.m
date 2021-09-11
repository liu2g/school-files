%Reads the text version of the 1000-number MNIST data file and displays the
%first 100 in a 10X10 array.
%
%Written by: Ali Minai(3/14/16)


U = load('MNISTnumImages5000_balanced.txt');

%indexes = round(5000*rand(100,1));

indexes = [];
for j=0:9
    indexes = [indexes ; [1:10]+500*j];
end

U100 = U(indexes,:);

for i=1:10
    for j = 1:10
        v = reshape(U100((i-1)*10+j,:),28,28);
        subplot(10,10,(i-1)*10+j)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end