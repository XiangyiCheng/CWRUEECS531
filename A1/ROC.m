clear all
y = [0.875 0.875 0.875 0.875 1 1 1 1 1 1];
x = [0 0 0 0 0 0 0.04 0.5 0.9 1];

values = spcrv([[x(1) x x(end)];[y(1) y y(end)]],3);
p1=plot(values(1,:),values(2,:),'r');
hold on

y1=[0.25 0.75 0.75 0.75 0.875 1 1 1 1 1];
x1=[0 0 0 0 0 0 0.039 0.5 0.9 1];

values1 = spcrv([[x1(1) x1 x1(end)];[y1(1) y1 y1(end)]],3);
p2=plot(values1(1,:),values1(2,:),'b');
hold on

y2=[0 0 0 0 0 0 0.125 0.75 0.875 1];
x2=[0 0 0 0 0 0 0 0 0.5 1];
values2 = spcrv([[x2(1) x2 x2(end)];[y2(1) y2 y2(end)]],3);
p3=plot(values2(1,:),values2(2,:),'m');
hold on 


xlabel('FPR')
ylabel('TPR')
set(gca,'XLim',[0 1])
set(gca,'XTick',(0:0.2:1))
set(gca,'YLim',[0 1])
set(gca,'YTick',(0:0.2:1))
title('ROC')
hold off
legend([p1;p2;p3],'original image','image with less noise','image with more noise','Location','southeast')

