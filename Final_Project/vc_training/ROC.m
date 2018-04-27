clear all
y6 = [0.87 0.87 0.87 0.86 0.89 0.83 0.79 0.69 0.61 0];
x6 = [1.00 0.97 0.82 0.58 0.47 0.36 0.31 0.25 0.20 0];

y8 = [0.92 0.92 0.90 0.89 0.88 0.74 0.56 0.42 0];
x8 = [1.00 0.88 0.68 0.43 0.33 0.21 0.11 0.04 0];

y10 = [1.00 1.00 0.97 0.89 0.80 0.65 0.52 0];
x10 = [1.00 0.77 0.61 0.31 0.18 0.11 0.08 0];

y12 = [1.00 1.00 1.00 0.98 0.85 0.78 0.62 0.50 0];
x12 = [1.00 0.80 0.62 0.45 0.28 0.17 0.10 0.08 0];

y14 = [1.00 1.00 0.99 0.97 0.90 0.77 0.62 0.50 0];
x14 = [1.00 0.80 0.63 0.44 0.31 0.12 0.09 0.04 0];

y16 = [0.99 0.95 0.85 0.78 0.74 0.66 0];
x16 = [0.53 0.39 0.30 0.19 0.12 0.10 0];

y18 = [0.98 0.95 0.80 0.69 0.59 0.39 0];
x18 = [0.61 0.43 0.29 0.21 0.12 0.05 0];

y20 = [0.95 0.95 0.84 0.69 0.52 0.41 0];
x20 = [1.00 0.51 0.30 0.21 0.12 0.07 0];

% p6 = plot(x6,y6,'o');
% hold on
values6 = spcrv([[x6(1) x6 x6(end)];[y6(1) y6 y6(end)]],3);
p66=plot(values6(1,:),values6(2,:),'r');
hold on
% p8 = plot(x8,y8,'o');
% hold on
values8 = spcrv([[x8(1) x8 x8(end)];[y8(1) y8 y8(end)]],3);
p88=plot(values8(1,:),values8(2,:),'b');
hold on
% p10 = plot(x10,y10,'o');
% hold on
values10 = spcrv([[x10(1) x10 x10(end)];[y10(1) y10 y10(end)]],3);
p1010=plot(values10(1,:),values10(2,:),'m');
hold on 
% p12 = plot(x12,y12,'o');
% hold on
values12 = spcrv([[x12(1) x12 x12(end)];[y12(1) y12 y12(end)]],3);
p1212=plot(values12(1,:),values12(2,:),'g');
hold on 
% p14 = plot(x14,y14,'o');
% hold on
values14 = spcrv([[x14(1) x14 x14(end)];[y14(1) y14 y14(end)]],3);
p1414=plot(values14(1,:),values14(2,:),'k');

values16 = spcrv([[x16(1) x16 x16(end)];[y16(1) y16 y16(end)]],3);
p1616=plot(values16(1,:),values16(2,:),'c');

values18 = spcrv([[x18(1) x18 x18(end)];[y18(1) y18 y18(end)]],3);
p1818=plot(values18(1,:),values18(2,:),'y');

values20 = spcrv([[x20(1) x20 x20(end)];[y20(1) y20 y20(end)]],3);
p2020=plot(values20(1,:),values20(2,:),'--');

xlabel('FPR')
ylabel('TPR')
set(gca,'XLim',[0 1])
set(gca,'XTick',(0:0.2:1))
set(gca,'YLim',[0 1])
set(gca,'YTick',(0:0.2:1))
title('ROC')
hold off
legend([p66;p88;p1010;p1212;p1414;p1616;p1818;p2020],'6 stage','8 stage','10 stage','12 stage','14 stage','16 stage','18 stage','20 stage','Location','southeast')
