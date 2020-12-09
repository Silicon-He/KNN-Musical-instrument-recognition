i = 1
low = []
high = []
all = []
while(i < 70)
    filename = sprintf('sax (%d).mp3',i);
    
    i = i + 1;
    
    [x,fs]=audioread(filename);%��ȡ��Ƶ����
    
    x = x(:,1);
    
    x = x';
    
    N = length(x);%��ȡ��������
    
    t = (0:N-1)/fs;%��ʾʵ��ʱ��
    
    y = fft(x);%���źŽ��и���Ҷ�任
                     
    y = abs(y(1:round(N/10)))%0-4500
    
    y_l = abs(y(1:round(N/10)/45*10))%0-1000

    y_h = y(round(N/10)*4/9:round(N/10)-1)%2000-4500

    pinjun_l = mean(y_l)%0-1000ƽ����
    
    pinjun = mean(y)

    pinjun_h = mean(y_h) 
    
    low = [low,pinjun_l]
    
    high = [high,pinjun_h]
    
    all = [all,pinjun]
end
low = low';
high = high';
all = all';
data = [low high all]

xlswrite('D:\result_violin.xlsx',data,'violin');

