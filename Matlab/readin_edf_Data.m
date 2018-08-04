
function [ECG, Resp, EMG, EOG, Chin]=readin_edf_Data(loadfolder,edfFile,plotreadin)


cd(loadfolder)

% edfFiles=dir('*.edf');
% for i=1:size(edfFiles(:,1),1) % the edf files are split into several files
%     [hdr, record] = edfread(edfFiles(i).name); % Load EDF file
    [hdr, record] = edfread(edfFile); % Load EDF file
    
    Resp_idx = find(strcmp(hdr.label, 'Chest')); % find the index of certain signal
    ECG_idx = find(strcmp(hdr.label, 'Pulse')); % find the index of certain signal
    EOG_idx = find(strcmp(hdr.label, 'EEGT5T6')); % find the index of certain signal
    Chin_idx = find(strcmp(hdr.label, 'EEGA2Ref')); % find the index of certain signal
    EMG_idx = find(strcmp(hdr.label, 'EEGP4Ref')); % find the index of certain signal
    if isempty(EMG_idx)==1
        EMG_idx = find(strcmp(hdr.label, 'EMG')); % find the index of certain signal
    end

        


    % Signal artificailly elonginated. Cut it again to find correct FS
    zResp=find(diff(record(Resp_idx,:),40)==0); %find where the signal is set to zero
    zEMG=find(diff(record(EMG_idx,:),40)==0); %find wh
    zEOG=find(diff(record(EOG_idx,:),40)==0); %find where the signal is set to zero
    zChin=find(diff(record(Chin_idx,:),40)==0); %find where the signal is set to zero


    if zResp(1,1)==1 %if the signal starts with zero search for jump 
        zResp2=find(diff(zResp)>=20); %find where there is a jump of series of zeroes in signal eg.: 00001110011111110000000000000000
        zResp=zResp(zResp2(1,1)+1);%+1 because we find the index of the start of the nans,but we want the end.That is one index further.
    end
    if zEOG(1,1)==1 %if the signal starts with zero search for jump 
        zEOG2=find(diff(zEOG)>=20); %find where there is a jump of series of zeroes in signal eg.: 00001110011111110000000000000000
        zEOG=zEOG(zEOG2(1,1)+1);
    end
    if zChin(1,1)==1 %if the signal starts with zero search for jump 
        zChin2=find(diff(zChin)>=20); %find where there is a jump of series of zeroes in signal eg.: 00001110011111110000000000000000
        zChin=zChin(zChin2(1,1)+1);
    end
    if zEMG(1,1)==1 %if the signal starts with zero search for jump 
        zEMG2=find(diff(zEMG)>=20); %find where there is a jump of series of zeroes in signal eg.: 00001110011111110000000000000000
        zEMG=zEMG(zEMG2(1,1)+1);
    end
    clearvars zEMG2 zResp2 zEOG2 zChin2

    ECG=record(ECG_idx,:); % load ECG
    Resp=(record(Resp_idx,1:zResp)); %Load Respiration
    EMG =(record(EMG_idx,1:zEMG));%Load EMG
    EOG =(record(EOG_idx,1:zEMG));%Load EOG
    Chin=(record(Chin_idx,1:zChin));%Laod Chin
   
    
    clearvars zEMG zEMG2 zResp zResp2 zEOG zEOG2 zChin zChin2

% end % for number of edf files in folder


%Interpolate to same length (FS) as ECG
FSResp=length(Resp)/(length(ECG)/500);
FSEMG=length(EMG)/(length(ECG)/500);
FSEOG=length(EOG)/(length(ECG)/500);
FSChin=length(Chin)/(length(ECG)/500);

tecg=linspace(0,length(ECG)*500,length(ECG));
tResp=linspace(0,length(Resp)*FSResp,length(Resp));
tEMG=linspace(0,length(EMG)*FSEMG,length(EMG));
tEOG=linspace(0,length(EOG)*FSEMG,length(EOG));
tChin=linspace(0,length(Chin)*FSEMG,length(Chin));

if FSResp> 50; Resp=interp1(tResp,Resp,tecg); end %interpolate the singal to same FS as ECG
if FSEMG> 200; EMG=interp1(tEMG,EMG,tecg); end %interpolate the singal to same FS as ECG
if FSEOG> 200; EOG=interp1(tEOG,EOG,tecg); end %interpolate the singal to same FS as ECG
if FSChin> 200; Chin=interp1(tChin,Chin,tecg); end %interpolate the singal to same FS as ECG

offset_Resp=(length(record(ECG_idx,:))-length(Resp))/500; % if neg resp is delayed. If pos. ECG is delayed
offset_EMG=(length(record(EMG_idx,:))-length(EMG))/500; % if neg resp is delayed. If pos. ECG is delayed
offset_EOG=(length(record(ECG_idx,:))-length(EOG))/500; % if neg EOG is delayed. If pos. ECG is delayed
offset_Chin=(length(record(EMG_idx,:))-length(Chin))/500; % if neg Chin is delayed. If pos. ECG is delayed

disp(['The resp signal offset is ' num2str(offset_Resp) 's (neg: resp. is delayed; pos: ECG is delayed)'])
disp(['The EMG signal offset is ' num2str(offset_EMG) 's (neg: EMG is delayed; pos: ECG is delayed)'])
disp(['The EOG signal offset is ' num2str(offset_EOG) 's (neg: EOG is delayed; pos: ECG is delayed)'])
disp(['The Chin EMG signal offset is ' num2str(offset_Chin) 's (neg: Chin EMG is delayed; pos: ECG is delayed)'])

% [EDR]=Respiration_from_ECG(ECG',500);
% EDR=EDR';
%Plotting
if plotreadin
    x=2;
    ax1= subplot(x,1,x-1);plot(tECG,EDR);ylabel('EDR');
    ax2= subplot(x,1,x);plot(tResp,Resp); ylabel('Resp');
    linkaxes
    
    ax1= subplot(5,1,1);plot(tECG,record(ECG_idx,:));ylabel('ECG');
    ax2= subplot(5,1,2);plot(tResp,Resp); ylabel('Resp');
    ax3= subplot(5,1,3);plot(tEOG,EOG);ylabel('EOG');
    ax4= subplot(5,1,4);plot(tChin,Chin);ylabel('Chin');
    ax5= subplot(5,1,5);plot(tEMG,EMG);ylabel('EMG');

    linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')
    % close figures
    input('press any key to close all Figures: ','s');

     h=findall(0);
     delete(h(2:end));
    delete(findall(0,'Type','figure'))
    
end
 
clearvars record path
clearvars h ax1 ax2 plot
clearvars Chin_idx divChin divEMG divEOG divResp ECG_idx EMG_idx EOG_idx Resp_idx hdr 
clearvars offset_Chin offset_EMG offset_EOG offset_Resp x
clearvars tChin tEMG tEOG tResp tECG

end
