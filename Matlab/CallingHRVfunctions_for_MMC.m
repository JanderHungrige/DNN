%Calling RHV analysis functions

% About Patient Nr4. Session 2 (1341399361) does not have Intellivue data. Therefore,
% create DAQ data for that particular session(or pat 4 in total) rename it
% to Intellivue manually and do the same with the annotations(if total 4
% delete the others). Then you can create the matrix without lost data(6h).
% You cannot simply use the DAQ data as there are annotations missing and
% to correct for that is more difficult. 
%
% For single addition: outcomment the for loop command and end command for the sessions loop ( Line 75) and fill in e.g. S=2

clear
clc
tic
PatientID=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; % core. Show all patients in the folder
pat=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
pat=22
user='c3po'; % c3po Philips

RRMethod='R'; %M or R to calculate the RR with Michiel or Ralphs algorythm
saving=1
plotting1=0; % raw signals
plotting=0; % R peaks etc.
win=30;
faktor=30; % how much is the data moving forward? 30s is classic
FS_ecg=500;
S=1;
shutdown=0
onlyAnnotations=0;

Pat_weight=[1180,1180,1290,1290,1020,1020,1310,1310,1250,1070,1070,1880,1130,1050,1230,890,920,1050,600,600,790,680];
Pat_GA=[29*7+6,29*7+6,29*7+1,29*7+1,29*7+1,29*7+1,29*7+2,29*7+2,27*7+5,26*7+4,26*7+6,30*7+3,28*7+6,28*7+4,27*7+3,31*7+4,30*7+3,28*7+4,27*7+6,27*7+6,29*7+6,26*7+2]; %pat 9 does not have any
Pat_CA=[31*7+3,34*7+3,30*7+0,32*7+0,30*7+1,35*7+1,30*7+5,34*7+0,32*7+6,32*7+5,33*7+7,33*7+1,33*7+0,35*7+1,33*7+7,33*7+5,33*7+2,31*7+0,33*7+1,34*7+1,35*7+0,28*7+1];
Pat_GACA=Pat_CA-Pat_GA; % difference between the birth and recording of data
NICU_info=[];% NICU=1, NMCU=2
C02=[1,2,1,2,1,2,1,1,2,1,1,2,2,1,1,2,2,1,2,2,2,1]; %1=CPAP, 2=lowflow 3=no

% Order AGe and weight after 10, 25 and 75 percentiles
Pat_weight(Pat_weight<=765)=1;
Pat_weight(Pat_weight>765 & Pat_weight<=1050)=2;
Pat_weight(Pat_weight>1050 & Pat_weight<=1665)=3;
Pat_weight(Pat_weight>1665)=4;     

Pat_GA(Pat_GA<=26.9143*7)=1;
Pat_GA(Pat_GA>26.9143*7 & Pat_GA<=27.8571*7)=2;
Pat_GA(Pat_GA>27.8571*7 & Pat_GA<=30.4286*7)=3;
Pat_GA(Pat_GA>30.4286*7)=4;  

Pat_CA(Pat_CA<=30.0571*7)=1;
Pat_CA(Pat_CA>30.0571*7 & Pat_CA<=31.2857 *7)=2;
Pat_CA(Pat_CA>31.2857*7 & Pat_CA<=34.6429*7)=3;
Pat_CA(Pat_CA>34.6429*7)=4;  

Pat_GACA(Pat_GACA<=0.8571*7)=1;
Pat_GACA(Pat_GACA>0.8571*7 & Pat_GACA<=1.4286  *7)=2;
Pat_GACA(Pat_GACA>1.4286 *7 & Pat_GACA<=5.8214*7)=3;
Pat_GACA(Pat_GACA>5.8214*7)=4;  

if strcmp(user,'c3po')
    basepath='C:\Users\C3PO';
    basepath2='D:\PhD\Article_4_(MMC)';
elseif strcmp(user,'Philips')
    basepath='C:\Users\310122653';
end
cd([basepath '\Documents\GitHub\DNN\Matlab'])
addpath([basepath '\Documents\GitHub\DNN\Matlab'])
addpath([basepath '\Documents\GitHub\Joined_Matlab'])
addpath([basepath '\Documents\GitHub\Joined_Matlab\HRV feature creation\'])
addpath([basepath '\Documents\GitHub\Joined_Matlab\ECG feature creation'])
addpath([basepath '\Documents\GitHub\Joined_Matlab\R peak detection'])
addpath([basepath '\Documents\GitHub\DNN\Matlab\Create data'])
addpath([basepath '\Documents\GitHub\DNN\Matlab\Annotation'])
% Load EDF files 
addpath([basepath '\Documents\GitHub\Joined_Matlab\read edf files\Without annot'])
addpath([basepath '\Documents\GitHub\Joined_Matlab\read edf files\With annot'])

if strcmp (user,'c3po')
    loadfolder=([basepath2 '\Raw Data\Data\']);
    loadfolderA=([basepath2 '\Raw Data\Annotations\']);
    savefolder= ([basepath2 '\Processed data\']);
    SavefolderAnnotations=([ savefolder 'Annotations\']);    
elseif strcmp(user,'Philips')
    loadfolder=([basepath '\Documents\PhD\Article_4_(MMC)\Data\']);
    loadfolderA=([basepath '\Documents\PhD\Article_4_(MMC)\Annotations\']);
    savefolder= ([basepath '\Documents\PhD\Article_4_(MMC)\Processed data\']);
    SavefolderAnnotations=([ savefolder 'Annotations\']);
end


if (exist([savefolder 'HRV_features\RR\']) )==0;  mkdir([savefolder 'HRV_features\RR\']);end
if (exist([savefolder 'HRV_features\EDR\']) )==0;  mkdir([savefolder 'HRV_features\EDR\']);end
if (exist([savefolder 'HRV_features\timedomain\']) )==0;  mkdir([savefolder 'HRV_features\timedomain\']);end
if (exist([savefolder 'HRV_features\freqdomain\']) )==0;  mkdir([savefolder 'HRV_features\freqdomain\']);end
if (exist([savefolder 'HRV_features\nonlinear\']) )==0;   mkdir([savefolder 'HRV_features\nonlinear\']);end
if (exist([savefolder 'HRV_features\weigthAge\']) )==0;   mkdir([savefolder 'HRV_features\weigthAge\']);end    

savefolderECG= ([ savefolder 'HRV_features\ECG\']);
savefolderRR= ([ savefolder 'HRV_features\RR\']);
savefolderEDR= ([ savefolder 'HRV_features\EDR\']);    
savefolderHRVtime= ([ savefolder 'HRV_features\timedomain\']);
savefolderHRVfreq= ([ savefolder 'HRV_features\freqdomain\']);        
savefolderHRVnonlin= ([ savefolder 'HRV_features\nonlinear\']);  
savefolderHRVweightAge=([savefolder 'HRV_features\weigthAge\']);



 for I=1:length(pat)
    disp('***************************************')
    disp(['Working on patient ' num2str(pat(I))])
    Neonate=pat(I);   
    N_I=find(PatientID==Neonate); % IF we do not start with 1 we have to choose the correct index
    Sessions.name=num2str(Neonate);
  
%% ************ Load data **************
    Loadsession=dir(loadfolder);Loadsession=Loadsession(3:end,:);
    if onlyAnnotations ~=1
        [ECG,Resp,EMG, EOG, Chin]=readin_edf_Data(loadfolder,[loadfolder Loadsession(N_I).name],plotting1);
    end
  

%% ************ Load annotations (1s) **************  

    Annotation=loading_annotations_MMC(Loadsession(N_I).name, loadfolderA);
    disp('* Annotation loaded')
    if saving % saving annotations
       Annotations=num2cell(Annotation');% In the cECG set the annotations where on 1s base. Therfore they needed to be cut into 30s. MMC data is based on 30s
       SavingAnnotations(Annotations,SavefolderAnnotations, Neonate, win)
       disp('* Annotation saved')
    end

 %% ************ Window  ECG /  Annotation signals 
    if onlyAnnotations ~=1    
        t_ECG=linspace(0,length(ECG)/FS_ecg,length(ECG))';
%         t_EDR=linspace(0,length(EDR)/FS_ecg,length(EDR))';   
    %       
        % The differnec in t_300 and t_ECG_300 is that t_ECG_300 is a
        % continuous run of time, while t_300 is 0 to t for each cell element
       [ECG_win_300,ECG_win_30,t_ECG_300,t_ECG_30]=SlidingWindow_ECG(ECG,t_ECG,Neonate,1,savefolderECG,faktor); 
%        [Resp_win_300,Resp_win_30,t_Resp_300,t_Resp_30]=SlidingWindow_ECG(Resp.values,t_Resp,Neonate,0,savefolder,faktor); 
%        [EDR_win_300,EDR_win_30,t_EDR_300,t_EDR_30]=SlidingWindow_EDR(EDR,t_EDR,Neonate,1,savefolderEDR,faktor,win); 

        disp(['* Data is merged into windows of length: ' num2str(win) 's and ' num2str(30) 's'] )  
    end
    
    if onlyAnnotations
        continue 
    end
    %% ************ Creating RR signal for ECG-Signal **************
    Ralphsfactor=1;%{1;1;1;1;1;-1;1;-1; 1; 1;-1; 1;-1; 1;-1;-1;-1;-1};%Determine if the ECG signal should be turned -1 or not 1. 
                 %1  2 3  4 5  6 7  8  9  10 11 12 13 14 15 16 17 18
    padding=0; %Determine if the RR should be same length as ECG. Don`t have to be

%Ralph            
    for R=1:length(ECG_win_300)
        t_300{1,R}=linspace(0,length(ECG_win_300{1,R})/FS_ecg,length(ECG_win_300{1,R}))';
        if all(isnan(ECG_win_300{1,R}))==1 || range(ECG_win_300{1,R})==0 || length(unique(ECG_win_300{1,R}))<=4  % if only Nan Ralph cannot handle it or if all values are the same (Flat line)
           RR_300{R,1}=NaN(1,length(ECG_win_300{1,R})) ;
           RR_idx_300{R,1}=NaN;
        else
            [RR_idx_300{R,1}, ~, ~, ~, ~, RR_300{R,1}, ~] = ecg_find_rpeaks(t_300{1,R},Ralphsfactor*ECG_win_300{1,R}, FS_ecg, 250,plotting,0); %, , , maxrate,plotting,saving   -1* because Ralph optimized for a step s slope, we also have steep Q slope. inverting fixes that probel 
        end
    end
    for R=1:length(ECG_win_30)  
        t_30{1,R}=linspace(0,length(ECG_win_30{1,R})/FS_ecg,length(ECG_win_30{1,R}))';        
        if all(isnan(ECG_win_30{1,R}))==1 || range(ECG_win_30{1,R})==0 || length(unique(ECG_win_30{1,R}))<=4 % if all elements are NAN or the same value, R peaks cannot be calculated
           RR_30{R,1}=NaN(1,length(ECG_win_30{1,R})) ;
           RR_idx_30{R,1}=NaN;
        else        
            [RR_idx_30{R,1}, ~, ~, ~, ~, RR_30{R,1}, ~] = ecg_find_rpeaks(t_30{1,R},Ralphsfactor*ECG_win_30{1,R}, FS_ecg, 250,plotting,0); %, , , maxrate,plotting,saving   -1* because Ralph optimized for a step s slope, we also have steep Q slope. inverting fixes that probel             
        end
    end        
        
    disp('* RR calcuated')

    if saving
       Saving(RR_30,savefolderRR, Neonate, win)
       Saving(RR_300,savefolderRR, Neonate, win)           
       disp('* RR saved')
    end
        
    %% ************ Creating EDR signal from 30s epoch ECG **************
    [EDR_30] =Respiration_from_ECG(ECG_win_30,RR_idx_30,RR_30,500);       
    [EDR_300]=Respiration_from_ECG(ECG_win_300,RR_idx_300,RR_300,500);   

    if saving
       Saving(EDR_30,savefolderEDR, Neonate, win)
       Saving(EDR_300,savefolderEDR, Neonate, win)           
       disp('* EDR saved')
    end        

    % ************ Creating spectrum for ECG-Signal **************         
       [powerspectrum,f]=Lomb_scargel_single(RR_300,RR_idx_300,t_300) ;
       [powerspectrumEDR,fEDR]=Lomb_scargel_single(EDR_300,RR_idx_300,t_300) ;
       disp('* Periodogram calculated')
        if saving   
            Saving(powerspectrum,savefolderHRVfreq, Neonate, win)
            Saving(powerspectrumEDR,savefolderHRVfreq, Neonate, win)
            disp('* Spectrums saved')
        end
        
    %%  ************ AGE & Weight **************    
    for k=1:length(RR_30)
        Birthweight{k}=Pat_weight(N_I);
        GA{k}=Pat_GA(N_I); 
        CA{k}=Pat_CA(N_I);
        Age_diff{k}=Pat_GACA(N_I);
    end
    if saving
        Saving(Birthweight,savefolderHRVweightAge, Neonate, win)
        Saving(GA,savefolderHRVweightAge, Neonate, win)
        Saving(CA,savefolderHRVweightAge, Neonate, win)
        Saving(Age_diff,savefolderHRVweightAge, Neonate, win)        
        disp('* Age and Weight saved')
    end  
    clearvars Birthweight GA CA Age_diff
    %% ************ CALCULATE FEATURES **************
    %%%%%%%% FULL SIGNALS 
        disp('Full Signal analysis start') 
    
          ECG_HRV_power(powerspectrum,RR_30,ECG_win_30,RR_300,ECG_win_300,Neonate,saving,savefolderHRVtime,win,Sessions(S,1).name,S)
             disp('- totalECG finished')
          if strcmp('ECG',dataset)==1
            Resp_EDR(Resp_win_300,Resp_win_30,EDR_win_300,EDR_win_30,Neonate,saving,savefolderHRVtime,win,Sessions(S,1).name,S)    
             disp('- EDR finished')
          end
    %%%%%%% ECG TIME DOMAIN     
        disp('ECG time domain analysis start') 
        
        BpE=Beats_per_Epoch(RR_300);Saving(BpE,savefolderHRVtime, Neonate, win)   % S for session number
             disp('- BpE finished')
        LL=linelength(ECG_win_300,t_300);Saving(LL,savefolderHRVtime, Neonate, win)   
             disp('- Linelength finished')
        aLL=meanarclength(ECG_win_30,t_30,faktor,win);Saving(aLL,savefolderHRVtime, Neonate, win)
             disp('- Mean linelength finished')
        SDLL=SDLL_F(ECG_win_30,t_30,faktor,win);Saving(SDLL,savefolderHRVtime, Neonate, win) %Standart derivation of 5min linelength
             disp('- SDLL finsihed')
        SDaLL=SDaLL_F(ECG_win_30,t_30,faktor,win);Saving(SDaLL,savefolderHRVtime, Neonate, win) %Standart derivation of 30s linelength meaned over 5min
             disp('- SDaLL finished') 

  %%%%%%%% HRV TIME DOMAIN
        disp('HRV time domain analysis start')

        SDNN=SDNN_F(RR_300);Saving(SDNN,savefolderHRVtime, Neonate, win)
            disp('- SDNN finished') 
        RMSSD=RMSSD_F(RR_300);Saving(RMSSD,savefolderHRVtime, Neonate, win)
            disp('- RMSSD finished')  
        [NN50,NN30,NN20,NN10]=NNx(RR_300);Saving(NN50,savefolderHRVtime, Neonate, win);Saving(NN30,savefolderHRVtime, Neonate, win);Saving(NN20,savefolderHRVtime, Neonate, win);Saving(NN10,savefolderHRVtime, Neonate, win)
            disp('- NNx finished') 
        [pNN50,pNN30,pNN20,pNN10]=pNNx(RR_300);Saving(pNN50,savefolderHRVtime, Neonate, win);Saving(pNN30,savefolderHRVtime, Neonate, win);Saving(pNN20,savefolderHRVtime, Neonate, win);Saving(pNN10,savefolderHRVtime, Neonate, win)
            disp('- pNNx finished') 
        SDANN=SDANN_F(RR_30,faktor,win);Saving(SDANN,savefolderHRVtime, Neonate, win)
            disp('- SDANN finished')
        pDec=pDec_F(RR_300);Saving(pDec,savefolderHRVtime, Neonate, win)
            disp('- pDEC finished') 
        SDDec=SDDec_F(RR_300);Saving(SDDec,savefolderHRVtime, Neonate, win)
           disp('- SDDec finished')
           
  %%%%%%% HRV Frequency domain
        disp('Frequency time domain start')

           [totpow,VLF,LF,LFnorm,HF,HFnorm,ratioLFHF,sHF,sHFnorm,uHF,uHFnorm]=...
        freqdomainHRV (powerspectrum,f);Saving(totpow,savefolderHRVfreq, Neonate, win); Saving(VLF,savefolderHRVfreq, Neonate, win); Saving(LF,savefolderHRVfreq, Neonate, win); Saving(LFnorm,savefolderHRVfreq, Neonate, win);...
                                        Saving(HF,savefolderHRVfreq, Neonate, win);Saving(HFnorm,savefolderHRVfreq, Neonate, win); Saving(ratioLFHF,savefolderHRVfreq, Neonate, win); Saving(sHF,savefolderHRVfreq, Neonate, win);...
                                        Saving(sHFnorm,savefolderHRVfreq, Neonate, win); Saving(uHF,savefolderHRVfreq, Neonate, win); Saving(uHFnorm,savefolderHRVfreq, Neonate, win)
           disp('- Frequency finished') 
           [totpowR,LFR,LFnormR,HFR,HFnormR,ratioLFHFR,MFR,MFnormR,ratioMFHFR]=...
        freqdomainEDR (powerspectrumEDR,fEDR);Saving(totpowR,savefolderHRVfreq, Neonate, win); Saving(LFR,savefolderHRVfreq, Neonate, win); Saving(LFnormR,savefolderHRVfreq, Neonate, win); Saving(HFR,savefolderHRVfreq, Neonate, win);...
                                           Saving(HFnormR,savefolderHRVfreq, Neonate, win); Saving(ratioLFHFR,savefolderHRVfreq, Neonate, win); Saving(MFR,savefolderHRVfreq, Neonate, win); Saving(MFnormR,savefolderHRVfreq, Neonate, win);  Saving(ratioMFHFR,savefolderHRVfreq, Neonate, win)

           disp('- EDR requency finished')           

    %%%%%%% HRV Non linear
        disp('Nonlinear analysis start')

%         [SampEn,QSE,SEAUC,r_opt]=SampEn_QSE_SEAUC(RR_300,faktor);Saving(SampEn,savefolderHRVnonlin, Neonate, win); Saving(QSE,savefolderHRVnonlin, Neonate, win);Saving(SEAUC,savefolderHRVnonlin, Neonate, win);Saving(r_opt,savefolderHRVnonlin, Neonate, win)
%           disp('- SampEn QSE SEAUC finished')
%         LZECG=LempelZivECG(ECG_win_300); Saving(LZECG,savefolderHRVnonlin, Neonate, win )
%           disp('- LepelZiv ECG finished')         
%         LZNN=LempelZivRR(RR_300); Saving(LZNN,savefolderHRVnonlin, Neonate, win)
          disp('- LepelZiv HRV finished')   
        [SampEn_EDR,QSE_EDR,SEAUC_EDR,r_opt_EDR]=SampEn_QSE_SEAUC(EDR_300,faktor);Saving(SampEn_EDR,savefolderHRVnonlin, Neonate, win);Saving(QSE_EDR,savefolderHRVnonlin, Neonate, win);Saving(SEAUC_EDR,savefolderHRVnonlin, Neonate, win);Saving(r_opt_EDR,savefolderHRVnonlin, Neonate, win)  
          disp('- SampEn_EDR QSE_EDR SEAUC_EDR finished')          
        LZEDR=LempelZivEDR(EDR_300); Saving(LZEDR,savefolderHRVnonlin, Neonate, win)
          disp('- LepelZiv EDR finished')  


        clearvars ECG_win_300 ECG_win_30 t_ECG_300 t_ECG_30 RR_idx_300 RR_300 RR_idx_30 RR_30 powerspectrum f powerspectrumEDR fEDR ECG Resp EMG EOG Chin...
            EDR_300 EDR_30 t_30 t_300
            
        
%     end %Sessionp
 end% Patient
 disp('----------------------------------')
 t1 = toc;
 dur=datestr(t1/(24*60*60),'DD:HH:MM:SS');
 disp('Finished' )
 disp (['Duration: ' dur]);  


 if shutdown
     pause('on');
     pause(20);
     system('shutdown -s')
 end
 
 %% Nested saving
    function Saving(Feature,savefolder, Neonate, win)
        if exist('Feature','var')==1
            name=inputname(1); % variable name of function input
            save([savefolder name '_win_' num2str(win) '_' num2str(Neonate)],'Feature')
        else
            disp(['saving of ' name ' not possible'])
        end       
    end
    function SavingAnnotations(Annotations,savefolder, Neonate, win)
        if exist('Annotations','var')==1
            name=inputname(1); % variable name of function input
            save([savefolder name '_win_' num2str(win) '_' num2str(Neonate)],'Annotations')
        else
            disp(['saving of ' name ' not possible'])
        end       
    end
    function SavingF(Feature,savefolder, Neonate, win,Session,S)
        if exist('Feature','var')==1
            name=inputname(1); % variable name of function input
            save([savefolder name '_Session_' num2str(S) '_win_' num2str(win) '_' Session],'Feature')
        else
            disp(['saving of ' name ' not possible'])
        end       
    end