%%
%THIS  M FILE CREATE FEATURE MATRICES WHERE EVERYTHING IS COMBINED
% FOR MATRICES WHERE ECG, HR, BREATHING ETC ARE SEPARATE MATRICES USE Create_single_F_and_A_Matrix_Session_and_perPatient_forDNN
%%
%This m file generates FEature and annotation Matrices. Per patient and per
%session
% Each Row is one Feature. Where each session of the same Feature


%      '	1=	ActiveSleep';...
%         '	2=	QuietSleep';...
%         '	3=	Wake';...
%         '	4=	CareTaking';...
%         '	5=	UnknownBedState'...
%           6=  Transition


%#1 How many sessions?
%#2 Go through each session and load all Featrues and the annotation
%#3 Cut annotation and Feature to the same length
%#4 Save each combination as Session in multile matrices
%#5 Merge Feature Sessions together Safe them as one matrix


clear
clc
RRMethod='R'; %M or R if Michiels or Ralphs RR peak detection method was used 
dataset='ECG'; % ECG or cECG. In the fUture mayebe MMC and InnerSense
saving=1;
win=30; % window of annotations. 30 precicse, 300 smoothed
Datapack='onlyRAW';   %all onlyECGHRV  onlyRAW  ECGHRVRR
Pat=[4,5,6,7,9,10,11,12,13];
% Pat=[7,9,10,11,12,13];

path='E:\';
if strcmp('ECG',dataset)==1
    datapath=[path 'cECG_study\C_Processed_Data\HRV_features\'];
elseif strcmp('cECG',dataset)==1
    datapath=[path 'cECG_study\C_Processed_Data\cHRV_features\'];    
end

if strcmp(RRMethod,'R')
    if strcmp('ECG',dataset)==1 && strcmp('onlyECGHRV',Datapack)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\DNN-Matrices\Matrices_onlyECGHRV\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\DNN-Matrices\Matrices_onlyECGHRV\Sessions\']);    
    elseif strcmp('ECG',dataset)==1 && strcmp('all',Datapack)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\DNN-Matrices\Matrices_all\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\DNN-Matrices\Matrices_all\Sessions\']);            
    elseif strcmp('ECG',dataset)==1 && strcmp('onlyRAW',Datapack)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\DNN-Matrices\Matrices_onlyRAW\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\DNN-Matrices\Matrices_onlyRAW\Sessions\']);        
     elseif strcmp('ECG',dataset)==1 && strcmp('ECGHRVRR',Datapack)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\DNN-Matrices\Matrices_ECGHRVRR\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\DNN-Matrices\Matrices_ECGHRVRR\Sessions\']);        
        
    elseif strcmp('cECG',dataset)==1 && strcmp('onlyECGHRV',Datapack)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\DNN-Matrices\cMatrices_onlyECGHRV\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\DNN-Matrices\cMatrices_onlyECGHRV\Sessions\']);    
    elseif strcmp('cECG',dataset)==1 && strcmp('all',Datapack)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\DNN-Matrices\cMatrices_all\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\DNN-Matrices\cMatrices_all\Sessions\']);            
    elseif strcmp('cECG',dataset)==1 && strcmp('onlyRAW',Datapack)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\DNN-Matrices\cMatrices_onlyRAW\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\DNN-Matrices\cMatrices_onlyRAW\Sessions\']);        
     elseif strcmp('cECG',dataset)==1 && strcmp('ECGHRVRR',Datapack)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\DNN-Matrices\cMatrices_ECGHRVRR\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\DNN-Matrices\cMatrices_ECGHRVRR\Sessions\']);   
    end
    TFeature_path=[datapath 'timedomain\'];
    FFeature_path=[datapath 'freqdomain\'];
    NLFeature_path=[datapath 'nonlinear\'];    
    
elseif strcmp(RRMethod,'M')
     if strcmp('ECG',dataset)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\MatricesM\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\MatricesM\Sessions\']);    

    elseif strcmp('cECG',dataset)==1
        savefolder= ([path 'cECG_study\C_Processed_Data\cMatricesM\']);
        savefolderSession=([path 'cECG_study\C_Processed_Data\cMatricesM\Sessions\']);    
     end 
    TFeature_path=[datapath 'timedomainM\'];
    FFeature_path=[datapath 'freqdomainM\'];
    NLFeature_path=[datapath 'nonlinearM\'];
    
end


loadfolderAnnotation= [path 'cECG_study\C_Processed_Data\Annotations\'];

if strcmp('all',Datapack)==1
    Featurenames_time={...
        'ECG';...
        'HRV';...
        'EDR' ;...
        'Resp';...
        'BpE';...
        'lineLength';...
        'meanlineLength';...
        'NN10'; 'NN20';'NN30';'NN50';...
        'pNN10'; 'pNN20';'pNN30';'pNN50';...
        'RMSSD';...
        'SDaLL';...
        'SDANN';...
        'SDLL';...
        'SDNN';... 
        'pDEC';...
        'SDDEC';...               
        };

    Featurenames_frequency={...
        'HF';...
        'HFnorm';...
        'LF';...
        'LFnorm';...
        'ratioLFHF';...
        'sHF';...
        'sHFnorm';...
        'totpow';...
        'uHF';...
        'uHFnorm';...
        'VLF';...
        };

    Featurenames_nonlinear={...
        'SampEn';...
        'QSE';...
        'SEAUC';...
        'LZNN';...
        'LZECG';... 
        };
elseif strcmp('onlyECGHRV',Datapack)==1
        Featurenames_time={...
        'ECG';...
        'HRV';...
        };
         Featurenames_frequency={};
         Featurenames_nonlinear={};
elseif strcmp('onlyRAW',Datapack)==1   
      Featurenames_time={...
        'ECG';...
        'HRV';...
        'EDR' ;...
        'Resp';...
        };
         Featurenames_frequency={};
         Featurenames_nonlinear={};    
elseif strcmp('ECGHRVRR',Datapack)==1  % no EDR Resp
     Featurenames_time={...
        'ECG';...
        'HRV';...
        'BpE';...
        'lineLength';...
        'meanlineLength';...
        'NN10'; 'NN20';'NN30';'NN50';...
        'pNN10'; 'pNN20';'pNN30';'pNN50';...
        'RMSSD';...
        'SDaLL';...
        'SDANN';...
        'SDLL';...
        'SDNN';... 
        'pDEC';...
        'SDDEC';...  
        };

    Featurenames_frequency={...
        'HF';...
        'HFnorm';...
        'LF';...
        'LFnorm';...
        'ratioLFHF';...
        'sHF';...
        'sHFnorm';...
        'totpow';...
        'uHF';...
        'uHFnorm';...
        'VLF';...
        };

    Featurenames_nonlinear={...
        'SampEn';...
        'QSE';...
        'SEAUC';...     
        'LZNN';...
        'LZECG';... 
        };    
end    

for N=1:length(Pat)
    disp(['Working on patient ' num2str(Pat(N))])
    disp('-------------------------------')
    Neonate=Pat(N);
    Annottmp=[];FeatureMatrix_tmp=[];
    FeatureMatrix={};tmp={};tmp2={};tmp3={};tmp4={};
%--------------- Per Patient ---------------     
%#1 How many sessions?
%#2 Go through each session and load all Featrues and the annotation
%#3 Cut annotation and Feature to the same length
%#4 Merge Feature Sessions together or safe them as session

Sessionlength=dir([TFeature_path Featurenames_time{1,1} '_Session_*_win_' num2str(win) '_*_'  num2str(Neonate) '.mat']); 
Sessionlength=length(cellfun('isempty',{Sessionlength.name}));

    for i=1:Sessionlength  
        disp(['Session ' num2str(i) '/' num2str(Sessionlength)])

        dateiname=dir([loadfolderAnnotation 'Annotations_Session_' num2str(i) '_win_' num2str(win) '_Intellivue_*_' num2str(Neonate) '.mat']);
        load([loadfolderAnnotation dateiname.name]);
        
    % all from one patient TIME DOMAIN
        for j=1:length(Featurenames_time) 
            dateiname=dir([TFeature_path Featurenames_time{j,1} '_Session_' num2str(i) '_win_' num2str(win) '_*_' num2str(Neonate) '.mat']);
            load([TFeature_path dateiname.name])
            if length(Feature)>length(Annotations)
                Feature=Feature(1:length(Annotations)); % Cut the Feature to the length of the annotation. We asume that the annotations always start at the beginning but end earlier
            elseif length(Annotations)>length(Feature)
                Annotations=Annotations(1:length(Feature));
            end
            if iscolumn(Feature)
                Feature=Feature';% Need to be a row vector
            end 
            for f=1:length(Feature)
                if iscell(Feature)==1 && isrow(Feature{1,f})
                    Feature{1,f}=Feature{1,f}';
                end
            end
            if iscell(Feature)==1 % Some features are saved in cell some as double
%                 tmp= {tmp; cell2mat(Feature)}; % adding each session of one feature to one single line of Feature 
                tmp= [tmp; Feature]; % adding each session of one feature to one single line of Feature 
                
            else
                tmp= [tmp; (num2cell(Feature))]; % adding each session of one feature to one single line of Feature 
            end             
        end
        tmp2=[tmp2,tmp]; % Adding the long single line of Feature into the Feature Matrix, where each row is one Feature        
        tmp={};

    % all from on patient FREQUENCY DOMAIN
    if strcmp('onlyECGHRV',Datapack)==1 || strcmp('all',Datapack)==1    
        for j=1:length(Featurenames_frequency) 
            dateiname=dir([FFeature_path Featurenames_frequency{j,1} '_Session_' num2str(i) '_win_' num2str(win) '_*_' num2str(Neonate) '.mat']);
            load([FFeature_path dateiname.name]);
            if length(Feature)>length(Annotations)
                Feature=Feature(1:length(Annotations)); % Cut the Feature to the length of the annotation. We asume that the annotations always start at the beginning but end earlier
            elseif length(Annotations)>length(Feature)
                Annotations=Annotations(1:length(Feature));
            end
            if iscolumn(Feature) % Need to be a row vector
                Feature=Feature';
            end         
            for f=1:length(Feature)
                if iscell(Feature)==1 && isrow(Feature{1,f})
                    Feature{1,f}=Feature{1,f}';
                end
            end
            if iscell(Feature)==1 % Some features are saved in cell some as double
%               tmp= {tmp; cell2mat(Feature)}; % adding each session of one feature to one single line of Feature 
                tmp= [tmp; Feature]; % adding each session of one feature to one single line of Feature 
            else
                tmp= [tmp; num2cell(Feature)]; % adding each session of one feature to one single line of Feature 
            end             
        end
        tmp3=[tmp3 ,tmp]; % Adding the long single line of Feature into the Feature Matrix, where each row is one Feature
        tmp={};
    
    
% all from on patient NONELINEAR    
        for j=1:length(Featurenames_nonlinear) 
            dateiname=dir([NLFeature_path Featurenames_nonlinear{j,1} '_Session_' num2str(i) '_win_' num2str(win) '_*_' num2str(Neonate) '.mat']);
            load([NLFeature_path dateiname.name])
            if length(Feature)>length(Annotations)
                Feature=Feature(1:length(Annotations)); % Cut the Feature to the length of the annotation. We asume that the annotations always start at the beginning but end earlier
            elseif length(Annotations)>length(Feature)
                Annotations=Annotations(1:length(Feature));
            end
            if iscolumn(Feature) % Need to be a row vector
                Feature=Feature';
            end
            for f=1:length(Feature) 
                if iscell(Feature)==1 && isrow(Feature{1,f})
                    Feature{1,f}=Feature{1,f}';
                end
            end            
            if iscell(Feature)==1 % Some features are saved in cell some as double
%               tmp= {tmp; cell2mat(Feature)}; % adding each session of one feature to one single line of Feature 
                tmp= [tmp; Feature]; % adding each session of one feature to one single line of Feature             
            else
                tmp= [tmp; num2cell(Feature)]; % adding each session of one feature to one single line of Feature 
            end             
        end
        tmp4=[tmp4 ,tmp]; % Adding the long single line of Feature into the Feature Matrix, where each row is one Feature
        tmp={};
    end
        FeatureMatrix=[tmp2; tmp3 ;tmp4]; % Adding the long single line of Feature into the Feature Matrix, where each row is one Feature
        tmp2={};tmp3={};tmp4={}; % Resetting 
    
        % remove total epoch if all values are nan, also in annotations to
        % not lose synch
        idx=cellfun(@(FeatureMatrix) all(isnan(FeatureMatrix)),FeatureMatrix);
        c=any(idx);% find index where one cell element(any) is all nan (row before)
        FeatureMatrix(:,c)=[];
        Annotations(:,c)=[];
        clearvars c
        % remove all nans from each cell, as there are mostly less nans than 30s, it just
        % reduces the first epoch. Deleting nans as the standart scaler of Python cannot handle nans
        for fm = 1:numel(FeatureMatrix)
            FeatureMatrix{fm} = FeatureMatrix{fm}(~isnan(FeatureMatrix{fm}),:) ;            
        end
        
        %interpolate to gain same length per Feature
        FMtmp = cell(size(FeatureMatrix));
        for k = 1:numel(FeatureMatrix)
           if numel(FeatureMatrix{k})~=1 % interpolation needs at least two values
               N = numel(FeatureMatrix{k});
               xo = 1:N;
               xi = linspace(1,N,win*500);
               FMtmp{k} = interp1(xo,FeatureMatrix{k},xi);
           else % if only one value (e.g. detected R peak ) due to noise, just create cell with this one value in the specific length 
               FMtmp{k}=kron(FeatureMatrix{k}, ones(1,win*500));
           end     
        end
        
%standart scale with (x-mean)/std 
        for F=1:size(FMtmp,1)
            flattenedM=cell2mat(FMtmp(F,:));
            MeanStd(F,1)=mean(flattenedM);
            MeanStd(F,2)=std(flattenedM);   
        end
        % Cell to 3D array
        % IN THE FOLLOWING LINE YOU CAN DETERMINE THE ORDER OF THE DATA.
        % KERAS NEEDS [SAMPLES, TIMESTEPS, FEATURES]. BUT IMPORTING THE
        % DATA TO PYTHON CHANGES THE ORDER. WITH [1,2,3] MATLAB HAS THE
        % DESIRED ORDER BUT IT WILL BE CHANGED IN PYTHON TO [FEATURES,
        % TIMESPETS, SAMPLES] , THEREFORE USE [3 2 1]
        FMtmp=permute(reshape(cell2mat(FMtmp).',numel(FMtmp{1}),size(FMtmp,2),[]),[3,2,1]);
        for sc2=1: size(FMtmp,1) % used with [3 2 1]        
%         for sc2=1: size(FMtmp,3) % used with [1 2 3]
            for sc1=1:size(FMtmp,2)
                FMtmp(sc2,sc1,:)=(FMtmp(sc2,sc1,:)-MeanStd(sc2,1))/MeanStd(sc2,2); % used with [3 2 1]                
%                 FMtmp(:,sc1,sc2)=(FMtmp(:,sc1,sc2)-MeanStd(sc2,1))/MeanStd(sc2,2); % used with [1 2 3]
            end
        end
             
        FeatureMatrix=FMtmp;
        
        if saving
            Saving_Session_F(FeatureMatrix,savefolderSession, Neonate,i)
            Saving_Session_A(Annotations,savefolderSession, Neonate,i)
        end
        
        FeatureMatrix_tmp=[FeatureMatrix_tmp FeatureMatrix];
        Annottmp=[Annottmp Annotations]; % 

    end % Session       
    Annotations=Annottmp;
    FeatureMatrix=FeatureMatrix_tmp;

if saving
    Saving_F(FeatureMatrix,savefolder, Neonate,win)
    Saving_A(Annotations,savefolder, Neonate,win)
    
end
FeatureMatrix=[];
Annotations=[];
end


%% Nested saving
    function Saving_F(FeatureMatrix,savefolder, Neonate,win)
        if exist('FeatureMatrix','var')==1
            name=inputname(1); % variable name of function input
            save([savefolder name '_' num2str(Neonate) '_win_' num2str(win)],'FeatureMatrix')
        else
            disp(['saving of ' name ' not possible'])
        end       
    end
    
    function Saving_Session_F(FeatureMatrix,savefolder, Neonate,Session)
        if exist('FeatureMatrix','var')==1
            name=inputname(1); % variable name of function input
            save([savefolder name '_' num2str(Neonate) '_Session_' num2str(Session) ],'FeatureMatrix')
        else
            disp(['saving of ' name ' not possible'])
        end       
    end
    
    function Saving_A(Annotations,savefolder, Neonate,win)
        if exist('Annotations','var')==1
            name=inputname(1); % variable name of function input
            save([savefolder name '_' num2str(Neonate) '_win_' num2str(win)],'Annotations')
        else
            disp(['saving of ' name ' not possible'])
        end       
    end    
    
    function Saving_Session_A(Annotations,savefolder, Neonate,Session)
        if exist('Annotations','var')==1
            name=inputname(1); % variable name of function input
            save([savefolder name '_' num2str(Neonate) '_Session_' num2str(Session) ],'Annotations')
        else
            disp(['saving of ' name ' not possible'])
        end       
    end
    



% end