function Annotations=loading_annotations_MMC(Loadsession, loadfolderA,Session,user)

annotnames=dir(loadfolderA);
pause(1)
C=strsplit(Loadsession,'t'); C=C{3}; C=strsplit(C,'.'); C=C{1};%ge only the name of the file without 'patient...' and '.edf'
idx=find(strcmp({annotnames.name},[C '_Events.xls'])==1); % find the corresponding name from the annotations
if isempty(idx)
    idx=find(contains({annotnames.name},[C '_'])==1);
end

if strcmp(user,'c3po')
    Annotat=readtable([loadfolderA annotnames(idx).name]);
    Annotat=table2array(Annotat);
    Annotat=Annotat(6:end,2);
end
if strcmp(user,'Philips')
    [~,Annotat,~]=xlsread([loadfolderA annotnames(idx).name], 'B:B');
    if size(Annotat,2)==1
        Annotat=Annotat(6:end);
    else
        Annotat=Annotat(6:end,2); %6 or 7 ???chekc
    end
end
idxT=find(strcmp(Annotat,'T'));
idxR=find(strcmp(Annotat,'R'));
idxN=find(strcmp(Annotat,'N'));
idxW=find(strcmp(Annotat,'W'));

Annotations=zeros(length(Annotat),1 );
Annotations(idxT)=6;
Annotations(idxR)=1;
Annotations(idxN)=2;
Annotations(idxW)=3;

end
