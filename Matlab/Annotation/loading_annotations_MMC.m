function Annotations=loading_annotations_MMC(Loadsession, loadfolderA,Session)

annotnames=dir(loadfolderA);
pause(1)
C=strsplit(Loadsession,'t'); C=C{3}; C=strsplit(C,'.'); C=C{1};%ge only the name of the file without 'patient...' and '.edf'
idx=find(strcmp({annotnames.name},[C '_Events.xls'])==1); % find the corresponding name from the annotations
if isempty(idx)
    idx=find(contains({annotnames.name},[C '_'])==1);
end

[~,Annotat,~]=xlsread([loadfolderA annotnames(idx).name], 'B:B');

Annotat=Annotat(6:end);
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
