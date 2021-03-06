function Annotation=loading_annotations(patient, Session)
%This function loads the annotation for the patient and his session from
%"CallingHRVfunctions_for_cECG.m"

Annotators='B3A';% As Kappa is 1 we can use either of the annotations.
s = strsplit(Session,'_');  % get only session number

% Digging into the annotation folder
annotationfolder= ['E:\cECG_study\B_Annotations\' Annotators '\participant' num2str(patient)];
annotationfiles=dir([annotationfolder '\*.mat ' ]);
%Finding matching Session and annotations
for p=1:length(annotationfiles)
    as = strsplit(annotationfiles(p,1).name,'_'); `11
    tmp(p)= strcmp(as{2}, s{2}); % find which Annotation file matches the used Session
end
col=find(tmp==1); %index of the same session
    Annotation=load([annotationfolder '\' annotationfiles(col).name]);       
    Namen=fieldnames(Annotation);
    Position = Annotation.(Namen{2,1});
    Annotation=Annotation.(Namen{1,1});

end
