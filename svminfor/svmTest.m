function [acc,macro_f1]=svmTest(features, gnd, train_ratio,C)

    no_samples = size(features,1);
    numOfGroup = length(unique(gnd));
    
    %gnd = gnd + 1; %index start from 1
    
    ind = 1:no_samples;
    group = sparse(ind,gnd,ones(size(ind)),no_samples,numOfGroup);

    
    grouptmp=group;
    acc=0; macro_f1 = 0;
    for i=1:size(features,2)
        if (norm(features(:,i))>0)
            features(:,i) = features(:,i)/norm(features(:,i));
        end
    end
    
    nrun = 5;
    for i=1:nrun  % do the procedure for 10 times and take the average
        disp(['Runing:', num2str(i)])
        rp = randperm(no_samples);
        testId = rp(1:floor(no_samples*(1-train_ratio)));

        groupTest = group(testId,:);
        group(testId,:)=[];

        trainId = [1:no_samples]';
        trainId(testId,:)=[];

        result=SocioDim(features, group, trainId, testId, C);
        [res b] = evaluate(result,groupTest);
        acc=acc+res.micro_F1;
        macro_f1 = macro_f1 + res.macro_F1;
        group=grouptmp;
    end
    acc=acc/nrun;
    macro_f1 = macro_f1/nrun;