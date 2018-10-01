function acc=svmTestSamples(features, gnd, numLabel_per,C)

    no_samples = size(features,1);
    numOfGroup = length(unique(gnd));
    
    gnd = gnd; %+ 1; %index start from 1
    
    ind = 1:no_samples;
    group = sparse(ind,gnd,ones(size(ind)),no_samples,numOfGroup);

    
    grouptmp=group;
    acc=0;
    for i=1:size(features,2)
        if (norm(features(:,i))>0)
            features(:,i) = features(:,i)/norm(features(:,i));
        end
    end
    
    NoRun = 5;
    for i=1:NoRun  % do the procedure for 10 times and take the average
        disp(['Runing:', num2str(i)])
        
        %each class select fixed number of samples
        trainId = [];
        testId = [];
        for j = 1 : numOfGroup
            ids = find(gnd == j);
            ids = ids(randperm(length(ids))); % shuffle
            
            trainId = [trainId; ids(1:numLabel_per)];
            testId = [testId; ids(numLabel_per+1:end)];           
            
        end
        disp(['No of Training Samples:',num2str(length(trainId))])
        disp(['No of Test Samples:',num2str(length(testId))])

      
        groupTest = grouptmp(testId,:);
        group = grouptmp(trainId,:);


        result=SocioDim(features, group, trainId, testId, C);
        [res b] = evaluate(result,groupTest);
        acc=acc+res.micro_F1;
    end
    acc=acc/NoRun;